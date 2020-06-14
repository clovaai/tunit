"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license 
"""
import argparse
import warnings
from datetime import datetime
from glob import glob
from shutil import copyfile
from collections import OrderedDict

import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from models.generator import Generator as Generator
from models.discriminator import Discriminator as Discriminator
from models.guidingNet import GuidingNet
from models.inception import InceptionV3

from train.train_unsupervised import trainGAN_UNSUP
from train.train_semisupervised import trainGAN_SEMI
from train.train_supervised import trainGAN_SUP

from validation.validation import validateUN, calcFIDBatch
from validation.plot_tsne import plot_tSNE

from tools.utils import *
from datasets.datasetgetter import get_dataset
from tools.ops import initialize_queue

from tensorboardX import SummaryWriter

# Configuration
parser = argparse.ArgumentParser(description='PyTorch GAN Training')
parser.add_argument('--dataset', default='animal_faces', help='Dataset name to use',
                    choices=['afhq_cat', 'afhq_dog', 'afhq_wild', 'animal_faces', 'photo2ukiyoe', 'summer2winter', 'lsun_car', 'ffhq'])
parser.add_argument('--data_path', type=str, default='../data',
                    help='Dataset directory. Please refer Dataset in README.md')
parser.add_argument('--workers', default=4, type=int, help='the number of workers of data loader')

parser.add_argument('--model_name', type=str, default='GAN',
                    help='Prefix of logs and results folders. '
                         'ex) --model_name=ABC generates ABC_20191230-131145 in logs and results')

parser.add_argument('--epochs', default=200, type=int, help='Total number of epochs to run. Not actual epoch.')
parser.add_argument('--iters', default=1000, type=int, help='Total number of iterations per epoch')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--val_batch', default=10, type=int,
                    help='Batch size for validation. '
                         'The result images are stored in the form of (val_batch, val_batch) grid.')
parser.add_argument('--log_step', default=100, type=int)

parser.add_argument('--sty_dim', default=128, type=int, help='The size of style vector')
parser.add_argument('--output_k', default=10, type=int, help='Total number of classes to use')
parser.add_argument('--img_size', default=128, type=int, help='Input image size')
parser.add_argument('--dims', default=2048, type=int, help='Inception dims for FID')

parser.add_argument('--p_semi', default=0.0, type=float,
                    help='Ratio of labeled data '
                         '0.0 = unsupervised mode'
                         '1.0 = supervised mode'
                         '(0.0, 1.0) = semi-supervised mode')

parser.add_argument('--load_model', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)'
                         'ex) --load_model GAN_20190101_101010'
                         'It loads the latest .ckpt file specified in checkpoint.txt in GAN_20190101_101010')
parser.add_argument('--validation', dest='validation', action='store_true',
                    help='Call for valiation only mode')

parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU id to use.')
parser.add_argument('--ddp', dest='ddp', action='store_true', help='Call if using DDP')
parser.add_argument('--port', default='8989', type=str)

parser.add_argument('--iid_mode', default='iid+', type=str, choices=['iid', 'iid+'])

parser.add_argument('--w_gp', default=10.0, type=float, help='Coefficient of GP of D')
parser.add_argument('--w_rec', default=0.1, type=float, help='Coefficient of Rec. loss of G')
parser.add_argument('--w_adv', default=1.0, type=float, help='Coefficient of Adv. loss of G')
parser.add_argument('--w_vec', default=0.01, type=float, help='Coefficient of Style vector rec. loss of G')


def main():
    ####################
    # Default settings #
    ####################
    args = parser.parse_args()
    print("PYTORCH VERSION", torch.__version__)
    args.data_dir = args.data_path
    args.start_epoch = 0

    assert (args.p_semi >= 0.0) and (args.p_semi <= 1.0)

    # p_semi = 0.0 : unsupervised
    # p_semi = 1.0 : supervised
    # p_semi = 0.0~1.0 : semi-
    if args.p_semi == 0.0:
        args.train_mode = 'GAN_UNSUP'
    elif args.p_semi == 1.0:
        args.train_mode = 'GAN_SUP'
    else:
        args.train_mode = 'GAN_SEMI'

    den = args.iters//1000

    # unsup_start : train networks with supervised data only before unsup_start
    # separated : train IIC only until epoch = args.separated
    # ema_start : Apply EMA to Generator after args.ema_start
    if args.train_mode in ['GAN_SEMI']:
        args.unsup_start = 20
        args.separated = 35
        args.ema_start = 36
        args.fid_start = 36
    elif args.train_mode in ['GAN_UNSUP']:
        args.unsup_start = 0
        args.separated = 65
        args.ema_start = 66
        args.fid_start = 66
    elif args.train_mode in ['GAN_SUP']:
        args.unsup_start = 0
        args.separated = 0
        args.ema_start = 1
        args.fid_start = 1

    args.unsup_start = args.unsup_start // den
    args.separated = args.separated // den
    args.ema_start = args.ema_start // den
    args.fid_start = args.fid_start // den

    # Cuda Set-up
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.multiprocessing_distributed = False

    if len(args.gpu) > 1:
        args.multiprocessing_distributed = True
    print(args.multiprocessing_distributed)
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print(args.distributed)

    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node

    print("MULTIPROCESSING DISTRIBUTED : ", args.multiprocessing_distributed)

    # Logs / Results
    if args.load_model is None:
        args.model_name = '{}_{}'.format(args.model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        args.model_name = args.load_model

    makedirs('./logs')
    makedirs('./results')

    args.log_dir = os.path.join('./logs', args.model_name)
    args.event_dir = os.path.join(args.log_dir, 'events')
    args.res_dir = os.path.join('./results', args.model_name)

    makedirs(args.log_dir)
    dirs_to_make = next(os.walk('./'))[1]
    not_dirs = ['.idea', '.git', 'logs', 'results', '.gitignore', '.nsmlignore', 'resrc']
    makedirs(os.path.join(args.log_dir, 'codes'))
    for to_make in dirs_to_make:
        if to_make in not_dirs:
            continue
        makedirs(os.path.join(args.log_dir, 'codes', to_make))
    makedirs(args.res_dir)

    if args.load_model is None:
        pyfiles = glob("./*.py")
        for py in pyfiles:
            copyfile(py, os.path.join(args.log_dir, 'codes') + "/" + py)

        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            tmp_files = glob(os.path.join('./', to_make, "*.py"))
            for py in tmp_files:
                copyfile(py, os.path.join(args.log_dir, 'codes', py[2:]))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    if len(args.gpu) == 1:
        args.gpu = 0
    else:
        args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:'+args.port,
                                world_size=args.world_size, rank=args.rank)

    if '2' in args.dataset:
        args.output_k = 2

    # # of GT-classes
    args.num_cls = args.output_k

    # Classes to use
    if args.dataset == 'animal_faces':
        args.att_to_use = [11, 43, 56, 74, 89, 128, 130, 138, 140, 141]
    elif args.dataset == 'afhq_cat':
        args.att_to_use = [0, ]
    elif args.dataset == 'afhq_dog':
        args.att_to_use = [1, ]
    elif args.dataset == 'afhq_wild':
        args.att_to_use = [2, ]
    elif '2' in args.dataset:
        args.att_to_use = [0, 1]
        assert args.num_cls == len(args.att_to_use)
    elif args.dataset in ['ffhq', 'lsun_car']:
        args.att_to_use = [0, ]

    # IIC statistics
    args.epoch_acc = []
    args.epoch_avg_subhead_acc = []
    args.epoch_stats = []

    # Logging
    logger = SummaryWriter(args.event_dir)

    # build model - return dict
    networks, opts = build_model(args)

    # load model if args.load_model is specified
    load_model(args, networks, opts)
    cudnn.benchmark = True

    # get dataset and data loader
    train_dataset, val_dataset = get_dataset(args.dataset, args)
    train_loader, val_loader, train_sampler = get_loader(args, {'train': train_dataset, 'val': val_dataset})

    # map the functions to execute - un / sup / semi-
    trainFunc, validationFunc = map_exec_func(args)

    queue_loader = train_loader['UNSUP'] if 0.0 < args.p_semi < 1.0 else train_loader

    queue = initialize_queue(networks['C_EMA'], args.gpu, queue_loader, feat_size=args.sty_dim)

    # print all the argument
    print_args(args)

    # All the test is done in the training - do not need to call
    if args.validation:
        validationFunc(val_loader, networks, 999, args, {'logger': logger, 'queue': queue})
        return

    # For saving the model
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        record_txt = open(os.path.join(args.log_dir, "record.txt"), "a+")
        for arg in vars(args):
            record_txt.write('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))
        record_txt.close()

    # Run
    validationFunc(val_loader, networks, 0, args, {'logger': logger, 'queue': queue})

    fid_best_ema = 999.0

    for epoch in range(args.start_epoch, args.epochs):
        print("START EPOCH[{}]".format(epoch+1))
        if (epoch + 1) % (args.epochs // 10) == 0:
            save_model(args, epoch, networks, opts)

        if args.distributed:
            if 0.0 < args.p_semi < 1.0:
                assert 'SEMI' in args.train_mode
                train_sampler['SUP'].set_epoch(epoch)
                train_sampler['UNSUP'].set_epoch(epoch)
            else:
                train_sampler.set_epoch(epoch)

        if epoch == args.ema_start and 'GAN' in args.train_mode:
            if args.distributed:
                networks['G_EMA'].module.load_state_dict(networks['G'].module.state_dict())
            else:
                networks['G_EMA'].load_state_dict(networks['G'].state_dict())

        trainFunc(train_loader, networks, opts, epoch, args, {'logger': logger, 'queue': queue})

        validationFunc(val_loader, networks, epoch, args, {'logger': logger, 'queue': queue})

        # Calc fid
        if epoch >= args.fid_start and args.dataset not in ['ffhq', 'lsun_car', 'afhq_cat', 'afhq_dog', 'afhq_wild']:
            fid_ema = calcFIDBatch(args, {'VAL': val_loader, 'TRAIN': train_loader}, networks, 'EMA', train_dataset)
            fid_ema_mean = sum(fid_ema) / (len(fid_ema))

            if fid_best_ema > fid_ema_mean:
                fid_best_ema = fid_ema_mean
                save_model(args, 4567, networks, opts)

            print("Mean FID : [{}] AT EPOCH[{}] G_EMA / BEST EVER[{}]".format(fid_ema_mean, epoch + 1, fid_best_ema))

        # Write logs
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0):
            if (epoch + 1) % 10 == 0:
                save_model(args, epoch, networks, opts)
            if not args.train_mode in ['CLS_UN', 'CLS_SEMI']:
                if epoch >= args.fid_start and args.dataset not in ['ffhq', 'lsun_car']:
                    for idx_fid in range(len(args.att_to_use)):
                        add_logs(args, logger, 'STATEMA/G_EMA{}/FID'.format(idx_fid), fid_ema[idx_fid], epoch + 1)
                    add_logs(args, logger, 'STATEMA/G_EMA/mFID', fid_ema_mean, epoch + 1)
            if len(args.epoch_acc) > 0:
                add_logs(args, logger, 'STATC/Acc', float(args.epoch_acc[-1]), epoch + 1)


#################
# Sub functions #
#################
def print_args(args):
    for arg in vars(args):
        print('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))


def build_model(args):
    args.to_train = 'CDGI'

    networks = {}
    opts = {}
    is_semi = (0.0 < args.p_semi < 1.0)
    if is_semi:
        assert 'SEMI' in args.train_mode
    if 'C' in args.to_train:
        networks['C'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k})
        networks['C_EMA'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k})
    if 'D' in args.to_train:
        networks['D'] = Discriminator(args.img_size, num_domains=args.output_k)
    if 'G' in args.to_train:
        networks['G'] = Generator(args.img_size, args.sty_dim, use_sn=False)
        networks['G_EMA'] = Generator(args.img_size, args.sty_dim, use_sn=False)
    if 'I' in args.to_train:
        networks['inceptionNet'] = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]])

    if args.distributed:
        if args.gpu is not None:
            print('Distributed to', args.gpu)
            torch.cuda.set_device(args.gpu)
            args.batch_size = int(args.batch_size / args.ngpus_per_node)
            args.workers = int(args.workers / args.ngpus_per_node)
            for name, net in networks.items():
                if name in ['inceptionNet']:
                    continue
                net_tmp = net.cuda(args.gpu)
                networks[name] = torch.nn.parallel.DistributedDataParallel(net_tmp, device_ids=[args.gpu], output_device=args.gpu)
        else:
            for name, net in networks.items():
                net_tmp = net.cuda()
                networks[name] = torch.nn.parallel.DistributedDataParallel(net_tmp)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        for name, net in networks.items():
            networks[name] = net.cuda(args.gpu)
    else:
        for name, net in networks.items():
            networks[name] = torch.nn.DataParallel(net).cuda()

    if 'C' in args.to_train:
        opts['C'] = torch.optim.Adam(
            networks['C'].module.parameters() if args.distributed else networks['C'].parameters(),
            1e-4, weight_decay=0.001)
        if args.distributed:
            networks['C_EMA'].module.load_state_dict(networks['C'].module.state_dict())
        else:
            networks['C_EMA'].load_state_dict(networks['C'].state_dict())
    if 'D' in args.to_train:
        opts['D'] = torch.optim.RMSprop(
            networks['D'].module.parameters() if args.distributed else networks['D'].parameters(),
            1e-4, weight_decay=0.0001)
    if 'G' in args.to_train:
        opts['G'] = torch.optim.RMSprop(
            networks['G'].module.parameters() if args.distributed else networks['G'].parameters(),
            1e-4, weight_decay=0.0001)

    return networks, opts


def load_model(args, networks, opts):
    if args.load_model is not None:
        check_load = open(os.path.join(args.log_dir, "checkpoint.txt"), 'r')
        to_restore = check_load.readlines()[-1].strip()
        load_file = os.path.join(args.log_dir, to_restore)
        if os.path.isfile(load_file):
            print("=> loading checkpoint '{}'".format(load_file))
            checkpoint = torch.load(load_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            if not args.multiprocessing_distributed:
                for name, net in networks.items():
                    if name in ['inceptionNet']:
                        continue
                    tmp_keys = next(iter(checkpoint[name + '_state_dict'].keys()))
                    if 'module' in tmp_keys:
                        tmp_new_dict = OrderedDict()
                        for key, val in checkpoint[name + '_state_dict'].items():
                            tmp_new_dict[key[7:]] = val
                        net.load_state_dict(tmp_new_dict)
                        networks[name] = net
                    else:
                        net.load_state_dict(checkpoint[name + '_state_dict'])
                        networks[name] = net

            for name, opt in opts.items():
                opt.load_state_dict(checkpoint[name.lower() + '_optimizer'])
                opts[name] = opt
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(load_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.log_dir))


def get_loader(args, dataset):
    train_dataset = dataset['train']
    val_dataset = dataset['val']
    if 'afhq' in args.dataset:
        val_dataset = dataset['val']['VAL']

    print(len(val_dataset))

    # GAN_IIC_SEMI
    if 0.0 < args.p_semi < 1.0:
        assert 'SEMI' in args.train_mode
        train_sup_dataset = train_dataset['SUP']
        train_unsup_dataset = train_dataset['UNSUP']

        if args.distributed:
            train_sup_sampler = torch.utils.data.distributed.DistributedSampler(train_sup_dataset)
            train_unsup_sampler = torch.utils.data.distributed.DistributedSampler(train_unsup_dataset)
        else:
            train_sup_sampler = None
            train_unsup_sampler = None

        # If there are not cpus enough, set workers to 0
        train_sup_loader = torch.utils.data.DataLoader(train_sup_dataset, batch_size=args.batch_size,
                                                      shuffle=(train_sup_sampler is None), num_workers=0,
                                                      pin_memory=True, sampler=train_sup_sampler, drop_last=False)
        train_unsup_loader = torch.utils.data.DataLoader(train_unsup_dataset, batch_size=args.batch_size,
                                                      shuffle=(train_unsup_sampler is None), num_workers=0,
                                                      pin_memory=True, sampler=train_unsup_sampler, drop_last=False)

        train_loader = {'SUP': train_sup_loader, 'UNSUP': train_unsup_loader}
        train_sampler = {'SUP': train_sup_sampler, 'UNSUP': train_unsup_sampler}

    # GAN_SUP / GAN_IIC_UN
    else:
        train_dataset_ = train_dataset['TRAIN']
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_)
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(train_dataset_, batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None), num_workers=args.workers,
                                                   pin_memory=True, sampler=train_sampler, drop_last=False)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch, shuffle=True,
                                             num_workers=0, pin_memory=True, drop_last=False)

    val_loader = {'VAL': val_loader, 'VALSET': val_dataset if not args.dataset in ['afhq_cat', 'afhq_dog', 'afhq_wild'] else dataset['val']['FULL'], 'TRAINSET': train_dataset['FULL']}
    if 'afhq' in args.dataset:
        val_loader['IDX'] = train_dataset['IDX']

    return train_loader, val_loader, train_sampler


def map_exec_func(args):
    if args.train_mode == 'GAN_SUP':
        trainFunc = trainGAN_SUP
        validationFunc = validateUN
    elif args.train_mode == 'GAN_UNSUP':
        trainFunc = trainGAN_UNSUP
        validationFunc = validateUN
    elif args.train_mode == 'GAN_SEMI':
        trainFunc = trainGAN_SEMI
        validationFunc = validateUN
    else:
        exit(-6)

    return trainFunc, validationFunc


def save_model(args, epoch, networks, opts):
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0):
        check_list = open(os.path.join(args.log_dir, "checkpoint.txt"), "a+")
        # if (epoch + 1) % (args.epochs//10) == 0:
        with torch.no_grad():
            save_dict = {}
            save_dict['epoch'] = epoch + 1
            for name, net in networks.items():
                save_dict[name+'_state_dict'] = net.state_dict()
                if name in ['G_EMA', 'inceptionNet', 'C_EMA']:
                    continue
                save_dict[name.lower()+'_optimizer'] = opts[name].state_dict()
            print("SAVE CHECKPOINT[{}] DONE".format(epoch+1))
            save_checkpoint(save_dict, check_list, args.log_dir, epoch + 1)
        check_list.close()


if __name__ == '__main__':
    main()
