"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license 
"""
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import torch.nn.functional as F

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from scipy import linalg

from tools.utils import *
from validation.cluster_eval import cluster_eval


def validateUN(data_loader, networks, epoch, args, additional=None):
    # set nets
    D = networks['D']
    G = networks['G'] if not args.distributed else networks['G'].module
    C = networks['C'] if not args.distributed else networks['C'].module
    C_EMA = networks['C_EMA'] if not args.distributed else networks['C_EMA'].module
    G_EMA = networks['G_EMA'] if not args.distributed else networks['G_EMA'].module
    # switch to train mode
    D.eval()
    G.eval()
    C.eval()
    C_EMA.eval()
    G_EMA.eval()
    # data loader
    val_dataset = data_loader['TRAINSET'] if args.dataset in ['animal_faces', 'lsun_car', 'ffhq'] else data_loader['VALSET']
    val_loader = data_loader['VAL']

    # Calculate Acc. of Classifier
    with torch.no_grad():
        if not args.dataset in ['afhq_dog', 'afhq_cat', 'afhq_wild', 'lsun_car', 'ffhq'] and args.output_k == len(args.att_to_use):
            is_best = cluster_eval(args, C, val_loader, val_loader)
            print("EPOCH {} / BEST EVER {:.4f}".format(epoch+1, max(args.epoch_acc)))
    
    # Parse images for average reference vector
    x_each_cls = []
    if args.dataset == 'animal_faces':
        num_tmp_val = -50
    elif args.dataset == 'ffhq':
        num_tmp_val = -7000
    elif args.dataset == 'lsun_car':
        num_tmp_val = -10000
    else:
        num_tmp_val = 0
    
    with torch.no_grad():
        val_tot_tars = torch.tensor(val_dataset.targets)
        for cls_idx in range(len(args.att_to_use)):
            tmp_cls_set = (val_tot_tars == args.att_to_use[cls_idx]).nonzero()[num_tmp_val:]
            tmp_ds = torch.utils.data.Subset(val_dataset, tmp_cls_set)
            tmp_dl = torch.utils.data.DataLoader(tmp_ds, batch_size=50, shuffle=False,
                                                 num_workers=0, pin_memory=True, drop_last=False)
            tmp_iter = iter(tmp_dl)
            tmp_sample = None
            for sample_idx in range(len(tmp_iter)):
                imgs, _ = next(tmp_iter)
                x_ = imgs[0]
                if tmp_sample is None:
                    tmp_sample = x_.clone()
                else:
                    tmp_sample = torch.cat((tmp_sample, x_), 0)
            x_each_cls.append(tmp_sample)
    
    #######
    if epoch >= args.fid_start:
        val_iter = iter(val_loader)
        cluster_grid = [[] for _ in range(args.output_k)]
        for _ in tqdm(range(len(val_loader))):
            x, y = next(val_iter)
            x = x[0]
            x = x.cuda(args.gpu)
            outs = C(x)
            feat = outs['cont']
            logit = outs['disc']
    
            target = torch.argmax(logit, 1)
    
            for idx in range(len(feat.cpu().data.numpy())):
                cluster_grid[int(target[idx].item())].append(x[idx].view(1, *x[idx].shape))
    
        all_none_zero = True
    
        min_len = 9999
    
        for i in range(len(cluster_grid)):
            if len(cluster_grid[i]) == 0:
                all_none_zero = False
                break
            if min_len > len(cluster_grid[i]):
                min_len = len(cluster_grid[i])
            cluster_grid[i] = torch.cat(cluster_grid[i], 0)
    
        if all_none_zero:
            for cls in cluster_grid:
                print(len(cls), cls.shape)
            # AVG
            with torch.no_grad():
                grid_row = min(min_len, 30)
                for i in range(len(cluster_grid)):
                    s_tmp = C_EMA.moco(cluster_grid[i])
                    s_avg = torch.mean(s_tmp, 0, keepdim=True)
                    s_avg = s_avg.repeat((grid_row, 1, 1, 1))
                    for j in range(len(cluster_grid)):
                        c_tmp = G_EMA.cnt_encoder(cluster_grid[j][:grid_row])
                        x_avg = G_EMA.decode(c_tmp, s_avg)
                        x_res = torch.cat((cluster_grid[j][:grid_row], x_avg), 0)
                        vutils.save_image(x_res, os.path.join(args.res_dir, 'AVG{}{}.jpg'.format(j, i)), normalize=True, nrow=grid_row, padding=0)
    
        # Reference guided
        with torch.no_grad():
            # Just a buffer image ( to make a grid )
            ones = torch.ones(1, x_each_cls[0].size(1), x_each_cls[0].size(2), x_each_cls[0].size(3)).cuda(args.gpu, non_blocking=True)
            for src_idx in range(len(args.att_to_use)):
                x_src = x_each_cls[src_idx][:args.val_batch, :, :, :].cuda(args.gpu, non_blocking=True)
                rnd_idx = torch.randperm(x_each_cls[src_idx].size(0))[:args.val_batch]
                x_src_rnd = x_each_cls[src_idx][rnd_idx].cuda(args.gpu, non_blocking=True)
                for ref_idx in range(len(args.att_to_use)):
                    x_res_ema = torch.cat((ones, x_src), 0)
                    x_rnd_ema = torch.cat((ones, x_src_rnd), 0)
                    x_ref = x_each_cls[ref_idx][:args.val_batch, :, :, :].cuda(args.gpu, non_blocking=True)
                    rnd_idx = torch.randperm(x_each_cls[ref_idx].size(0))[:args.val_batch]
                    x_ref_rnd = x_each_cls[ref_idx][rnd_idx].cuda(args.gpu, non_blocking=True)
                    for sample_idx in range(args.val_batch):
                        x_ref_tmp = x_ref[sample_idx: sample_idx + 1].repeat((args.val_batch, 1, 1, 1))
    
                        c_src = G_EMA.cnt_encoder(x_src)
                        s_ref = C_EMA(x_ref_tmp, sty=True)
                        x_res_ema_tmp = G_EMA.decode(c_src, s_ref)
    
                        x_ref_tmp = x_ref_rnd[sample_idx: sample_idx + 1].repeat((args.val_batch, 1, 1, 1))
    
                        c_src = G_EMA.cnt_encoder(x_src_rnd)
                        s_ref = C_EMA(x_ref_tmp, sty=True)
                        x_rnd_ema_tmp = G_EMA.decode(c_src, s_ref)
    
                        x_res_ema_tmp = torch.cat((x_ref[sample_idx: sample_idx + 1], x_res_ema_tmp), 0)
                        x_res_ema = torch.cat((x_res_ema, x_res_ema_tmp), 0)
    
                        x_rnd_ema_tmp = torch.cat((x_ref_rnd[sample_idx: sample_idx + 1], x_rnd_ema_tmp), 0)
                        x_rnd_ema = torch.cat((x_rnd_ema, x_rnd_ema_tmp), 0)
    
                    vutils.save_image(x_res_ema, os.path.join(args.res_dir, '{}_EMA_{}_{}{}.jpg'.format(args.gpu, epoch+1, src_idx, ref_idx)), normalize=True,
                                    nrow=(x_res_ema.size(0) // (x_src.size(0) + 2) + 1))
                    vutils.save_image(x_rnd_ema, os.path.join(args.res_dir, '{}_RNDEMA_{}_{}{}.jpg'.format(args.gpu, epoch+1, src_idx, ref_idx)), normalize=True,
                                    nrow=(x_res_ema.size(0) // (x_src.size(0) + 2) + 1))


def calcFIDBatch(args, data_loader, networks, model='NONE', train_dataset=None):
    # Set non-shuffle train loader + extract class-wise images
    trainset = train_dataset['FULL']
    val_dataset = data_loader['VAL']['TRAINSET'] if args.dataset == 'animal_faces' else data_loader['VAL']['VALSET']

    if model == 'EMA':
        G = networks['G_EMA'] if not args.distributed else networks['G_EMA'].module
        C = networks['C_EMA'] if not args.distributed else networks['C_EMA'].module
    else:
        G = networks['G'] if not args.distributed else networks['G'].module
        C = networks['C'] if not args.distributed else networks['C'].module

    inceptionNet = networks['inceptionNet']

    inceptionNet.eval()
    G.eval()
    C.eval()

    eps = 1e-6
    bs_fid = 50

    fid_classwise = []

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    model_device = next(G.parameters()).device

    use_cuda = not (model_device == torch.device('cpu'))

    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)

    if use_cuda:
        mean = mean.cuda(args.gpu)
        std = std.cuda(args.gpu)
        inceptionNet.cuda(args.gpu)

    with torch.no_grad():
        # ========================= #
        # Parse source images (val) #
        # ========================= #
        x_val_list = []
        num_target_dataset = (args.min_data + args.max_data) // 2
        multiples = bs_fid // len(args.att_to_use)
        if multiples < 1:
            multiples = 1
        if len(args.att_to_use) == 1:
            num_each_val = (num_target_dataset // multiples)
        else:
            num_each_val = (num_target_dataset // multiples) // (len(args.att_to_use) - 1)

        val_tot_tars = torch.tensor(val_dataset.targets)
        for cls_idx in range(len(args.att_to_use)):
            num_tmp_val = -50 if args.dataset == 'animal_faces' else 0
            tmp_cls_set = (val_tot_tars == args.att_to_use[cls_idx]).nonzero()[num_tmp_val:]
            tmp_ds = torch.utils.data.Subset(val_dataset, tmp_cls_set)
            tmp_dl = torch.utils.data.DataLoader(tmp_ds, batch_size=50, shuffle=False,
                                                 num_workers=0, pin_memory=True, drop_last=False)
            tmp_iter = iter(tmp_dl)
            tmp_sample = None
            for _ in range(len(tmp_iter)):
                imgs, _ = next(tmp_iter)
                x_ = imgs[0]
                if tmp_sample is None:
                    tmp_sample = x_.clone()
                else:
                    tmp_sample = torch.cat((tmp_sample, x_), 0)
            x_val_list.append(tmp_sample)

        # ================ #
        # Parse real preds #
        # ================ #
        pred_real_list = []
        for idx_cls in range(len(args.att_to_use)):
            pred_real = np.empty((args.max_data, args.dims))

            tot_targets = torch.tensor(trainset.targets)
            tmp_sub_idx = (tot_targets == args.att_to_use[idx_cls]).nonzero()
            num_train_to_use = -50 if args.dataset == 'animal_faces' else num_target_dataset
            train_to_use = tmp_sub_idx[:num_train_to_use]

            tmp_dataset = torch.utils.data.Subset(trainset, train_to_use)

            tmp_train_loader = torch.utils.data.DataLoader(tmp_dataset, batch_size=bs_fid,
                                                           shuffle=False, num_workers=args.workers,
                                                           pin_memory=True, drop_last=False)

            tmp_num_sample = len(tmp_dataset)
            tmp_num_iter = iter(tmp_train_loader)

            remainder = tmp_num_sample - (bs_fid * (len(tmp_train_loader) - 1))

            for i in range(len(tmp_train_loader)):
                imgs, _ = next(tmp_num_iter)
                x_train = imgs[0]
                if use_cuda:
                    x_train = x_train.cuda(args.gpu)
                x_train = (x_train + 1.0) / 2.0
                x_train = (x_train - mean) / std
                x_train = F.interpolate(x_train, size=[299, 299])

                tmp_real = inceptionNet(x_train)[0]
                if tmp_real.shape[2] != 1 or tmp_real.shape[3] != 1:
                    tmp_real = F.adaptive_avg_pool2d(tmp_real, (1, 1))

                if i == (len(tmp_num_iter) - 1) and remainder > 0:
                    pred_real[i * bs_fid: i * bs_fid + remainder] = tmp_real.cpu().data.numpy().reshape(remainder, -1)
                else:
                    pred_real[i * bs_fid: (i + 1) * bs_fid] = tmp_real.cpu().data.numpy().reshape(bs_fid, -1)
            pred_real = pred_real[:tmp_num_sample]
            # print("NUM REAL", idx_cls, len(pred_real), tmp_num_sample)
            pred_real_list.append(pred_real)

        # ================ #
        # Parse fake preds #
        # ================ #
        pred_fake_list = []
        for i in range(len(args.att_to_use)):
            pred_fake = np.empty((args.max_data, args.dims))
            # sty. vector of target domain
            s_tmp_list = []
            for sel_idx in range(multiples):
                x_sty_idx = torch.randperm(x_val_list[i].size(0))[:num_each_val]
                x_val_selected = x_val_list[i][x_sty_idx]
                if use_cuda:
                    x_val_selected = x_val_selected.cuda(args.gpu)
                s_tmp = C(x_val_selected, sty=True)
                s_tmp_list.append(s_tmp)

            num_used = 0

            if len(args.att_to_use) == 1:
                inner_iter = 1
            else:
                inner_iter = len(args.att_to_use) - 1

            for j in range(inner_iter):
                if inner_iter == 1:
                    x_cnt_tmp = x_val_list[j][:num_each_val]
                else:
                    x_cnt_tmp = x_val_list[(i + j + 1) % len(args.att_to_use)][:num_each_val]
                if use_cuda:
                    x_cnt_tmp = x_cnt_tmp.cuda(args.gpu)
                x_tmp_list = []
                for sel_idx in range(multiples):
                    c_tmp = G.cnt_encoder(x_cnt_tmp)
                    x_fake_tmp = G.decode(c_tmp, s_tmp_list[sel_idx])
                    x_tmp_list.append(x_fake_tmp)

                x_fake_tmp = torch.cat(x_tmp_list, 0)

                tot_num_in_fake = x_fake_tmp.size(0)
                tot_iter_in_fake = (tot_num_in_fake // bs_fid)
                remainder_fake = tot_num_in_fake - tot_iter_in_fake * bs_fid

                if remainder_fake > 0:
                    tot_iter_in_fake += 1

                for k in range(tot_iter_in_fake):
                    if k == tot_iter_in_fake - 1 and remainder_fake > 0:
                        x_fake_tmp_ = x_fake_tmp[k * bs_fid: k * bs_fid + remainder_fake]
                    else:
                        x_fake_tmp_ = x_fake_tmp[k * bs_fid: (k + 1) * bs_fid]
                    num_used += len(x_fake_tmp_)
                    x_fake_tmp_ = (x_fake_tmp_ + 1.0) / 2.0
                    x_fake_tmp_ = (x_fake_tmp_ - mean) / std
                    x_fake_tmp_ = F.interpolate(x_fake_tmp_, size=[299, 299])

                    tmp_fake = inceptionNet(x_fake_tmp_)[0]

                    if tmp_fake.shape[2] != 1 or tmp_fake.shape[3] != 1:
                        tmp_fake = F.adaptive_avg_pool2d(tmp_fake, output_size=(1, 1))

                    if k == tot_iter_in_fake - 1 and remainder_fake > 0:
                        pred_fake[j * tot_num_in_fake + k * bs_fid:j * tot_num_in_fake + k * bs_fid + remainder_fake] = tmp_fake.cpu().data.numpy().reshape(remainder_fake, -1)
                    else:
                        pred_fake[j * tot_num_in_fake + k * bs_fid:j * tot_num_in_fake + (k + 1) * bs_fid] = tmp_fake.cpu().data.numpy().reshape(bs_fid, -1)

            mult = (len(args.att_to_use) - 1) if len(args.att_to_use) != 1 else 1

            pred_fake = pred_fake[:mult * num_each_val * multiples]
            pred_fake = pred_fake[:num_target_dataset]

            # print("NUM FAKE", i, len(pred_fake), mult * num_each_val * multiples, num_used)
            pred_fake_list.append(pred_fake)

        for i in range(len(args.att_to_use)):
            pred_real_ = pred_real_list[i]
            pred_fake_ = pred_fake_list[i]

            mu_real = np.atleast_1d(np.mean(pred_real_, axis=0))
            std_real = np.atleast_2d(np.cov(pred_real_, rowvar=False))

            mu_fake = np.atleast_1d(np.mean(pred_fake_, axis=0))
            std_fake = np.atleast_2d(np.cov(pred_fake_, rowvar=False))

            assert mu_fake.shape == mu_real.shape
            assert std_fake.shape == std_real.shape

            mu_diff = mu_fake - mu_real

            covmean, _ = linalg.sqrtm(std_fake.dot(std_real), disp=False)

            if not np.isfinite(covmean).all():
                msg = ('fid calculation produces singular product; '
                       'adding %s to diagonal of cov estimates') % eps
                print(msg)
                offset = np.eye(std_fake.shape[0]) * eps
                covmean = linalg.sqrtm((std_fake + offset).dot(std_real + offset))

            # Numerical error might give slight imaginary component
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError('Imaginary component {}'.format(m))
                covmean = covmean.real

            tr_covmean = np.trace(covmean)

            fid = mu_diff.dot(mu_diff) + np.trace(std_fake) + np.trace(std_real) - 2 * tr_covmean

            fid_classwise.append(fid)

        return fid_classwise
