"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license 
"""
from tqdm import trange
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tools.utils import *
from tools.ops import compute_grad_gp, update_average, copy_norm_params, calc_iic_loss, \
    queue_data, dequeue_data, average_gradients, calc_adv_loss, calc_contrastive_loss, calc_recon_loss


def trainGAN_SEMI(data_loader, networks, opts, epoch, args, additional):
    # avg meter
    d_losses = AverageMeter()
    d_advs = AverageMeter()
    d_gps = AverageMeter()

    g_losses = AverageMeter()
    g_advs = AverageMeter()
    g_imgrecs = AverageMeter()
    g_styconts = AverageMeter()

    c_losses = AverageMeter()
    moco_losses = AverageMeter()
    iic_losses = AverageMeter()
    ce_losses = AverageMeter()

    # set nets
    D = networks['D']
    G = networks['G'] if not args.distributed else networks['G'].module
    C = networks['C'] if not args.distributed else networks['C'].module
    G_EMA = networks['G_EMA'] if not args.distributed else networks['G_EMA'].module
    C_EMA = networks['C_EMA'] if not args.distributed else networks['C_EMA'].module
    # set opts
    d_opt = opts['D']
    g_opt = opts['G']
    c_opt = opts['C']
    # switch to train mode
    D.train()
    G.train()
    C.train()
    G_EMA.train()
    C_EMA.train()

    logger = additional['logger']
    queue = additional['queue']

    train_sup_it = iter(data_loader['SUP'])
    train_unsup_it = iter(data_loader['UNSUP'])

    sup_iter_tmp = int(args.iters * args.p_semi)

    sup_iter_tmp = 100 if sup_iter_tmp < 100 and epoch < args.unsup_start else sup_iter_tmp

    sup_iter = sup_iter_tmp
    un_iter = (args.iters - sup_iter_tmp) if epoch >= args.unsup_start else 0

    t_sup = trange(0, sup_iter, initial=0, total=sup_iter)
    t_un = trange(0, un_iter, initial=0, total=un_iter)

    for i in t_sup:
        try:
            imgs, y_org = next(train_sup_it)
        except:
            train_sup_it = iter(data_loader['SUP'])
            imgs, y_org = next(train_sup_it)

        x_org = imgs[0]
        x_tf = imgs[1]

        x_ref_idx = torch.randperm(x_org.size(0))

        x_org = x_org.cuda(args.gpu)
        x_tf = x_tf.cuda(args.gpu)
        y_org = y_org.cuda(args.gpu)
        x_ref_idx = x_ref_idx.cuda(args.gpu)

        x_ref = x_org.clone()
        x_ref = x_ref[x_ref_idx]

        #################
        # BEGIN Train C #
        #################
        training_mode = 'SUP'

        q_cont = C.moco(x_org)
        k_cont = C_EMA.moco(x_tf)
        k_cont = k_cont.detach()

        q_disc = C.iic(x_org)
        k_disc = C.iic(x_tf)

        moco_loss = calc_contrastive_loss(args, q_cont, k_cont, queue)
        ce_loss = F.cross_entropy(q_disc, y_org) + F.cross_entropy(k_disc, y_org)

        c_loss = moco_loss + 5.0 * ce_loss

        if epoch >= args.separated:
            c_loss = 0.1 * c_loss

        c_opt.zero_grad()
        c_loss.backward()
        if args.distributed:
            average_gradients(C)
        c_opt.step()
        ###############
        # END Train C #
        ###############

        if epoch >= args.separated:
            ####################
            # BEGIN Train GANs #
            ####################
            with torch.no_grad():
                y_ref = y_org.clone()
                y_ref = y_ref[x_ref_idx]
                s_ref = C.moco(x_ref)
                c_src = G.cnt_encoder(x_org)
                x_fake = G.decode(c_src, s_ref)

            x_ref.requires_grad_()

            d_real_logit, _ = D(x_ref, y_ref)
            d_fake_logit, _ = D(x_fake.detach(), y_ref)

            d_adv_real = calc_adv_loss(d_real_logit, 'd_real')
            d_adv_fake = calc_adv_loss(d_fake_logit, 'd_fake')

            d_adv = d_adv_real + d_adv_fake

            d_gp = args.w_gp * compute_grad_gp(d_real_logit, x_ref, is_patch=False)

            d_loss = d_adv + d_gp

            d_opt.zero_grad()
            d_adv_real.backward(retain_graph=True)
            d_gp.backward()
            d_adv_fake.backward()
            if args.distributed:
                average_gradients(D)
            d_opt.step()

            # Train G
            s_src = C.moco(x_org)
            s_ref = C.moco(x_ref)

            c_src = G.cnt_encoder(x_org)
            x_fake = G.decode(c_src, s_ref)

            x_rec = G.decode(c_src, s_src)

            g_fake_logit, _ = D(x_fake, y_ref)
            g_rec_logit, _ = D(x_rec, y_org)

            g_adv_fake = calc_adv_loss(g_fake_logit, 'g')
            g_adv_rec = calc_adv_loss(g_rec_logit, 'g')

            g_adv = g_adv_fake + g_adv_rec

            g_imgrec = calc_recon_loss(x_rec, x_org)

            s_fake = C.moco(x_fake)
            s_ref_ema = C_EMA.moco(x_ref)
            s_ref_ema = s_ref_ema.detach()

            g_sty_contrastive = calc_contrastive_loss(args, s_fake, s_ref_ema, queue)

            g_loss = args.w_adv * g_adv + args.w_rec * g_imgrec + args.w_vec * g_sty_contrastive

            g_opt.zero_grad()
            c_opt.zero_grad()
            g_loss.backward()
            if args.distributed:
                average_gradients(G)
                average_gradients(C)
            c_opt.step()
            g_opt.step()
            ##################
            # END Train GANs #
            ##################

        queue = queue_data(queue, k_cont)
        queue = dequeue_data(queue)

        if epoch >= args.ema_start:
            training_mode = training_mode + "_EMA"
            update_average(G_EMA, G)

        update_average(C_EMA, C)

        torch.cuda.synchronize()

        with torch.no_grad():
            if epoch >= args.separated:
                d_losses.update(d_loss.item(), x_org.size(0))
                d_advs.update(d_adv.item(), x_org.size(0))
                d_gps.update(d_gp.item(), x_org.size(0))

                g_losses.update(g_loss.item(), x_org.size(0))
                g_advs.update(g_adv.item(), x_org.size(0))
                g_imgrecs.update(g_imgrec.item(), x_org.size(0))
                g_styconts.update(g_sty_contrastive.item(), x_org.size(0))

            c_losses.update(c_loss.item(), x_org.size(0))
            moco_losses.update(moco_loss.item(), x_org.size(0))
            ce_losses.update(ce_loss.item(), x_org.size(0))

            if (i + 1) % args.log_step == 0 and (args.gpu == 0 or args.gpu == '0'):
                print('Epoch: [{}/{}] [{}/{}] MODE[{}] Avg Loss: D[{d_losses.avg:.2f}] G[{g_losses.avg:.2f}] '
                      'C[{c_losses.avg:.2f}]'.format(epoch + 1, args.epochs, i+1, sup_iter, training_mode,
                                                     d_losses=d_losses, g_losses=g_losses, c_losses=c_losses))

    for i in t_un:
        try:
            imgs, _ = next(train_unsup_it)
        except:
            train_unsup_it = iter(data_loader['UNSUP'])
            imgs, _ = next(train_unsup_it)

        x_org = imgs[0]
        x_tf = imgs[1]

        x_ref_idx = torch.randperm(x_org.size(0))

        x_org = x_org.cuda(args.gpu)
        x_tf = x_tf.cuda(args.gpu)
        x_ref_idx = x_ref_idx.cuda(args.gpu)

        x_ref = x_org.clone()
        x_ref = x_ref[x_ref_idx]

        #################
        # BEGIN Train C #
        #################
        training_mode = 'ONLYCLS'
        q_cont = C.moco(x_org)
        k_cont = C_EMA.moco(x_tf)
        k_cont = k_cont.detach()

        q_disc = C.iic(x_org)
        k_disc = C.iic(x_tf)

        q_disc = F.softmax(q_disc, 1)
        k_disc = F.softmax(k_disc, 1)

        moco_loss = calc_contrastive_loss(args, q_cont, k_cont, queue)
        iic_loss = calc_iic_loss(q_disc, k_disc)

        c_loss = moco_loss + 5.0 * iic_loss

        c_loss = 0.1 * c_loss

        c_opt.zero_grad()
        c_loss.backward()
        if args.distributed:
            average_gradients(C)
        c_opt.step()
        ###############
        # END Train C #
        ###############

        ####################
        # BEGIN Train GANs #
        ####################
        if epoch >= args.separated:
            training_mode = 'C2GANs'
            with torch.no_grad():
                y_org = torch.argmax(q_disc, 1)
                y_ref = y_org.clone()
                y_ref = y_ref[x_ref_idx]
                s_ref = C.moco(x_ref)
                c_src = G.cnt_encoder(x_org)
                x_fake = G.decode(c_src, s_ref)

            x_ref.requires_grad_()

            d_real_logit, _ = D(x_ref, y_ref)
            d_fake_logit, _ = D(x_fake.detach(), y_ref)

            d_adv_real = calc_adv_loss(d_real_logit, 'd_real')
            d_adv_fake = calc_adv_loss(d_fake_logit, 'd_fake')

            d_adv = d_adv_real + d_adv_fake

            d_gp = args.w_gp * compute_grad_gp(d_real_logit, x_ref, is_patch=False)

            d_loss = d_adv + d_gp

            d_opt.zero_grad()
            d_adv_real.backward(retain_graph=True)
            d_gp.backward()
            d_adv_fake.backward()
            if args.distributed:
                average_gradients(D)
            d_opt.step()

            # Train G
            s_src = C.moco(x_org)
            s_ref = C.moco(x_ref)

            c_src = G.cnt_encoder(x_org)
            x_fake = G.decode(c_src, s_ref)

            x_rec = G.decode(c_src, s_src)

            g_fake_logit, _ = D(x_fake, y_ref)
            g_rec_logit, _ = D(x_rec, y_org)

            g_adv_fake = calc_adv_loss(g_fake_logit, 'g')
            g_adv_rec = calc_adv_loss(g_rec_logit, 'g')

            g_adv = g_adv_fake + g_adv_rec

            g_imgrec = calc_recon_loss(x_rec, x_org)

            s_fake = C.moco(x_fake)
            s_ref_ema = C_EMA.moco(x_ref)
            s_ref_ema = s_ref_ema.detach()

            g_sty_contrastive = calc_contrastive_loss(args, s_fake, s_ref_ema, queue)

            g_loss = args.w_adv * g_adv + args.w_rec * g_imgrec + args.w_vec * g_sty_contrastive

            g_opt.zero_grad()
            c_opt.zero_grad()
            g_loss.backward()
            if args.distributed:
                average_gradients(G)
                average_gradients(C)
            c_opt.step()
            g_opt.step()
        ##################
        # END Train GANs #
        ##################

        queue = queue_data(queue, k_cont)
        queue = dequeue_data(queue)

        if epoch >= args.ema_start:
            training_mode = training_mode + "_EMA"
            update_average(G_EMA, G)

        update_average(C_EMA, C)

        torch.cuda.synchronize()

        with torch.no_grad():
            if epoch >= args.separated:
                d_losses.update(d_loss.item(), x_org.size(0))
                d_advs.update(d_adv.item(), x_org.size(0))
                d_gps.update(d_gp.item(), x_org.size(0))

                g_losses.update(g_loss.item(), x_org.size(0))
                g_advs.update(g_adv.item(), x_org.size(0))
                g_imgrecs.update(g_imgrec.item(), x_org.size(0))
                g_styconts.update(g_sty_contrastive.item(), x_org.size(0))

            c_losses.update(c_loss.item(), x_org.size(0))
            iic_losses.update(iic_loss.item(), x_org.size(0))
            moco_losses.update(moco_loss.item(), x_org.size(0))

            if (i + 1) % args.log_step == 0 and (args.gpu == 0 or args.gpu == '0'):
                summary_step = epoch * args.iters + i
                add_logs(args, logger, 'D/LOSS', d_losses.avg, summary_step)
                add_logs(args, logger, 'D/ADV', d_advs.avg, summary_step)
                add_logs(args, logger, 'D/GP', d_gps.avg, summary_step)

                add_logs(args, logger, 'G/LOSS', g_losses.avg, summary_step)
                add_logs(args, logger, 'G/ADV', g_advs.avg, summary_step)
                add_logs(args, logger, 'G/IMGREC', g_imgrecs.avg, summary_step)
                add_logs(args, logger, 'G/STYCONT', g_styconts.avg, summary_step)

                add_logs(args, logger, 'C/LOSS', c_losses.avg, summary_step)
                add_logs(args, logger, 'C/IID', iic_losses.avg, summary_step)
                add_logs(args, logger, 'C/MOCO', moco_losses.avg, summary_step)
                add_logs(args, logger, 'C/CE', ce_losses.avg, summary_step)

                print('Epoch: [{}/{}] [{}/{}] MODE[{}] Avg Loss: D[{d_losses.avg:.2f}] G[{g_losses.avg:.2f}] '
                      'C[{c_losses.avg:.2f}]'.format(epoch + 1, args.epochs, i+1, un_iter, training_mode,
                                                     d_losses=d_losses, g_losses=g_losses, c_losses=c_losses))

    copy_norm_params(G_EMA, G)
    copy_norm_params(C_EMA, C)
