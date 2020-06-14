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

import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from tools.utils import *


def plot_tSNE(data_loader, networks, epoch, args, additional=None):
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
    val_loader = data_loader['VAL']

    with torch.no_grad():
        data = []
        targets = []

        val_iter = iter(val_loader)

        cluster_grid = [[] for _ in range(args.output_k)]

        gt = [[] for _ in range(args.output_k)]
        gtlist = []

        for i in tqdm(range(len(val_loader))):
            x, y = next(val_iter)
            x = x[0]
            # x = x.view(1, *x.shape)
            x = x.cuda(args.gpu)
            outs = C_EMA(x)
            # outs = C(x)
            feat = outs['cont']
            logit = outs['disc']

            target = torch.argmax(logit, 1)

            for idx in range(len(feat.cpu().data.numpy())):
                data.append(feat.cpu().data.numpy()[idx])
                targets.append(int(target[idx].item()))
                gtlist.append(int(y[idx].item()))
                gt[int(target[idx].item())].append(y[idx].item())
                cluster_grid[int(target[idx].item())].append(x[idx].view(1, *x[idx].shape))

        targets_np = np.array(targets)
        data_np = np.array(data)
        np.save('ours_AFHQ_{}_gt.npy'.format(args.load_model), targets_np)
        np.save('ours_AFHQ_{}_feat.npy'.format(args.load_model), data_np)

        cluster_map = {}

        for i in range(args.output_k):
            numlist = [0 for _ in range(args.output_k)]
            for g in gt[i]:
                numlist[g] += 1
            cluster_map[i] = np.argmax(numlist)

        for i in range(args.output_k):
            print(i, len(cluster_grid[i]), cluster_map[i])
            if len(cluster_grid[i]) == 0:
                continue
            tmp = torch.cat(cluster_grid[i], 0)
            vutils.save_image(tmp, 'GRID{}.jpg'.format(i), normalize=True, nrow=int(np.sqrt(tmp.size(0))), padding=0)

        print(cluster_map)
        cluster_map_list = sorted(cluster_map, key=cluster_map.get)
        print(cluster_map_list)
        ret = TSNE(n_components=2, random_state=0).fit_transform(data)

    def show(data_iter, targets, t_sne_ret):
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'violet', 'orange', 'purple']
        colors = colors[:args.output_k]

        plt.figure(figsize=(12, 10))
        print(set(targets))
        print()
        for label in set(targets):
            # idx = np.where(np.array(targets) == cluster_map[cluster_map_list[label]])[0]
            idx = np.where(np.array(targets) == label)[0]
            plt.scatter(t_sne_ret[idx, 0], t_sne_ret[idx, 1], c=colors[label], label=label)

        plt.legend()
        plt.ylim([-40, 40])
        plt.xlim([-35, 35])
        plt.savefig('{}.png'.format(args.load_model))

        print(len(t_sne_ret))

    val_iter = iter(val_loader)

    show(val_iter, targets, ret)
