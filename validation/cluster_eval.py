"""
Invariant Information Clustering for Unsupervised Image Classification and Segmentation
Copyright (c) 2019 Xu Ji
MIT license
"""
from __future__ import print_function

import sys
from datetime import datetime

import numpy as np
import torch

from validation.eval_metrics import _hungarian_match, _acc


def _clustering_get_data(args, net, dataloader):
    """
    Returns cuda tensors for flat preds and targets.
    """

    num_batches = len(dataloader)
    flat_targets_all = torch.zeros((num_batches * args.val_batch), dtype=torch.int32).cuda(int(args.gpu))
    flat_predss_all = torch.zeros((num_batches * args.val_batch), dtype=torch.int32).cuda(int(args.gpu))

    num_test = 0

    data_iter = iter(dataloader)

    for b_i in range(num_batches):
        imgs, flat_targets = next(data_iter)
        imgs = imgs[0]
        imgs = imgs.cuda(int(args.gpu))

        with torch.no_grad():
            outs = net(imgs)
            x_out = outs['disc']

        num_test_curr = flat_targets.shape[0]
        num_test += num_test_curr

        start_i = b_i * args.val_batch

        flat_preds_curr = torch.argmax(x_out, dim=1)  # along output_k
        flat_predss_all[start_i:(start_i + num_test_curr)] = flat_preds_curr

        flat_targets_all[start_i:(start_i + num_test_curr)] = flat_targets.cuda(int(args.gpu))

    flat_predss_all = flat_predss_all[:num_test]
    flat_targets_all = flat_targets_all[:num_test]

    return flat_predss_all, flat_targets_all


def cluster_subheads_eval(args, net,
                          mapping_assignment_dataloader,
                          mapping_test_dataloader,
                          get_data_fn=_clustering_get_data):
    """
    Used by both clustering and segmentation.
    Returns metrics for test set.
    Get result from average accuracy of all sub_heads (mean and std).
    All matches are made from training data.
    Best head metric, which is order selective unlike mean/std, is taken from
    best head determined by training data (but metric computed on test data).

    ^ detail only matters for IID+/semisup where there's a train/test split.

    Option to choose best sub_head either based on loss (set use_head in main
    script), or eval. Former does not use labels for the selection at all and this
    has negligible impact on accuracy metric for our models.
    """

    all_matches, train_accs = _get_assignment_data_matches(net,
                                                           mapping_assignment_dataloader,
                                                           args,
                                                           get_data_fn=get_data_fn)

    flat_predss_all, flat_targets_all, = \
        get_data_fn(args, net, mapping_test_dataloader)

    num_samples = flat_targets_all.shape[0]
    reordered_preds = torch.zeros(num_samples,
                                  dtype=flat_predss_all.dtype).cuda(int(args.gpu))
    for pred_i, target_i in all_matches:
        reordered_preds[flat_predss_all == pred_i] = torch.tensor(target_i).cuda(int(args.gpu))
    test_acc, conf_mat = _acc(reordered_preds, flat_targets_all, args.output_k, verbose=0)

    return {"test_accs": test_acc,
            "best": test_acc,
            "worst": test_acc,
            "train_accs": list(train_accs),
            "conf_mat": conf_mat}


def _get_assignment_data_matches(net, mapping_assignment_dataloader, args,
                                 get_data_fn=None,
                                 just_matches=False,
                                 verbose=0):
    """
    Get all best matches per head based on train set i.e. mapping_assign,
    and mapping_assign accs.
    """

    if verbose:
        print("calling cluster eval direct (helper) %s" % datetime.now())
        sys.stdout.flush()

    flat_predss_all, flat_targets_all = \
        get_data_fn(args, net, mapping_assignment_dataloader)

    if verbose:
        print("getting data fn has completed %s" % datetime.now())
        print("flat_targets_all %s, flat_predss_all[0] %s" %
              (list(flat_targets_all.shape), list(flat_predss_all.shape)))
        sys.stdout.flush()

    num_test = flat_targets_all.shape[0]
    if verbose == 2:
        print("num_test: %d" % num_test)
        for c in range(args.output_k):
            print("output_k: %d count: %d" % (c, (flat_targets_all == c).sum()))

    assert (flat_predss_all.shape == flat_targets_all.shape)
    num_samples = flat_targets_all.shape[0]

    if verbose:
        print("starting head %d with eval mode hung, %s" % (0, datetime.now()))
        sys.stdout.flush()

    match = _hungarian_match(flat_predss_all, flat_targets_all,
                             preds_k=args.output_k,
                             targets_k=args.output_k)
    if verbose:
        print("got match %s" % (datetime.now()))
        sys.stdout.flush()

    all_matches = match
    all_accs = []

    if not just_matches:
        # reorder predictions to be same cluster assignments as output_k
        found = torch.zeros(args.output_k)
        reordered_preds = torch.zeros(num_samples,
                                      dtype=flat_predss_all.dtype).cuda(int(args.gpu))

        for pred_i, target_i in match:
            # reordered_preds[flat_predss_all[i] == pred_i] = target_i
            reordered_preds[torch.eq(flat_predss_all, int(pred_i))] = torch.from_numpy(
                np.array(target_i)).cuda(int(args.gpu)).int().item()
            found[pred_i] = 1
            if verbose == 2:
                print((pred_i, target_i))
        assert (found.sum() == args.output_k)  # each output_k must get mapped

        if verbose:
            print("reordered %s" % (datetime.now()))
            sys.stdout.flush()

        acc, _ = _acc(reordered_preds, flat_targets_all, args.output_k, verbose)
        all_accs.append(acc)

    if just_matches:
        return all_matches
    else:
        return all_matches, all_accs


def cluster_eval(args, net, mapping_assignment_dataloader,
                 mapping_test_dataloader):

    net.eval()
    stats_dict = cluster_subheads_eval(args, net,
                                       mapping_assignment_dataloader=mapping_assignment_dataloader,
                                       mapping_test_dataloader=mapping_test_dataloader)

    acc = stats_dict["best"]
    best_conf_mat = stats_dict["conf_mat"]
    print("EPOCH ACC : {}".format(acc), 'MAX : {}'.format(max(args.epoch_acc) if len(args.epoch_acc) > 0 else 0.0))
    print('--------')
    is_best = (len(args.epoch_acc) > 0) and (acc > max(args.epoch_acc))

    args.epoch_stats.append(stats_dict)
    args.epoch_acc.append(acc)

    return is_best

