"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license 
"""
import os
import torch


class Logger(object):
    def __init__(self, log_dir):
        self.last = None

    def scalar_summary(self, tag, value, step):
        if self.last and self.last['step'] != step:
            print(self.last)
            self.last = None
        if self.last is None:
            self.last = {'step':step,'iter':step,'epoch':1}
        self.last[tag] = value

    def images_summary(self, tag, images, step, nrow=8):
        """Log a list of images."""
        self.viz.images(
            images,
            opts=dict(title='%s/%d' % (tag, step), caption='%s/%d' % (tag, step)),
            nrow=nrow
        )


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint(state, check_list, log_dir, epoch=0):
    check_file = os.path.join(log_dir, 'model_{}.ckpt'.format(epoch))
    torch.save(state, check_file)
    check_list.write('model_{}.ckpt\n'.format(epoch))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def add_logs(args, logger, tag, value, step):
    logger.add_scalar(tag, value, step)
