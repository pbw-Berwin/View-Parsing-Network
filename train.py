from utils import Foo
from models import VPNModel
from datasets import OVMDataset
from opts import parser
from transform import *
import torchvision
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim
import os
import time
from torch.nn.utils import clip_grad_norm
# from examples.cognitive_mapping.Logger import Logger
import cv2
import shutil

mean_rgb = [0.485, 0.456, 0.406]
std_rgb = [0.229, 0.224, 0.225]

def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()
    network_config = Foo(
        encoder=args.encoder,
        decoder=args.decoder,
        fc_dim=args.fc_dim,
        output_size=args.label_resolution,
        num_views=args.n_views,
        num_class=args.num_class,
        transform_type=args.transform_type,
    )
    train_dataset = OVMDataset(args.data_root, args.train_list,
                         transform=torchvision.transforms.Compose([
                             Stack(roll=True),
                             ToTorchFormatTensor(div=True),
                             GroupNormalize(mean_rgb, std_rgb)
                         ]),
                         num_views=network_config.num_views, input_size=args.input_resolution,
                         label_size=args.label_resolution, use_mask=args.use_mask, use_depth=args.use_depth)
    val_dataset = OVMDataset(args.data_root, args.eval_list,
                         transform=torchvision.transforms.Compose([
                             Stack(roll=True),
                             ToTorchFormatTensor(div=True),
                             GroupNormalize(mean_rgb, std_rgb)
                         ]),
                         num_views=network_config.num_views, input_size=args.input_resolution,
                         label_size=args.label_resolution, use_mask=args.use_mask, use_depth=args.use_depth)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
        pin_memory=True
    )

    mapper = VPNModel(network_config)
    mapper = nn.DataParallel(mapper.cuda())

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            mapper.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))


    criterion = nn.NLLLoss(weight=None, size_average=True)
    optimizer = optim.Adam(mapper.parameters(),
                          lr=args.start_lr, betas=(0.95, 0.999))

    if not os.path.isdir(args.log_root):
        os.mkdir(args.log_root)
    log_train = open(os.path.join(args.log_root, '%s.csv' % args.store_name), 'w')

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        train(train_loader, mapper, criterion, optimizer, epoch, log_train)

        if (epoch + 1) % args.ckpt_freq == 0 or epoch == args.epochs - 1:
            prec1 = eval(val_loader, mapper, criterion, log_train, epoch)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': network_config.encoder,
                'state_dict': mapper.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)

def train(train_loader, mapper, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    mapper.train()

    end = time.time()
    for step, data in enumerate(train_loader):
        rgb_stack, target = data
        data_time.update(time.time() - end)
        target_var = target.cuda()
        input_rgb_var = torch.autograd.Variable(rgb_stack).cuda()
        output = mapper(input_rgb_var)
        target_var = target_var.view(-1)
        output = output.view(-1, args.num_class)
        loss = criterion(output, target_var)
        losses.update(loss.data[0], input_rgb_var.size(0))
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        top1.update(prec1[0], rgb_stack.size(0))
        top5.update(prec5[0], rgb_stack.size(0))

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch + 1, step + 1, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            log.write(output + '\n')
            log.flush()

def eval(val_loader, mapper, criterion, log, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    mapper.eval()

    end = time.time()
    for step, (rgb_stack, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            input_rgb_var = torch.autograd.Variable(rgb_stack).cuda()
            output = mapper(input_rgb_var)
        target_var = target.cuda()
        target_var = target_var.view(-1)
        output = output.view(-1, args.num_class)
        loss = criterion(output, target_var)
        losses.update(loss.data[0], input_rgb_var.size(0))
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        top1.update(prec1[0], rgb_stack.size(0))
        top5.update(prec5[0], rgb_stack.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            output = ('Test: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch + 1, step + 1, len(val_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
            print(output)
            log.write(output + '\n')
            log.flush()
    output = ('Testing Results: Prec@1 {top1.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f' % (best_prec1)
    print(output_best)
    log.write(output + ' ' + output_best + '\n')
    log.flush()
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name), '%s/%s_best.pth.tar' % (args.root_model, args.store_name))


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.start_lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = decay

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
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__=='__main__':
    main()
