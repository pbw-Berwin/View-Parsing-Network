from utils import Foo
from models import VPNModel, FCDiscriminator
from datasets import House3D_Dataset, MP3D_Dataset
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
import torch.nn.functional as F
import os.path as osp
from tensorboardX import SummaryWriter

mean_rgb = [0.485, 0.456, 0.406]
std_rgb = [0.229, 0.224, 0.225]


def main():
    global args, best_prec1
    best_prec1 = 0

    parser.add_argument('--source_dir', type=str,
                        default='/mnt/lustre/share/VPN_driving_scene/TopViewMaskDataset')
    parser.add_argument('--target_dir', type=str,
                        default='/mnt/lustre/share/VPN_driving_scene/mp3d')
    parser.add_argument('--num-steps', type=int, default=250000)
    parser.add_argument('--iter-size', type=int, default=1)
    parser.add_argument('--learning-rate-D', type=float, default=1e-4)
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument("--SegSize", type=int, default=128,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--SegSize-target", type=int, default=128,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument('--resume-D', type=str)
    parser.add_argument('--resume-G', type=str)
    parser.add_argument('--train_source_list', type=str, default='./train_source_list.txt')
    parser.add_argument('--train_target_list', type=str, default='./train_target_list.txt')
    parser.add_argument('--num-classes', type=int, default=94)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--lambda_adv_target', type=float, default=0.001)
    parser.add_argument('--num_steps_stop', type=int, default=150000)
    parser.add_argument('--save_pred_every', type=int, default=5000)
    parser.add_argument('--snapshot-dir', type=str, default='/mnt/lustre/panbowen/VPN-transfer/snapshot/')
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--tf_logdir", type=str, default='/mnt/lustre/panbowen/VPN-transfer/tf_log/',
                        help="Path to the directory of log.")


    args = parser.parse_args()

    network_config = Foo(
        encoder=args.encoder,
        decoder=args.decoder,
        fc_dim=args.fc_dim,
        output_size=args.label_resolution,
        num_views=args.n_views,
        num_class=94,
        transform_type=args.transform_type,
    )

    train_source_dataset = House3D_Dataset(args.source_dir, args.train_source_list,
                        transform=torchvision.transforms.Compose([
                             Stack(roll=True),
                             ToTorchFormatTensor(div=True),
                             GroupNormalize(mean_rgb, std_rgb)
                        ]),
                        num_views=network_config.num_views, input_size=args.input_resolution,
                        label_size=args.SegSize)

    train_target_dataset = MP3D_Dataset(args.target_dir, args.train_target_list,
                         transform=torchvision.transforms.Compose([
                            Stack(roll=True),
                            ToTorchFormatTensor(div=True),
                            GroupNormalize(mean_rgb, std_rgb)
                        ]),
                        num_views=network_config.num_views, input_size=args.input_resolution,
                        label_size=args.SegSize_target)

    source_loader = torch.utils.data.DataLoader(
        train_source_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True,
        pin_memory=True
    )

    target_loader = torch.utils.data.DataLoader(
        train_target_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True,
        pin_memory=True
    )

    mapper = VPNModel(network_config)
    mapper = nn.DataParallel(mapper.cuda())

    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D1 = nn.DataParallel(model_D1.cuda())
    # model_D2 = FCDiscriminator(num_classes=args.num_classes).cuda()
    model_D1.train()
    # model_D1.to(device)
    # model_D2.train()
    # model_D2.to(device)

    if args.resume_G:
        if os.path.isfile(args.resume_G):
            print(("=> loading checkpoint '{}'".format(args.resume_G)))
            checkpoint = torch.load(args.resume_G)
            args.start_epoch = checkpoint['epoch']
            mapper.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume_G)))

    if args.resume_D:
        if os.path.isfile(args.resume_D):
            print(("=> loading checkpoint '{}'".format(args.resume_D)))
            checkpoint = torch.load(args.resume_D)
            args.start_epoch = checkpoint['epoch']
            model_D1.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume_D)))

    optimizer = optim.SGD(mapper.parameters(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    # optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.start_lr, betas=(0.9, 0.99))
    # optimizer_D2.zero_grad()


    criterion_seg = nn.NLLLoss(weight=None, size_average=True)
    criterion_bce = nn.BCEWithLogitsLoss()

    if not os.path.isdir(args.log_root):
        os.mkdir(args.log_root)
    log_train = open(os.path.join(args.log_root, '%s.csv' % args.store_name), 'w')

    train(source_loader, target_loader, mapper, model_D1, criterion_seg, criterion_bce, optimizer, optimizer_D1, log_train)

def train(source_loader, target_loader, mapper, model_D1, seg_loss, bce_loss, optimizer, optimizer_D1, log):
    source_loader_iter = enumerate(source_loader)

    # raise NotImplementedError
    target_loader_iter = enumerate(target_loader)

    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)

    interp = nn.Upsample(size=(args.SegSize, args.SegSize), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(args.SegSize_target, args.SegSize_target), mode='bilinear', align_corners=True)
    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):
        loss_seg_value = 0
        loss_adv_target_value = 0
        loss_D_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)


        for sub_i in range(args.iter_size):
            # train G
            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False
            # train with source

            # raise NotImplementedError
            try:
                _, batch = source_loader_iter.__next__()
            except:
                source_loader_iter = enumerate(source_loader)
                _, batch = source_loader_iter.__next__()
            rgb_stack, label = batch
            label_var = label.cuda()
            input_rgb_var = torch.autograd.Variable(rgb_stack).cuda()

            _, pred_feat = mapper(input_rgb_var, return_feat=True)
            pred_feat = pred_feat.transpose(3, 2).transpose(2, 1).contiguous()
            pred = interp(pred_feat)

            pred = F.log_softmax(pred, dim=1)
            pred.transpose(1, 2).transpose(2, 3).contiguous()
            label_var = label_var.view(-1)
            output = pred.view(-1, args.num_class)

            loss_seg = seg_loss(output, label_var)
            loss = loss_seg / args.iter_size
            loss.backward()
            loss_seg_value += loss_seg.item() / args.iter_size

            # train with target
            try:
                _, batch = target_loader_iter.__next__()
            except:
                target_loader_iter = enumerate(target_loader)
                _, batch = target_loader_iter.__next__()
            rgb_stack = batch
            input_rgb_var = torch.autograd.Variable(rgb_stack).cuda()
            _, pred_target = mapper(input_rgb_var, return_feat=True)
            pred_target = pred_target.transpose(3, 2).transpose(2, 1).contiguous()

            # pred_t = interp_target(pred_t)
            pred_target = interp_target(pred_target)

            D_out = model_D1(torch.exp(pred_target))

            loss_adv_target = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())
            loss = args.lambda_adv_target * loss_adv_target / args.iter_size
            loss.backward()
            loss_adv_target_value += loss_adv_target.item() / args.iter_size

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            # train with source

            pred = pred.detach()

            D_out = model_D1(pred)
            loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())
            loss_D = loss_D / args.iter_size / 2
            loss_D.backward()

            loss_D_value += loss_D.item()

            # train with target
            pred_target = pred_target.detach()

            D_out = model_D1(pred_target)

            loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).cuda())

            loss_D = loss_D / args.iter_size / 2
            loss_D.backward()
            loss_D_value += loss_D.item()
        optimizer.step()
        optimizer_D1.step()

        if args.tensorboard:
            scalar_info = {
                'loss_seg': loss_seg_value,
                'loss_adv': loss_adv_target_value,
                'loss_D': loss_D_value,
            }

            if i_iter % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        print(
        'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f} loss_adv = {3:.3f}, loss_D = {4:.3f} '.format(
            i_iter, args.num_steps, loss_seg_value, loss_adv_target_value, loss_D_value))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(mapper.state_dict(), osp.join(args.snapshot_dir, 'House3D_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'House3D_' + str(args.num_steps_stop) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(mapper.state_dict(), osp.join(args.snapshot_dir, 'House3D_' + str(i_iter) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'House3D_' + str(i_iter) + '_D.pth'))

    if args.tensorboard:
        writer.close()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name), '%s/%s_best.pth.tar' % (args.root_model, args.store_name))

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

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
