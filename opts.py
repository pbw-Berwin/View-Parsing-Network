import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description="PyTorch implementation of CMP")
parser.add_argument('--data_root', type=str, default='/data/vision/oliva/scenedataset/syntheticscene/TopViewMaskDataset')
parser.add_argument('--test-dir', type=str, default='')
parser.add_argument('--train-list', type=str, default='./metadata/train_list.txt')
parser.add_argument('--eval-list', type=str, default='./metadata/val_list.txt')
parser.add_argument('--start_lr', type=float, default=2e-4)
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--transform-type', type=str, default='fc')
parser.add_argument('--use-mask', type=str2bool, nargs='?', const=False)
parser.add_argument('--use-depth', type=str2bool, nargs='?', const=False)
parser.add_argument('--input-resolution', default=400, type=int, metavar='N')
parser.add_argument('--label-resolution', default=25, type=int, metavar='N')
parser.add_argument('--fc-dim', default=256, type=int, metavar='N')
parser.add_argument('--segSize', default=256, type=int, metavar='N')
parser.add_argument('--log-root', type=str, default='./log')
parser.add_argument('--encoder', type=str, default='resnet18')
parser.add_argument('--decoder', type=str, default='ppm_bilinear')
parser.add_argument('--store-name', type=str, default='')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--n-views', type=int, default=8)
parser.add_argument('--lr_steps', default=[10], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--print-img-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--ckpt-freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 2)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tflogdir', default='', type=str, metavar="PATH")
parser.add_argument('--logname', type=str, default="")
parser.add_argument('-b', '--batch-size', default=104, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--scale_size', default=224, type=int)
parser.add_argument('--num-class', default=94, type=int)
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--root_model', type=str, default='model')
parser.add_argument('--weights', type=str,
                    default='')
parser.add_argument('--visualize', type=str, default='./visualize')
parser.add_argument('--centralSize', type=int, default=12)
parser.add_argument('--mapSize', type=int, default=1000)
parser.add_argument('--ppi', type=int, default=4)
parser.add_argument('--scale', type=float, default=2)
parser.add_argument('--trajectory-file', type=str, default='')
parser.add_argument('--real-scale', type=float, default=1.2)
parser.add_argument('--use_topdown', type=str2bool, default=False)
parser.add_argument('--visual_input', type=str2bool, default=False)
