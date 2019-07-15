import argparse
from model import wideresnet
from model.preactresnet import get_preact_resnet
from model.densenet import get_densenet
from lib.utils.avgmeter import AverageMeter
from lib.dataloader import cifar10_dataset, cifar100_dataset, svhn_dataset, get_sl_sampler
import os
from os import path
import time
import shutil
import ast
from sklearn.metrics import roc_auc_score
from lib.utils.mixup import mixup_criterion, mixup_data


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


parser = argparse.ArgumentParser(description='Mixup and Manifold Mixup Based Supervised learning for '
                                             'WideResNet,PreActResNet,DenseNet in Cifar10 and Cifar100')
# Dataset Parameters
parser.add_argument('-bp', '--base_path', default="/data/fhz")
parser.add_argument('--dataset', default="Cifar10", type=str, help="The dataset name")
parser.add_argument('-is', "--image-size", default=[32, 32], type=arg_as_list,
                    metavar='Image Size List', help='the size of h * w for image')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# Train Strategy Parameters
parser.add_argument('-t', '--train-time', default=1, type=int,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--dp', '--data-parallel', action='store_false', help='Use Data Parallel')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-arg', action='store_false', help='if we not resume the argument')
# Mixup Strategy Parameters
parser.add_argument('--mixup', default=False, type=bool, help="use mixup method")
parser.add_argument('--manifold-mixup', default=False, type=bool, help="use manifold mixup method")
parser.add_argument('--mll', "--mixup-layer-list", default=[0, 2], type=arg_as_list,
                    help="The mixup layer list for manifold mixup strategy")
parser.add_argument('--ma', "--mixup-alpha", default=0.2, type=float, help="the lambda for mixup method")
# Deep Learning Model Parameters
parser.add_argument('--net-name', default="wideresnet", type=str, help="the name for network to use")
parser.add_argument('--depth', default=28, type=int, metavar='D', help="the depth of neural network")
parser.add_argument('--width', default=2, type=int, metavar='W', help="the width of neural network")
parser.add_argument('--dr', '--drop-rate', default=0, type=float, help='dropout rate')
# Optimizer Parameters
parser.add_argument('--optimizer', default="SGD", type=str, metavar="Optimizer Name")
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M', help='Momentum in SGD')
parser.add_argument('--nesterov', action='store_true', help='nesterov in sgd')
parser.add_argument('-ad', "--adjust-lr", default=[60, 120, 160], type=arg_as_list,
                    help="The milestone list for adjust learning rate")
parser.add_argument('--lr-decay-ratio', default=0.2, type=float)
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)
parser.add_argument('--wul', '--warm-up-lr', default=0.02, type=float, help='the learning rate for warm up method')
# GPU Parameters
parser.add_argument("--gpu", default="0,1", type=str, metavar='GPU plans to use', help='The GPU id plans to use')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch.utils.data import DataLoader
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import torch.nn as nn


def main(args=args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # build dataset
    if args.dataset == "Cifar10":
        dataset_base_path = path.join(args.base_path, "dataset", "cifar")
        train_dataset = cifar10_dataset(dataset_base_path)
        test_dataset = cifar10_dataset(dataset_base_path, train_flag=False)
        sampler_valid, sampler_train, sampler_train = get_sl_sampler(
            torch.tensor(train_dataset.train_labels, dtype=torch.int32), 500, 10)
        test_dloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        valid_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                                   sampler=sampler_valid)
        train_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                   pin_memory=True,
                                   sampler=sampler_train)
        num_classes = 10
    elif args.dataset == "Cifar100":
        dataset_base_path = path.join(args.base_path, "dataset", "cifar")
        train_dataset = cifar100_dataset(dataset_base_path)
        test_dataset = cifar100_dataset(dataset_base_path, train_flag=False)
        sampler_valid, sampler_train, sampler_train = get_sl_sampler(
            torch.tensor(train_dataset.train_labels, dtype=torch.int32), 50, 100)
        test_dloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        valid_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                                   sampler=sampler_valid)
        train_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                   pin_memory=True,
                                   sampler=sampler_train)
        num_classes = 100
    elif args.dataset == "SVHN":
        dataset_base_path = path.join(args.base_path, "dataset", "svhn")
        train_dataset = svhn_dataset(dataset_base_path)
        test_dataset = svhn_dataset(dataset_base_path, train_flag=False)
        sampler_valid, sampler_train = get_sl_sampler(
            torch.tensor(train_dataset.labels, dtype=torch.int32), 732, 10)
        test_dloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        valid_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                                   sampler=sampler_valid)
        train_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                   pin_memory=True,
                                   sampler=sampler_train)
        num_classes = 10
    else:
        raise NotImplementedError("Dataset {} Not Implemented".format(args.dataset))
    if args.net_name == "wideresnet":
        model = wideresnet.WideResNet(depth=args.depth, width=args.width, img_size=tuple(args.image_size),
                                      num_classes=num_classes, data_parallel=args.dp, drop_rate=args.dr)
    elif "preact" in args.net_name:
        model = get_preact_resnet(args.net_name, img_size=tuple(args.image_size), num_classes=num_classes,
                                  data_parallel=args.dp,
                                  drop_rate=args.dr)
    elif "densenet" in args.net_name:
        model = get_densenet(args.net_name, img_size=tuple(args.image_size), num_classes=num_classes,
                             data_parallel=args.dp, drop_rate=args.dr)
    else:
        raise NotImplementedError("model {} not implemented".format(args.net_name))
    model = model.cuda()

    input("Begin the {} time's training, Dataset:{} Mixup Method:{} \
    Manifold Mixup Method :{}".format(args.train_time, args.dataset, args.mixup, args.manifold_mixup))
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd,
                                    nesterov=args.nesterov)
    else:
        raise NotImplementedError("{} not find".format(args.optimizer))
    scheduler = MultiStepLR(optimizer, milestones=args.adjust_lr, gamma=args.lr_decay_ratio)
    writer_log_dir = "{}/{}/runs/train_time:{}".format(args.base_path, args.dataset, args.train_time)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if  args.resume_arg:
                args = checkpoint['args']
                args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise FileNotFoundError("Checkpoint Resume File {} Not Found".format(args.resume))
    else:
        if os.path.exists(writer_log_dir):
            flag = input("{} train_time:{} will be removed, input yes to continue:".format(
                args.dataset, args.train_time))
            if flag == "yes":
                shutil.rmtree(writer_log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=writer_log_dir)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step(epoch)
        if epoch == 0:
            # do warmup
            modify_lr_rate(opt=optimizer, lr=args.wul)
        train(train_dloader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, writer=writer)
        test(valid_dloader, test_dloader, model=model, criterion=criterion, epoch=epoch, writer=writer,
             num_classes=num_classes)
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            "state_dict": model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })
        if epoch == 0:
            modify_lr_rate(opt=optimizer, lr=args.lr)


def train(train_dloader, model, criterion, optimizer, epoch, writer):
    # some records
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    optimizer.zero_grad()
    for i, (image, label) in enumerate(train_dloader):
        data_time.update(time.time() - end)
        image = image.float().cuda()
        label = label.long().cuda()
        if args.mixup:
            mixed_image, label_a, label_b, lam = mixup_data(image, label, args.ma)
            cls_result = model(mixed_image)
            loss = mixup_criterion(criterion, cls_result, label_a, label_b, lam)
        elif args.manifold_mixup:
            cls_result, label_a, label_b, lam = model(image, mixup_alpha=args.ma, label=label, manifold_mixup=True,
                                                      mixup_layer_list=args.mll)
            loss = mixup_criterion(criterion, cls_result, label_a, label_b, lam)
        else:
            cls_result = model(image)
            loss = criterion(cls_result, label)
        loss.backward()
        losses.update(float(loss.item()), image.size(0))
        optimizer.step()
        optimizer.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            train_text = 'Epoch: [{0}][{1}/{2}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                         'Cls Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})'.format(
                epoch, i + 1, len(train_dloader), batch_time=batch_time, data_time=data_time,
                cls_loss=losses)
            print(train_text)
    writer.add_scalar(tag="Train/cls_loss", scalar_value=losses.avg, global_step=epoch + 1)
    return losses.avg


def test(valid_dloader, test_dloader, model, criterion, epoch, writer, num_classes):
    model.eval()
    # calculate index for valid dataset
    losses = AverageMeter()
    all_score = []
    all_label = []
    for i, (image, label) in enumerate(valid_dloader):
        image = image.float().cuda()
        label = label.long().cuda()
        with torch.no_grad():
            cls_result = model(image)
        label_onehot = torch.zeros(label.size(0), num_classes).cuda().scatter_(1, label.view(-1, 1), 1)
        loss = criterion(cls_result, label)
        losses.update(float(loss.item()), image.size(0))
        # here we add the all score and all label into one list
        all_score.append(torch.softmax(cls_result, dim=1))
        # turn label into one-hot code
        all_label.append(label_onehot)
    writer.add_scalar(tag="Valid/cls_loss", scalar_value=losses.avg, global_step=epoch + 1)
    all_score = torch.cat(all_score, dim=0).detach()
    all_label = torch.cat(all_label, dim=0).detach()
    _, y_true = torch.topk(all_label, k=1, dim=1)
    _, y_pred = torch.topk(all_score, k=5, dim=1)
    top_1_accuracy = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    top_5_accuracy = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
    writer.add_scalar(tag="Valid/top 1 accuracy", scalar_value=top_1_accuracy, global_step=epoch + 1)
    if args.dataset == "Cifar100":
        writer.add_scalar(tag="Valid/top 5 accuracy", scalar_value=top_5_accuracy, global_step=epoch + 1)
    # calculate index for test dataset
    losses = AverageMeter()
    all_score = []
    all_label = []
    # don't use roc
    # roc_list = []
    for i, (image, label) in enumerate(test_dloader):
        image = image.float().cuda()
        label = label.long().cuda()
        with torch.no_grad():
            cls_result = model(image)
        label_onehot = torch.zeros(label.size(0), num_classes).cuda().scatter_(1, label.view(-1, 1), 1)
        loss = criterion(cls_result, label)
        losses.update(float(loss.item()), image.size(0))
        # here we add the all score and all label into one list
        all_score.append(torch.softmax(cls_result, dim=1))
        # turn label into one-hot code
        all_label.append(label_onehot)
    writer.add_scalar(tag="Test/cls_loss", scalar_value=losses.avg, global_step=epoch + 1)
    all_score = torch.cat(all_score, dim=0).detach()
    all_label = torch.cat(all_label, dim=0).detach()
    _, y_true = torch.topk(all_label, k=1, dim=1)
    _, y_pred = torch.topk(all_score, k=5, dim=1)
    # don't use roc auc
    # all_score = all_score.cpu().numpy()
    # all_label = all_label.cpu().numpy()
    # for i in range(num_classes):
    #     roc_list.append(roc_auc_score(all_label[:, i], all_score[:, i]))
    # ap_list.append(average_precision_score(all_label[:, i], all_score[:, i]))
    # calculate accuracy by hand
    top_1_accuracy = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    top_5_accuracy = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
    writer.add_scalar(tag="Test/top 1 accuracy", scalar_value=top_1_accuracy, global_step=epoch + 1)
    if args.dataset == "Cifar100":
        writer.add_scalar(tag="Test/top 5 accuracy", scalar_value=top_5_accuracy, global_step=epoch + 1)
    # writer.add_scalar(tag="Test/mean RoC", scalar_value=mean(roc_list), global_step=epoch + 1)
    return top_1_accuracy


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    :param state: a dict including:{
                'epoch': epoch + 1,
                'args': args,
                "state_dict": model.state_dict(),
                'optimizer': optimizer.state_dict(),
        }
    :param filename: the filename for store
    :return:
    """
    filefolder = "{}/{}/parameter/train_time:{}".format(args.base_path, args.dataset, args.train_time)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))


def modify_lr_rate(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
