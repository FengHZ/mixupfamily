import argparse
from model import wideresnet
from model.preactresnet import get_preact_resnet
from model.densenet import get_densenet
from lib.utils.avgmeter import AverageMeter
from lib.utils.zca import apply_zca
from lib.dataloader import cifar10_dataset, get_ssl_sampler, cifar100_dataset, svhn_dataset
import os
from os import path
import time
import shutil
import ast
import numpy as np
from lib.utils.mixup import mixup_criterion, mixup_data
from itertools import cycle
import math


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


parser = argparse.ArgumentParser(description='Mixup and Manifold Mixup Based Semi-supervised learning for '
                                             'WideResNet,PreActResNet,DenseNet in Cifar10 and Cifar100')
# Dataset Parameters
parser.add_argument('-bp', '--base_path', default="/data/fhz")
parser.add_argument('--dataset', default="Cifar10", type=str, help="The dataset name")
parser.add_argument('--zca', action='store_true', help='if we use zca preprocess')
parser.add_argument('-is', "--image-size", default=[32, 32], type=arg_as_list,
                    metavar='Image Size List', help='the size of h * w for image')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# Semi-supervised Train Strategy Parameters
parser.add_argument('-t', '--train-time', default=1, type=int,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--epochs', default=600, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-am', "--alpha-max", default=10, type=float,
                    help="the max weight(alpha) for the balance between supervised and unsupervised loss")
parser.add_argument('-amf', '--alpha-modify-factor', default=0.4, type=float,
                    help="weight(alpha) will get alpha-max at amf * epochs")
parser.add_argument('--dp', '--data-parallel', action='store_false', help='Not Use Data Parallel')
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
parser.add_argument('--mas', "--mixup-alpha-supervised", default=0.1, type=float,
                    help="the alpha for supervised mixup method")
parser.add_argument('--mau', "--mixup-alpha-unsupervised", default=2, type=float,
                    help="the alpha for unsupervised mixup method")
# Deep Learning Model Parameters
parser.add_argument('--net-name', default="wideresnet", type=str, help="the name for network to use")
parser.add_argument('--depth', default=28, type=int, metavar='D', help="the depth of neural network")
parser.add_argument('--width', default=2, type=int, metavar='W', help="the width of neural network")
parser.add_argument('--dr', '--drop-rate', default=0, type=float, help='Dropout Rate usually 0 when use mixup method')
# Optimizer Parameters
parser.add_argument('--optimizer', default="SGD", type=str, metavar="Optimizer Name")
parser.add_argument('--nesterov', action='store_true', help='nesterov in sgd')
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M', help='Momentum in SGD')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-ad', "--adjust-lr", default=[300, 400, 500], type=arg_as_list,
                    help="The milestone list for adjust learning rate")
parser.add_argument('--lr-decay-ratio', default=0.2, type=float)
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)
parser.add_argument('--wul', '--warm-up-lr', default=0.02, type=float, help='the learning rate for warm up method')
# GPU Parameters
parser.add_argument("--gpu", default="0,1", type=str, metavar='GPU plans to use', help='The GPU id plans to use')
# Temp Parameters
parser.add_argument("--flag-reg", action='store_true',
                    help="tmp parameters control regularization loss in unsupervised loss")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn


def main(args=args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    zca_mean = None
    zca_components = None

    # build dataset
    if args.dataset == "Cifar10":
        dataset_base_path = path.join(args.base_path, "dataset", "cifar")
        train_dataset = cifar10_dataset(dataset_base_path)
        test_dataset = cifar10_dataset(dataset_base_path, train_flag=False)
        sampler_valid, sampler_train_l, sampler_train_u = get_ssl_sampler(
            torch.tensor(train_dataset.train_labels, dtype=torch.int32), 500, 400, 10)
        test_dloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        valid_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                                   sampler=sampler_valid)
        train_dloader_l = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                     pin_memory=True,
                                     sampler=sampler_train_l)
        train_dloader_u = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                     pin_memory=True,
                                     sampler=sampler_train_u)
        num_classes = 10
        if args.zca:
            zca_mean = np.load(os.path.join(dataset_base_path, 'cifar10_zca_mean.npy'))
            zca_components = np.load(os.path.join(dataset_base_path, 'cifar10_zca_components.npy'))
            zca_mean = torch.from_numpy(zca_mean).view(1, -1).float().cuda()
            zca_components = torch.from_numpy(zca_components).float().cuda()
    elif args.dataset == "Cifar100":
        dataset_base_path = path.join(args.base_path, "dataset", "cifar")
        train_dataset = cifar100_dataset(dataset_base_path)
        test_dataset = cifar100_dataset(dataset_base_path, train_flag=False)
        sampler_valid, sampler_train_l, sampler_train_u = get_ssl_sampler(
            torch.tensor(train_dataset.train_labels, dtype=torch.int32), 50, 40, 100)
        test_dloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        valid_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                                   sampler=sampler_valid)
        train_dloader_l = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                     pin_memory=True,
                                     sampler=sampler_train_l)
        train_dloader_u = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                     pin_memory=True,
                                     sampler=sampler_train_u)
        num_classes = 100
    elif args.dataset == "SVHN":
        dataset_base_path = path.join(args.base_path, "dataset", "svhn")
        train_dataset = svhn_dataset(dataset_base_path)
        test_dataset = svhn_dataset(dataset_base_path, train_flag=False)
        sampler_valid, sampler_train_l, sampler_train_u = get_ssl_sampler(
            torch.tensor(train_dataset.labels, dtype=torch.int32), 732, 100, 10)
        test_dloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        valid_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                                   sampler=sampler_valid)
        train_dloader_l = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                     pin_memory=True,
                                     sampler=sampler_train_l)
        train_dloader_u = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                     pin_memory=True,
                                     sampler=sampler_train_u)
        num_classes = 10
    else:
        raise NotImplementedError("Dataset {} Not Implemented".format(args.dataset))
    if args.net_name == "wideresnet":
        model = wideresnet.WideResNet(depth=args.depth, width=args.width,
                                      num_classes=num_classes, data_parallel=args.dp, drop_rate=args.dr)
    elif "preact" in args.net_name:
        model = get_preact_resnet(args.net_name, num_classes=num_classes,
                                  data_parallel=args.dp,
                                  drop_rate=args.dr)
    elif "densenet" in args.net_name:
        model = get_densenet(args.net_name, num_classes=num_classes,
                             data_parallel=args.dp, drop_rate=args.dr)
    else:
        raise NotImplementedError("model {} not implemented".format(args.net_name))
    model = model.cuda()

    input("Begin the {} time's semi-supervised training, Dataset:{} Mixup Method:{} \
    Manifold Mixup Method :{}".format(args.train_time, args.dataset, args.mixup, args.manifold_mixup))
    criterion_l = nn.CrossEntropyLoss()
    criterion_u = nn.MSELoss()
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd,
                                    nesterov=args.nesterov)
    else:
        raise NotImplementedError("{} not find".format(args.optimizer))
    scheduler = MultiStepLR(optimizer, milestones=args.adjust_lr, gamma=args.lr_decay_ratio)
    writer_log_dir = "{}/{}-SSL/runs/train_time:{}".format(args.base_path, args.dataset, args.train_time)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.resume_arg:
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
            flag = input("{}-SSL train_time:{} will be removed, input yes to continue:".format(
                args.dataset, args.train_time))
            if flag == "yes":
                shutil.rmtree(writer_log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=writer_log_dir)
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step(epoch)
        if epoch == 0:
            # do warm up
            modify_lr_rate(opt=optimizer, lr=args.wul)
        alpha = alpha_schedule(epoch=epoch)
        train(train_dloader_l, train_dloader_u, model=model, criterion_l=criterion_l, criterion_u=criterion_u,
              optimizer=optimizer, epoch=epoch, writer=writer, alpha=alpha, zca_mean=zca_mean,
              zca_components=zca_components)
        test(valid_dloader, test_dloader, model=model, criterion=criterion_l, epoch=epoch, writer=writer,
             num_classes=num_classes, zca_mean=zca_mean, zca_components=zca_components)
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            "state_dict": model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })
        if epoch == 0:
            modify_lr_rate(opt=optimizer, lr=args.lr)


def train(train_dloader_l, train_dloader_u, model, criterion_l, criterion_u, optimizer, epoch, writer, alpha,
          zca_mean=None, zca_components=None):
    # some records
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_supervised = AverageMeter()
    losses_unsupervised = AverageMeter()
    model.train()
    end = time.time()
    optimizer.zero_grad()
    epoch_length = len(train_dloader_u)
    i = 0
    for (image_l, label_l), (image_u, _) in zip(cycle(train_dloader_l), train_dloader_u):
        if image_l.size(0) != image_u.size(0):
            bt_size = min(image_l.size(0), image_u.size(0))
            image_l = image_l[0:bt_size]
            image_u = image_u[0:bt_size]
            label_l = label_l[0:bt_size]
        else:
            bt_size = image_l.size(0)
        data_time.update(time.time() - end)
        image_l = image_l.float().cuda()
        image_u = image_u.float().cuda()
        label_l = label_l.long().cuda()
        if args.zca:
            image_l = apply_zca(image_l, zca_mean, zca_components)
            image_u = apply_zca(image_u, zca_mean, zca_components)
        if args.mixup:
            mixed_image_l, label_a, label_b, lam = mixup_data(image_l, label_l, args.mas)
            cls_result_l = model(mixed_image_l)
            loss_supervised = mixup_criterion(criterion_l, cls_result_l, label_a, label_b, lam)
            # here label_u_approx is not with any grad
            with torch.no_grad():
                if args.flag_reg:
                    label_u_approx = model(image_u)
                else:
                    label_u_approx = torch.softmax(model(image_u), dim=1)
            mixed_image_u, label_a_approx, label_b_approx, lam = mixup_data(image_u, label_u_approx, args.mau)
            cls_result_u = torch.softmax(model(mixed_image_u), dim=1)
            if args.flag_reg:
                cls_result_u = cls_result_u
            else:
                cls_result_u = torch.softmax(cls_result_u, dim=1)
            label_u_approx_mixup = lam * label_a_approx + (1 - lam) * label_b_approx
            loss_unsupervised = criterion_u(cls_result_u, label_u_approx_mixup)
            loss = loss_supervised + alpha * loss_unsupervised
        elif args.manifold_mixup:
            cls_result_l, label_a, label_b, lam = model(image_l, mixup_alpha=args.mas, label=label_l,
                                                        manifold_mixup=True,
                                                        mixup_layer_list=args.mll)
            loss_supervised = mixup_criterion(criterion_l, cls_result_l, label_a, label_b, lam)
            # here label_u_approx is not with any grad
            with torch.no_grad():
                if args.flag_reg:
                    label_u_approx = model(image_u)
                else:
                    label_u_approx = torch.softmax(model(image_u), dim=1)
            cls_result_u, label_a_approx, label_b_approx, lam = model(image_u, mixup_alpha=args.mas,
                                                                      label=label_u_approx,
                                                                      manifold_mixup=True,
                                                                      mixup_layer_list=args.mll)
            if args.flag_reg:
                cls_result_u = cls_result_u
            else:
                cls_result_u = torch.softmax(cls_result_u, dim=1)
            label_u_approx_mixup = lam * label_a_approx + (1 - lam) * label_b_approx
            loss_unsupervised = criterion_u(cls_result_u, label_u_approx_mixup)
            loss = loss_supervised + alpha * loss_unsupervised
        else:
            cls_result_l = model(image_l)
            loss = criterion_l(cls_result_l, label_l)
            loss_supervised = loss.detach()
            loss_unsupervised = torch.zeros(loss.size())
        loss.backward()
        losses.update(float(loss.item()), bt_size)
        losses_supervised.update(float(loss_supervised.item()), bt_size)
        losses_unsupervised.update(float(loss_unsupervised.item()), bt_size)
        optimizer.step()
        optimizer.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            train_text = 'Epoch: [{0}][{1}/{2}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                         'Cls Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t' \
                         'Regularization Loss {reg_loss.val:.4f} ({reg_loss.avg:.4f})\t' \
                         'Total Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t'.format(
                epoch, i + 1, epoch_length, batch_time=batch_time, data_time=data_time,
                cls_loss=losses_supervised, reg_loss=losses_unsupervised, total_loss=losses)
            print(train_text)
        i += 1
    writer.add_scalar(tag="Train/cls_loss", scalar_value=losses_supervised.avg, global_step=epoch + 1)
    writer.add_scalar(tag="Train/reg_loss", scalar_value=losses_unsupervised.avg, global_step=epoch + 1)
    writer.add_scalar(tag="Train/total_loss", scalar_value=losses.avg, global_step=epoch + 1)
    return losses.avg


def test(valid_dloader, test_dloader, model, criterion, epoch, writer, num_classes, zca_mean=None, zca_components=None):
    model.eval()
    # calculate result for valid dataset
    losses = AverageMeter()
    all_score = []
    all_label = []
    for i, (image, label) in enumerate(valid_dloader):
        image = image.float().cuda()
        if args.zca:
            image = apply_zca(image, zca_mean, zca_components)
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
    # calculate accuracy by hand
    losses = AverageMeter()
    all_score = []
    all_label = []
    top_1_accuracy = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    top_5_accuracy = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
    writer.add_scalar(tag="Valid/top 1 accuracy", scalar_value=top_1_accuracy, global_step=epoch + 1)
    if args.dataset == "Cifar100":
        writer.add_scalar(tag="Valid/top 5 accuracy", scalar_value=top_5_accuracy, global_step=epoch + 1)
    # calculate result for test dataset
    for i, (image, label) in enumerate(test_dloader):
        image = image.float().cuda()
        if args.zca:
            image = apply_zca(image, zca_mean, zca_components)
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
    # calculate accuracy by hand
    top_1_accuracy = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    top_5_accuracy = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
    writer.add_scalar(tag="Test/top 1 accuracy", scalar_value=top_1_accuracy, global_step=epoch + 1)
    if args.dataset == "Cifar100":
        writer.add_scalar(tag="Test/top 5 accuracy", scalar_value=top_5_accuracy, global_step=epoch + 1)

    return losses.avg


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
    filefolder = "{}/{}-SSL/parameter/train_time:{}".format(args.base_path, args.dataset, args.train_time)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))


def modify_lr_rate(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr


def alpha_schedule(epoch):
    max_epoch = args.alpha_modify_factor * args.epochs
    alpha = args.alpha_max * math.exp(-5 * (1 - min(1, epoch / max_epoch)) ** 2)
    return alpha


if __name__ == "__main__":
    main()
