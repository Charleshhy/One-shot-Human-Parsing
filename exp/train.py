import socket
import timeit
from datetime import datetime
import os
import sys
import glob
import numpy as np

sys.path.append('../../')
sys.path.append('../../dataloaders/')
sys.path.append('../../networks/')
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from dataloaders import oshp_loader
from utils import util, get_iou_from_list
from networks import eopnet
from dataloaders import transforms as tr
import argparse
import random
from utils.util import get_beta, get_dataset_labels_classes, save_img
from utils.test_human import get_binary_iou

gpu_id = 0
nEpochs = 100  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume


def get_iou(pred_lis, label_lis, writer, epoch, classes, name,
            base_class=None, novel_class=None, class_labels=None, sem_class=None):
    assert base_class and novel_class and class_labels and sem_class

    meanIoU, iu, overall_acc, mean_acc = get_iou_from_list(pred_lis, label_lis, n_cls=classes)

    lis = iu[class_labels['all']]
    meanIoU = np.nanmean(lis)
    base_lis = [iu[i] for i in base_class]
    base_mean = np.nanmean(base_lis)
    novel_lis = [iu[i] for i in novel_class]
    novel_mean = np.nanmean(novel_lis)

    # Build text
    sem_lis = sem_class
    text = ''
    print('Validation per-class results: \n',
          ['{}:{}, '.format(i, j) for i, j in zip(sem_class, lis)])
    assert len(sem_class) == len(lis)

    for sem, IoU in zip(sem_lis, lis):
        text += sem + ": "
        text += str(IoU) + ', '

    writer.add_text('IoU', text, epoch)
    writer.add_scalar('data/test_miour_' + name, meanIoU, epoch)
    writer.add_scalars('data/cate_miou_' + name, {'base_mean': base_mean,
                                                  'novel_mean': novel_mean, }, epoch)
    writer.add_scalars('data/acc' + name, {'overall': overall_acc,
                                           'mean': mean_acc, }, epoch)

    print('{}: mean mIoU {}, base class mIoU {}, novel class mIoU {}'.format(name, meanIoU, base_mean, novel_mean))

    return meanIoU


def get_oneway_iou(pred_dic, lbl_dic, binary_pred_list, binary_label_list, writer, epoch, classes, name,
                   base_class=None, novel_class=None, dataset=None, class_labels=None, sem_class=None):

    assert base_class and novel_class and class_labels and sem_class
    fg_lis = []
    bg_lis = []

    for jj in range(classes):
        bg, fg = get_binary_iou(pred_dic[jj], lbl_dic[jj])
        fg_lis.append(fg)
        bg_lis.append(bg)

    iu = []
    bg_lis = [bg_lis[i] for i in class_labels['all']]
    iu.append(np.mean(bg_lis[1:]))
    iu += fg_lis[1:]

    lis = [iu[i] for i in class_labels['all']]
    meanIoU = np.mean(lis)

    base_lis = [iu[i] for i in base_class]
    base_mean = np.mean(base_lis)
    novel_lis = [iu[i] for i in novel_class]
    novel_mean = np.mean(novel_lis)

    assert len(sem_class) == len(lis)
    sem_lis = sem_class
    text = ''

    print('Validation per-class results: \n',
          ['{}:{}, '.format(i, j) for i, j in zip(sem_class, lis)])
    assert len(sem_class) == len(lis)

    for sem, IoU in zip(sem_lis, lis):
        text += sem + ": "
        text += str(IoU) + ', '

    writer.add_text('IoU', text, epoch)
    writer.add_scalar('data/test_miour_' + name, meanIoU, epoch)
    writer.add_scalars('data/cate_miou_' + name, {'base_mean': base_mean,
                                                  'novel_mean': novel_mean, }, epoch)

    bi_bg, bi_fg = get_binary_iou(binary_pred_list, binary_label_list)
    writer.add_scalars('data/bi_miour', {'bi_fg': bi_fg,
                                         'bi_bg': bi_bg,
                                         'bi_miou': (bi_fg + bi_bg) / 2}, epoch)

    print('{}: mean mIoU {}, base class mIoU {}, novel class mIoU {}, binary mIoU {}'.format(name, meanIoU, base_mean,
                                                                                             novel_mean, (bi_fg + bi_bg) / 2))

    return meanIoU


def get_parser():
    '''argparse begin'''
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--lr', default=1e-7, type=float)
    parser.add_argument('--numworker', default=12, type=int)
    parser.add_argument('--step', default=10, type=int)
    parser.add_argument('--testInterval', default=10, type=int)
    parser.add_argument('--load_model', default='', type=str)
    parser.add_argument('--resume_model', default='', type=str)
    parser.add_argument('--gpus', default=4, type=int)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--fg_weight', default=1.0, type=float)
    parser.add_argument('--contrast_weight', default=1.0, type=float)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--testing_screening', action='store_true',
                        help='screening noise using coarse-grained prediction')
    parser.add_argument('--temperature', default=1.0, type=float)

    # Dataset parameters
    parser.add_argument('--dataset', default='cihp', type=str)
    parser.add_argument('--test_classes', default=20, type=int)
    parser.add_argument('--poly', default='yes', type=str)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--way', default='kway', type=str)

    # Model parameters
    parser.add_argument('--hidden_layers', default=256, type=int)
    parser.add_argument('--structure', default='simple', type=str)
    parser.add_argument('--size', default=512, type=int, help='image resolution')
    parser.add_argument('--prototype_warmup', default=25, type=int)
    # Need to be changed
    parser.add_argument('--feature_lvl', default='high', type=str)
    ####################

    opts = parser.parse_args()

    return opts


def validate(**kwargs):
    if opts.way == 'kway':
        return validate_kway(**kwargs)
    else:
        return validate_1way(**kwargs)


def validate_kway(net_=None, testloader=None, epoch=0, writer=None, classes=4, save_dir='None',
                  base_class=None, novel_class=None, dataset=None, class_labels=None, sem_class=None):

    assert epoch != 0, 'The epoch should not be 0.'
    num_img_ts = len(testloader)
    net_.eval()

    agm_list = []
    npm_list = []
    label_list = []

    fgpred_list = []
    fglabel_list = []

    print("=> Validation starts")

    for ii, sample_batched in enumerate(testloader):
        query, query_mask, query_fg, support, support_mask, support_fg = \
            sample_batched['query'].cuda(), sample_batched['query_mask'].cuda(), \
            sample_batched['query_fg'].cuda(), sample_batched['support'].cuda(), \
            sample_batched['support_mask'].cuda(), sample_batched['support_fg'].cuda()

        with torch.no_grad():

            masks, losses = net_.forward((query, query_mask, query_fg, support, support_mask, support_fg),
                                         epoch=epoch)

            agm_out, npm_out, fg_out, s2_qry_gt = masks['agm_out'], masks['npm_out'], masks['s1_qry_out'], \
                                                  masks['s2_qry_gt']

            agm_predictions = torch.max(agm_out, 1)[1].cpu()
            npm_predictions = torch.max(npm_out, 1)[1].cpu()

            body_mask = torch.max(fg_out, 1)[1].cpu()

            # To save cpu memory usage in validation, we evaluate fg predictions every 10 images
            if ii % 10 == 0:
                fgpred_list.append(body_mask)
                fglabel_list.append(query_fg.squeeze(1).cpu())

            # Screening noisy pixels using our fg prediction
            if opts.testing_screening:
                body_predictions = body_mask.detach().cpu()
                npm_predictions = torch.where(body_predictions != 1, torch.zeros(1, dtype=torch.long), npm_predictions)
                agm_predictions = torch.where(body_predictions != 1, torch.zeros(1, dtype=torch.long), agm_predictions)

            agm_list.append(agm_predictions)
            npm_list.append(npm_predictions)
            label_list.append(s2_qry_gt.squeeze(1).cpu())

            # Visualize some images
            if ii <= classes * 3:
                save_dir_tmp = save_dir + '/' + str(epoch) + '_' + str(ii)
                save_img(agm_predictions, save_dir_tmp, 'agm')
                save_img(npm_predictions, save_dir_tmp, 'npm')
                save_img(body_mask, save_dir_tmp, 'fg')

        # Print stuff
        if ii % num_img_ts == num_img_ts - 1:
            print('Validation results:')
            print('[Epoch: %d, numImages: %5d]' % (epoch, ii * 1 + query.data.shape[0]))

            agm_iou = get_iou(agm_list, label_list, writer, epoch, classes, name='agm',
                              base_class=base_class, novel_class=novel_class,
                              class_labels=class_labels, sem_class=sem_class)

            npm_iou = get_iou(npm_list, label_list, writer, epoch, classes, name='npm',
                              base_class=base_class, novel_class=novel_class,
                              class_labels=class_labels, sem_class=sem_class)

            fg_iou = get_iou_from_list(fgpred_list, fglabel_list, n_cls=2)[0]

            print('npm_iou MIoU: %f\n' % npm_iou)
            print('fg MIoU: %f\n' % fg_iou)

            writer.add_scalar('data/fg_miou', fg_iou, epoch)

            return npm_iou


def validate_1way(net_=None, testloader=None, epoch=0, writer=None, classes=4, save_dir='None',
                  base_class=None, novel_class=None, dataset=None, class_labels=None, sem_class=None):

    assert epoch != 0, 'The epoch should not be 0.'

    num_img_ts = len(testloader)
    net_.eval()

    npm_dic = {}
    label_dic = {}

    print("=> Validation starts")

    for jj in range(opts.test_classes):

        npm_dic[jj] = []
        label_dic[jj] = []

    fgpred_list = []
    fglabel_list = []

    binary_pred_list = []
    binary_label_list = []

    for ii, sample_batched in enumerate(testloader):
        query, query_mask, query_fg, support, support_mask, support_fg = \
            sample_batched['query'].cuda(), sample_batched['query_mask'].cuda(), \
            sample_batched['query_fg'].cuda(), sample_batched['support'].cuda(), \
            sample_batched['support_mask'].cuda(), sample_batched['support_fg'].cuda()

        with torch.no_grad():

            cate_id = sample_batched['cate_id']

            masks, losses = net_.forward((query, query_mask, query_fg, support, support_mask, support_fg),
                                         cate_mapping=cate_id, epoch=epoch)

            agm_out, npm_out, fg_out, s2_qry_gt = masks['agm_out'], masks['npm_out'], masks['s1_qry_out'], \
                                                  masks['s2_qry_gt']

            npm_predictions = torch.max(npm_out, 1)[1].cpu()
            body_mask = torch.max(fg_out, 1)[1].cpu()
            body_predictions = body_mask.detach().cpu()
            npm_predictions = torch.where(body_predictions != 1, torch.zeros(1, dtype=torch.long), npm_predictions)

            npm_dic[int(cate_id)].append(npm_predictions.cpu())
            label_dic[int(cate_id)].append(query_mask.squeeze(1).cpu())

            if ii % 10 == 0:
                npm_predictions_ = npm_predictions.clone()
                query_mask_ = query_mask.clone()

                npm_predictions_ = torch.where(npm_predictions_ == cate_id, torch.ones(1).long(), npm_predictions_)
                query_mask_ = torch.where(query_mask_ == int(cate_id), torch.ones(1).cuda(), query_mask_)

                binary_pred_list.append(npm_predictions_.cpu())
                binary_label_list.append(query_mask_.squeeze(1).cpu())

            if ii <= classes * 3:
                save_dir_tmp = save_dir + '/' + str(epoch) + '_' + str(ii)
                save_img(npm_predictions, save_dir_tmp, 'npm')

        # Print stuff
        if ii % num_img_ts == num_img_ts - 1:

            print('Validation results:')
            print('[Epoch: %d, numImages: %5d]' % (epoch, ii * 1 + query.data.shape[0]))

            npm_iou = get_oneway_iou(npm_dic, label_dic, binary_pred_list, binary_label_list, writer, epoch,
                                     classes, name='npm', base_class=base_class, novel_class=novel_class,
                                     dataset=dataset, class_labels=class_labels, sem_class=sem_class)

            fg_iou = get_iou_from_list(fgpred_list, fglabel_list, n_cls=2)[0]

            # print('att MIoU: %f\n' % att_iou)
            print('sim MIoU: %f\n' % npm_iou)
            print('fg MIoU: %f\n' % fg_iou)

            return npm_iou


def main(opts):
    backbone = 'xception'
    nEpochs = opts.epochs
    resume_epoch = opts.resume_epoch

    # Initializing writers
    max_id = 0
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    runs = glob.glob(os.path.join(save_dir_root, 'run_cihp', 'run_*'))
    for r in runs:
        run_id = int(r.split('_')[-1])
        if run_id >= max_id:
            max_id = run_id + 1
    save_dir = os.path.join(save_dir_root, 'run_cihp', 'run_' + str(max_id))

    modelName = '{}_{}_{}_{}_{}_{}'.format(opts.structure, opts.dataset, opts.way,
                                           opts.fold, backbone, datetime.now().strftime('%b%d_%H-%M-%S'))

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('load model', opts.load_model, 1)
    writer.add_text('setting', sys.argv[0], 1)

    print("=> Structure:{}, dataset:{}, way:{}, fold:{}".format(opts.structure, opts.dataset, opts.way, opts.fold))
    print("=> Saving log to: {}".format(save_dir))

    # Model settings
    if opts.structure == 'kway_oshp':
        net_ = eopnet.EOPNet_kway(os=16, hidden_layers=opts.hidden_layers, feature_lvl=opts.feature_lvl,
                                  temperature=opts.temperature, class_num=opts.test_classes,
                                  prototype_warmup=opts.prototype_warmup)

    elif opts.structure == '1way_oshp':
        net_ = eopnet.EOPNet_1way(os=16, hidden_layers=opts.hidden_layers, feature_lvl=opts.feature_lvl,
                                  temperature=opts.temperature, class_num=opts.test_classes,
                                  prototype_warmup=opts.prototype_warmup)

    if not opts.load_model == '':
        x = torch.load(opts.load_model)
        net_.load_state_dict_new(x)
        print('=> Load model:', opts.load_model)
    else:
        print('=> No model load!')

    if not opts.resume_model == '':
        x = torch.load(opts.resume_model)
        net_.load_state_dict(x)
        print('=> Resume model:', opts.resume_model)
    else:
        print('=> Start from pretrained model!')

    # Dataset settings
    composed_transforms_tr = transforms.Compose([
        tr.RandomSized_new(opts.size),
        tr.Normalize_xception_tf(),
        tr.ToTensor_()])

    composed_transforms_ts = transforms.Compose([
        tr.Normalize_xception_tf(),
        tr.ToTensor_()])

    class_labels, sem_class = get_dataset_labels_classes(opts.dataset)
    voc_val = oshp_loader.VOCSegmentation(split='meta_test', transform=composed_transforms_ts,
                                          remain_cate=class_labels['all'], dataset=opts.dataset, way=opts.way)
    voc_train = oshp_loader.VOCSegmentation(split='meta_train', transform=composed_transforms_tr, flip=True,
                                            remain_cate=class_labels[opts.fold], dataset=opts.dataset, way=opts.way)

    trainloader = DataLoader(voc_train, batch_size=opts.batch, shuffle=True, num_workers=opts.numworker,
                             drop_last=True)
    testloader = DataLoader(voc_val, batch_size=1, shuffle=False, num_workers=opts.numworker)
    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    print("=> Len(trainloader): {}, len(testloader): {}".format(num_img_tr, num_img_ts))
    base_class = class_labels[opts.fold]
    novel_class = [i for i in class_labels['all'] if i not in base_class]
    print("=> Novel class: ", novel_class)
    print("=> Training starts")
    if gpu_id >= 0:
        net_.cuda()

    running_loss_tr = agm_loss_tr = npm_loss_tr = fg_loss_tr = contrast_loss_tr = 0.0
    global_step = 0
    optimizer = optim.SGD(net_.parameters(), lr=opts.lr, momentum=0.9, weight_decay=5e-4)

    net = torch.nn.DataParallel(net_)
    for epoch in range(resume_epoch, nEpochs):

        if opts.validate:
            # The resume epoch must be illustrated in opts
            validate(net_=net_, testloader=testloader, classes=opts.test_classes, epoch=epoch,
                     writer=writer,
                     save_dir=save_dir, base_class=base_class, novel_class=novel_class,
                     dataset=opts.dataset,
                     class_labels=class_labels, sem_class=sem_class)
            return

        start_time = timeit.default_timer()

        if opts.poly:
            if epoch % opts.step == opts.step - 1:
                lr_ = util.lr_poly(opts.lr, epoch, nEpochs, 0.9)
                optimizer = optim.SGD(net_.parameters(), lr=lr_, momentum=0.9, weight_decay=5e-4)
                writer.add_scalar('data/lr_', lr_, epoch)
                print('(poly lr policy) learning rate: ', lr_)

        beta = get_beta(epoch, nEpochs)
        print("=> beta for this epoch: ", beta[0], beta[1])

        net.train()
        for ii, sample_batched in enumerate(trainloader):

            query, query_mask, query_fg, support, support_mask, support_fg = \
                sample_batched['query'].cuda(), sample_batched['query_mask'].cuda(), \
                sample_batched['query_fg'].cuda(), sample_batched['support'].cuda(), \
                sample_batched['support_mask'].cuda(), sample_batched['support_fg'].cuda()

            # Forward-Backward of the mini-batch
            cate_id = sample_batched['cate_id']  # cate_id is only used in 1way oshp
            global_step += query.data.shape[0]

            masks, losses = net.forward((query, query_mask, query_fg, support, support_mask, support_fg),
                                        cate_mapping=cate_id, epoch=epoch)

            agm_out, npm_out, fg_out, s2_qry_gt = masks['agm_out'], masks['npm_out'], masks['s1_qry_out'], \
                                                  masks['s2_qry_gt']

            agm_loss, npm_loss, fg_loss, contrast_loss = losses['agm'], losses['npm'], losses['s1'], \
                                                         losses['contrast']

            loss = beta[0] * agm_loss + beta[1] * npm_loss + \
                   opts.fg_weight * fg_loss + opts.contrast_weight * contrast_loss

            running_loss_tr += loss.item()
            agm_loss_tr += agm_loss.item()
            npm_loss_tr += npm_loss.item()
            fg_loss_tr += fg_loss.item()
            contrast_loss_tr += contrast_loss.item()

            # Print stuff
            if ii % num_img_tr == (num_img_tr - 1):
                running_loss_tr = running_loss_tr / num_img_tr
                agm_loss_tr = agm_loss_tr / num_img_tr
                npm_loss_tr = npm_loss_tr / num_img_tr
                fg_loss_tr = fg_loss_tr / num_img_tr
                contrast_loss_tr = contrast_loss_tr / num_img_tr

                writer.add_scalars('data/running_loss', {'agm_loss': agm_loss_tr,
                                                         'npm_loss': npm_loss_tr,
                                                         'fg_loss': fg_loss_tr,
                                                         'contrast_loss': contrast_loss_tr}, epoch)

                writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii * opts.batch + query.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")

            # Backward the averaged gradient
            loss.backward()

            # Update the weights
            writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
            optimizer.step()
            optimizer.zero_grad()

            # Visualize some results
            if ii % (num_img_tr * 20) == 0:
                grid_image = make_grid(query[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('Image', grid_image, global_step)

                grid_image = make_grid(
                    util.decode_seg_map_sequence(torch.max(fg_out[:3], 1)[1].detach().cpu().numpy()), 3,
                    normalize=False,
                    range=(0, 255))
                writer.add_image('Predicted qry_fg', grid_image, global_step)

                grid_image = make_grid(
                    util.decode_seg_map_sequence(torch.max(npm_out[:3], 1)[1].detach().cpu().numpy()), 3,
                    normalize=False,
                    range=(0, 255))
                writer.add_image('Predicted npm output', grid_image, global_step)

                grid_image = make_grid(
                    util.decode_seg_map_sequence(torch.squeeze(s2_qry_gt[:3], 1).detach().cpu().numpy()), 3,
                    normalize=False, range=(0, 255))
                writer.add_image('Groundtruth label', grid_image, global_step)

                grid_image = make_grid(support[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('Support', grid_image, global_step)
                grid_image = make_grid(
                    util.decode_seg_map_sequence(torch.squeeze(support_mask[:3], 1).detach().cpu().numpy()), 3,
                    normalize=False, range=(0, 255))
                writer.add_image('Groundtruth Support', grid_image, global_step)

            print('[Epoch: {}] loss: {}, '
                  'agm loss: {}, npm loss: {}, fg loss: {}, contrastive loss: {}'.format(epoch, loss.item(),
                                                                                         agm_loss.item(),
                                                                                         npm_loss.item(),
                                                                                         fg_loss.item(),
                                                                                         contrast_loss.item()))

        # Save the model
        # One testing epoch
        if epoch % opts.testInterval == (opts.testInterval - 1):
            validate(net_=net_, testloader=testloader, classes=opts.test_classes, epoch=epoch,
                     writer=writer,
                     save_dir=save_dir, base_class=base_class, novel_class=novel_class,
                     dataset=opts.dataset,
                     class_labels=class_labels, sem_class=sem_class)

            torch.save(net_.state_dict(),
                       os.path.join(save_dir, 'models', modelName + '_' + str(epoch) + '_' + '.pth'))

        torch.save(net_.state_dict(), os.path.join(save_dir, 'models', modelName + '_current' + '.pth'))
        print("Save model at {}\n".format(
            os.path.join(save_dir, 'models', modelName + '_current.pth as our current model')))

        torch.cuda.empty_cache()


if __name__ == '__main__':
    opts = get_parser()
    main(opts)
