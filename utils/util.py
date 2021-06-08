import os

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from utils import get_iou_from_list


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


def get_cityscapes_labels():
    return np.array([
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def get_mhp_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128],  # 21
                       [96, 0, 0], [0, 96, 0], [96, 96, 0],
                       [0, 0, 96], [96, 0, 96], [0, 96, 96], [96, 96, 96],
                       [32, 0, 0], [160, 0, 0], [32, 96, 0], [160, 96, 0],
                       [32, 0, 96], [160, 0, 96], [32, 96, 96], [160, 96, 96],
                       [0, 32, 0], [96, 32, 0], [0, 160, 0], [96, 160, 0],
                       [0, 32, 96],  # 41
                       [48, 0, 0], [0, 48, 0], [48, 48, 0],
                       [0, 0, 96], [48, 0, 48], [0, 48, 48], [48, 48, 48],
                       [16, 0, 0], [80, 0, 0], [16, 48, 0], [80, 48, 0],
                       [16, 0, 48], [80, 0, 48], [16, 48, 48], [80, 48, 48],
                       [0, 16, 0], [48, 16, 0], [0, 80, 0],  # 59

                       ])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    elif dataset == 'mhp':
        n_classes = 59
        label_colours = get_mhp_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()


def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):

    n, c, h, w = logit.size()
    nt, ct, ht, wt = target.size()

    assert h == ht and w == wt
    target = target.squeeze(1)

    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(),
                                        ignore_index=ignore_index, reduction='mean')
    loss = criterion(logit, target.long())

    return loss


def binary_cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, reduction=True, ce=True):
    # n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    # print(logit.shape, target.shape)
    n, c, h, w = logit.size()
    nt, ct, ht, wt = target.size()

    assert h == ht and w == wt
    # # Handle inconsistent size between input and target
    # if h != ht and w != wt:  # upsample labels
    #     input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
    # target = target.squeeze(1)
    # print(torch.unique(target))
    # print("target.shape(): ", logit.shape, target.squeeze(1).shape, target.shape)
    if weight is None:
        criterion = nn.BCELoss(weight=weight, size_average=size_average)

    else:
        criterion = nn.BCELoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), size_average=size_average,
                               reduce=False)

    loss1 = criterion(logit, target)

    return loss1


def cross_entropy2d_dataparallel(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    # n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.DataParallel(
            nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=size_average))
    else:
        criterion = nn.DataParallel(
            nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index,
                                size_average=size_average))
    loss = criterion(logit, target.long())

    return loss.sum()


def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def lr_new_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((2 / 100 - float(iter_) / 10000) ** power)


def get_iou(pred, gt, n_classes=21):
    total_iou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou

    return total_iou


def scale_tensor(input, size=512, mode='bilinear'):
    print(input.size())
    # b,h,w = input.size()
    _, _, h, w = input.size()
    if mode == 'nearest':
        if h == 512 and w == 512:
            return input
        return F.upsample_nearest(input, size=(size, size))
    if h > 512 and w > 512:
        return F.upsample(input, size=(size, size), mode=mode, align_corners=True)
    return F.upsample(input, size=(size, size), mode=mode, align_corners=True)


def scale_tensor_list(input, ):
    output = []
    for i in range(len(input) - 1):
        output_item = []
        for j in range(len(input[i])):
            _, _, h, w = input[-1][j].size()
            output_item.append(F.upsample(input[i][j], size=(h, w), mode='bilinear', align_corners=True))
        output.append(output_item)
    output.append(input[-1])
    return output


def scale_tensor_list_0(input, base_input):
    output = []
    assert len(input) == len(base_input)
    for j in range(len(input)):
        _, _, h, w = base_input[j].size()
        after_size = F.upsample(input[j], size=(h, w), mode='bilinear', align_corners=True)
        base_input[j] = base_input[j] + after_size
    # output.append(output_item)
    # output.append(input[-1])
    return base_input


def get_dataset_labels_classes(dataset):
    CLASS_LABELS_CIHP = {
        'all': [0, 1, 2, 5, 6, 7, 9, 10, 12, 13, 14, 16, 18],
        1: [0, 1, 2, 5, 7, 10, 13, 14, 16, 18],
        2: [0, 2, 6, 9, 10, 12, 13, 14, 16, 18],
    }

    CLASS_LABELS_LIP = {
        'all': [0, 1, 2, 5, 6, 7, 9, 10, 12, 13, 14, 16, 18],
        1: [0, 1, 2, 5, 7, 10, 13, 14, 16, 18],
        2: [0, 1, 2, 6, 9, 12, 13, 14, 16, 18],
    }

    CLASS_LABELS_ATR = {
        'all': [0, 1, 2, 4, 5, 6, 7, 9, 11, 12, 14, 16],
        1: [0, 1, 2, 4, 9, 11, 12, 14, 16],
        2: [0, 2, 5, 6, 7, 9, 11, 12, 14],
    }

    SEM_CLASS = {
        'cihp': ['bg', 'hat', 'hair', 'upper-clothes', 'Dress', 'Coat', 'pants', 'torso-skin', 'skirt', 'face', 'arms', 'legs', 'shoes'],
        'atr': ['bg', 'hat', 'hair', 'upper-clothes', 'skirt', 'pants', 'dress', 'shoe', 'face', 'leg', 'arm', 'bag'],
        'lip': ['bg', 'hat', 'hair', 'upper-clothes', 'Dress', 'Coat', 'pants', 'Jumpsuits', 'skirt', 'face', 'arms', 'legs', 'shoes'],
    }

    if dataset == 'cihp':
        lbl, sem = CLASS_LABELS_CIHP, SEM_CLASS['cihp']

    elif dataset == 'lip':
        lbl, sem = CLASS_LABELS_LIP, SEM_CLASS['lip']

    elif dataset == 'atr':
        lbl, sem = CLASS_LABELS_ATR, SEM_CLASS['atr']
    else:
        raise NotImplementedError('No such dataset')

    return lbl, sem


# merging undesired classes into background
def model_mask(mask, delete_class_set):
    new_mask = mask.clone()
    for i in delete_class_set:
        new_mask = torch.where(mask == i, torch.zeros(1).cuda(), new_mask)
    # print("inside: ", torch.unique(new_mask))

    # print("outside: ", torch.unique(new_mask))
    return new_mask


def get_beta(epoch, max_epoch):
    att_beta = (max_epoch - epoch) / max_epoch
    sim_beta = (epoch / max_epoch)

    return att_beta, sim_beta


def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """

    label_set = [(0, 0, 0)
        , (128, 0, 0), (255, 0, 0), (0, 85, 0), (170, 0, 51), (255, 85, 0), (0, 0, 85), (0, 119, 221), (85, 85, 0),
                 (0, 85, 85), (85, 51, 0), (52, 86, 128), (0, 128, 0)
        , (0, 0, 255), (51, 170, 221), (0, 255, 255), (85, 255, 170), (170, 255, 85), (255, 255, 0), (255, 170, 0)]

    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
        n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_set[k]
        outputs[i] = np.array(img)
    return outputs


def save_img(predictions, path, name):
    results = predictions.cpu().numpy()
    vis_res = decode_labels(results)
    parsing_im = Image.fromarray(vis_res[0])

    parsing_im.save(path + '_' + name + '_prediction.png')


# Transparent images in paper
def save_transparent_img(predictions, img, path, name):
    """
    predictions: b * h * w
    img: b * 3 * h * w
    """

    label_colours_cv2 = [(0, 0, 0)
        , (0, 0, 128), (0, 0, 255), (0, 85, 0), (51, 0, 170), (0, 85, 255), (85, 0, 0), (221, 119, 0), (0, 85, 85),
                     (85, 85, 0), (0, 51, 85), (128, 86, 52), (0, 128, 0)
        , (255, 0, 0), (221, 170, 51), (255, 255, 0), (170, 255, 85), (85, 255, 170), (0, 255, 255), (0, 170, 255)]

    results = predictions.squeeze(1).long().cpu().numpy()

    vis_res = decode_labels(results, label_set=label_colours_cv2)
    parsing_im = vis_res[0]
    parsing_im = np.transpose(parsing_im, (2, 0, 1)).astype('float32')

    img = (img[0] + 1) * 256 / 2

    cv2.addWeighted(parsing_im, 1.4, img, 0.6, 0, img)
    img = np.transpose(img, (1, 2, 0))

    name = path + '_' + name + '_prediction.jpg'
    cv2.imwrite(name, img)




if __name__ == '__main__':
    print(lr_poly(0.007, iter_=99, max_iter=150))
