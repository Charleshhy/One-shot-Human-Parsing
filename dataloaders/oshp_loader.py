from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
from .mypath import Path
import random
from PIL import ImageFile
import numpy as np
import pickle
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True

# We can use both 1way and kway in this loader

def generate_fgmask(target, classes):
    new_target = target.copy()

    for i in range(1, classes):
        new_target = np.where(target == i, 1, new_target)

    # print('fgmask: ', np.unique(new_target))
    return new_target


class VOCSegmentation(Dataset):
    """
    ATR-OS dataset
    """

    def __init__(self,
                 dataset='',
                 split='meta_train',
                 transform=None,
                 flip=False,
                 dtrain_dtest_split=7500,
                 remain_cate=None,
                 simp_classes=False,
                 way='kway',
                 ):
        """
        :param dataset: dataset name, 'ATR' or 'CIHP' or 'LIP'
        :param split: meta_train/meta_test
        :param transform: transform to apply
        :param dtrain_dtest_split: generating customized query-support split
        :param remain_cate: get rid of the categories that are not in the current fold
        :param sem_cls: 19 semantic classes for CIHP (background excluded)
        :param way: 'kway' that parses k classes in one episode or '1way' that parses 1 class in one episode
        """
        super(VOCSegmentation).__init__()
        self._flip_flag = flip

        self.dataset = dataset.split('_')[0]
        self._base_dir = Path.db_root_dir(self.dataset)
        self.way = way
        if self.dataset == 'cihp' or self.dataset == 'lip':
            self.sem_cls = sem_cls = 19
        elif self.dataset == 'atr':
            self.sem_cls = sem_cls = 17
        else:
            raise NotImplementedError

        file_name = '_' + self.dataset + '_supports.pkl'
        self.classes = sem_cls + 1
        self.simp_classes = simp_classes
        self._image_dir = os.path.join(self._base_dir, 'trainval_images')
        self._cat_dir = os.path.join(self._base_dir, 'trainval_classes')
        self._flip_dir = os.path.join(self._base_dir,'Category_rev_ids')
        self.remain_cate = remain_cate
        self.transform = transform

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self._sup_cate_dir = os.path.join(self._base_dir + 'support')
        _splits_dir = os.path.join(self._base_dir, 'list')
        self.im_ids = []
        self.images = []
        self.categories = []
        self.flip_categories = []
        self.supports = {}
        for cls in range(1 + sem_cls * 2):
            self.supports[cls] = []

        # Load qry-support pairs for evaluation purpose
        for splt in self.split:
            with open(os.path.join(_splits_dir, splt + '_id.txt'), "r") as f:
                lines = f.read().splitlines()

            self.readlines(lines, str(split), dtrain_dtest_split, file_name)

        # for key in self.supports:
        #     print(str(split), key, len(self.supports[key]))

        assert (len(self.images) == len(self.categories))
        assert len(self.flip_categories) == len(self.categories)

        self.img_num_lis = [0]
        self.pairs = []

        if self.split[0] == 'meta_train':
            if self.way == 'kway':
                for i in range(dtrain_dtest_split):
                    r = np.random.choice(list(range(dtrain_dtest_split, len(self.images))))
                    self.pairs.append([i, r, 0])

            else:
                total_img = 0
                self.img_num_lis = [0]

                # Fixed number of training images
                num_each_cate = 1500

                for i in range(1, len(remain_cate)):
                    for j in range(num_each_cate):

                        # This shouldn't happen
                        if j >= len(self.supports[remain_cate[i]]):
                            break

                        else:
                            img = self.supports[remain_cate[i]][j]

                        r = np.random.choice(list(range(0, len(self.supports[remain_cate[i] + sem_cls]))))
                        sup = self.supports[remain_cate[i] + sem_cls][r]

                        self.pairs.append([img, sup, remain_cate[i]])
                        total_img += 1
                    self.img_num_lis.append(total_img)

        elif self.split[0] == 'meta_test':

            if not os.path.isfile(os.path.join(self._sup_cate_dir, self.dataset + '_meta_test_list.txt')):
                print('=> Start forming query list and support list for meta_test')
                qry_list, sup_list, cate_id = self.form_query_support_list()
            else:
                print('=> Start loading query list and support list for meta_test')
                with open(os.path.join(self._sup_cate_dir, self.dataset + '_meta_test_list.txt'), "rb") as fp:
                    qry_list, sup_list, cate_id = pickle.load(fp)

            print('=> Number of support list and query list: ', len(qry_list), len(sup_list))
            assert len(qry_list) == len(sup_list) == len(cate_id)

            for i in range(len(qry_list)):
                self.pairs.append([qry_list[i], sup_list[i], cate_id[i]])

        else:
            raise NotImplementedError("No such split")

        print('\n=> Number of images in {}: {:d}'.format(split, len(self.images)))
        print('=> Number of pairs in {}: {:d}'.format(split, len(self.pairs)))

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        pair = self.pairs[index]

        _img = Image.open(self.images[pair[0]]).convert('RGB')  # return is RGB pic
        _sup = Image.open(self.images[pair[1]]).convert('RGB')

        if self._flip_flag:
            if random.random() < 0.5:
                _target = Image.open(self.flip_categories[pair[0]])
                _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                _target = Image.open(self.categories[pair[0]])
        else:
            _target = Image.open(self.categories[pair[0]])

        if self._flip_flag:
            if random.random() < 0.5:
                _sup_target = Image.open(self.flip_categories[pair[1]])
                _sup = _sup.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                _sup_target = Image.open(self.categories[pair[1]])
        else:
            _sup_target = Image.open(self.categories[pair[1]])

        _sup_target_arr = np.array(_sup_target)
        _sup_arr = np.array(_sup)

        _target_arr = np.array(_target)
        _img_arr = np.array(_img)

        _img_fg_mask_arr = generate_fgmask(_target_arr, self.classes)
        _sup_fg_mask_arr = generate_fgmask(_sup_target_arr, self.classes)

        if self.way == '1way':
            _target_arr, _sup_target_arr = self.transform_label_1way(_target_arr, _sup_target_arr, pair[2])

        elif self.way == 'kway':
            _target_arr, _sup_target_arr = self.transform_label(_target_arr, _sup_target_arr, self.remain_cate)

        else:
            raise NotImplementedError

        _sup_seen_arr = generate_fgmask(_sup_target_arr, self.classes)

        return Image.fromarray(_img_arr), Image.fromarray(_target_arr), Image.fromarray(_sup_arr), \
               Image.fromarray(_sup_target_arr), Image.fromarray(_img_fg_mask_arr), Image.fromarray(_sup_fg_mask_arr), \
               pair[2]  # Seen classes are the (foreground - novel - prev_deleted class)

    def transform_label(self, target, sup_target, remain_cate):

        target, sup_target = self.mirror_cls(target), self.mirror_cls(sup_target)
        remain_lis_new = [i for i in remain_cate]
        get_rid_lis = [i for i in list(range(self.classes)) if i not in remain_lis_new]

        for i in range(len(get_rid_lis)):
            target = np.where(target == get_rid_lis[i], 0, target)
            sup_target = np.where(sup_target == get_rid_lis[i], 0, sup_target)

        sem_target = target.copy()
        sem_sup_target = sup_target.copy()

        return sem_target, sem_sup_target

    def transform_label_1way(self, target, sup_target, category):

        target, sup_target = self.mirror_cls(target), self.mirror_cls(sup_target)
        get_rid_lis = [i for i in list(range(self.classes)) if i != category]

        for i in range(len(get_rid_lis)):
            target = np.where(target == get_rid_lis[i], 0, target)
            sup_target = np.where(sup_target == get_rid_lis[i], 0, sup_target)

        sem_target = target.copy()
        sem_sup_target = sup_target.copy()

        # Map the target category into 1
        sem_target = np.where(target == category, 1, sem_target)
        sem_sup_target = np.where(sup_target == category, 1, sem_sup_target)

        return sem_target, sem_sup_target

    def read_mask(self, mask_path):

        assert os.path.exists(mask_path), "%s does not exist"%(mask_path)
        mask = cv2.imread(mask_path)
        mask = self.mirror_cls(mask)
        return mask

    def get_labels(self, mask_path):

        mask = self.read_mask(mask_path)
        mask = self.mirror_cls(mask)
        labels = np.unique(mask)
        labels = [label for label in labels if label != 255 and label != 0]

        return labels

    def write_labels_groups(self, path):

        with open(path, 'wb') as f:
            pickle.dump(self.supports, f)

    def load_label_groups(self, path):

        with open(path, 'rb') as f:
            self.supports = pickle.load(f)

    def readlines(self, lines, split, Dtrain_Dtest_split, file_name):

        path = os.path.join(self._sup_cate_dir, split + file_name)
        if not os.path.isfile(path):
            for ii, line in enumerate(lines):

                _image = os.path.join(self._image_dir, line + '.jpg')
                _cat = os.path.join(self._cat_dir, line + '.png')
                _flip = os.path.join(self._flip_dir, line + '.png')

                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                assert os.path.isfile(_flip)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.flip_categories.append(_flip)

                labels = self.get_labels(_cat)

                # print(labels)
                for i in labels:

                    if ii <= Dtrain_Dtest_split:
                        self.supports[i].append(ii)
                    else:

                        # print(labels, self.sem_cls, i)
                        self.supports[i + self.sem_cls].append(ii)

            self.write_labels_groups(path)

        else:
            for ii, line in enumerate(lines):

                _image = os.path.join(self._image_dir, line+'.jpg' )
                _cat = os.path.join(self._cat_dir, line +'.png')
                _flip = os.path.join(self._flip_dir,line + '.png')

                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                assert os.path.isfile(_flip)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.flip_categories.append(_flip)

            self.load_label_groups(path)

    def get_img_num_lis(self):
        return self.img_num_lis

    def form_query_support_list(self):

        qry_list, sup_list, cate_id = [], [], []

        # Hard coded class sequence, to make sure the less frequent classes can be selected first
        if self.dataset == 'lip':
            sequence_list = [10, 6, 12, 1, 2, 5, 7, 9, 13, 14, 16, 18]
        elif self.dataset == 'cihp':
            sequence_list = [12, 6, 1, 2, 5, 7, 9, 10, 13, 14, 16, 18]
        else:
            sequence_list = [1, 7, 5, 16, 9, 4, 2, 6, 11, 12, 14]

        for count, key in enumerate(sequence_list):

            qry_num = 10
            sup_num = 15

            i = 0
            j = 0

            while j < qry_num:
                if self.supports[key][i] not in qry_list:
                    qry_list += [self.supports[key][i]] * sup_num
                    j += 1
                    cate_id += [key] * sup_num

                i += 1

            i = 0
            j = 0

            while j < qry_num * sup_num:

                if self.supports[key + self.sem_cls][i] not in sup_list:
                    sup_list += [self.supports[key + self.sem_cls][i]]
                    j += 1

                i += 1

        with open(os.path.join(self._sup_cate_dir, self.dataset + '_meta_test_list.txt'), "wb") as fp:
            pickle.dump([qry_list, sup_list, cate_id], fp)

        return qry_list, sup_list, cate_id

    def mirror_cls(self, mask):
        # As described by the paper, the symmetric classes are merged into one class
        # E.g. left-leg and right-leg are merged into legs.

        if self.dataset == 'cihp' or self.dataset == 'lip':
            mask = np.where(mask == 15, 14, mask)
            mask = np.where(mask == 17, 16, mask)
            mask = np.where(mask == 19, 18, mask)

            mask = np.where(mask == 3, 0, mask)
            mask = np.where(mask == 4, 0, mask)
            mask = np.where(mask == 8, 0, mask)
            mask = np.where(mask == 11, 0, mask)
        else:
            mask = np.where(mask == 15, 14, mask)
            mask = np.where(mask == 10, 9, mask)
            mask = np.where(mask == 13, 12, mask)

            mask = np.where(mask == 3, 0, mask)
            mask = np.where(mask == 8, 0, mask)
            mask = np.where(mask == 17, 0, mask)

        # The rare classes are merged into background

        return mask

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        _img, _target, _sup, _sup_target, _img_fg, _sup_fg, _cate_id = self._make_img_gt_point_pair(index)

        sample = {'query': _img, 'query_mask': _target, 'support': _sup, 'support_mask': _sup_target,
                  'query_fg': _img_fg, 'support_fg': _sup_fg}

        if self.transform is not None:
            sample = self.transform(sample)

        id1, id2 = self.im_ids[self.pairs[index][0]], self.im_ids[self.pairs[index][1]]

        sample['img_id'] = id1
        sample['img_id2'] = id2
        sample['cate_id'] = _cate_id

        return sample

    def __str__(self):
        return 'CIHP(split=' + str(self.split) + ')'