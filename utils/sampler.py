import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math
from torchvision import transforms
from PIL import Image

__all__ = ['cusSampler','Sampler_uni']

'''common N-pairs sampler'''
def index_dataset(dataset):
    '''
    get the index according to the dataset type(e.g. pascal or atr or cihp)
    :param dataset:
    :return:
    '''
    return_dict = {}
    for i in range(len(dataset)):
        tmp_lbl = dataset.datasets_lbl[i]
        if tmp_lbl in return_dict:
            return_dict[tmp_lbl].append(i)
        else :
            return_dict[tmp_lbl] = [i]
    return return_dict

def sample_from_class(dataset,class_id):
    return dataset[class_id][random.randrange(len(dataset[class_id]))]

def sampler_npair_K(batch_size,dataset,K=2,label_random_list = [0,0,1,1,2,2,2]):
    images_by_class = index_dataset(dataset)
    for batch_idx in range(int(math.ceil(len(dataset) * 1.0 / batch_size))):
        example_indices = [sample_from_class(images_by_class, class_label_ind) for _ in range(batch_size)
                           for class_label_ind in [label_random_list[random.randrange(len(label_random_list))]]
                           ]
        yield example_indices[:batch_size]

def sampler_(images_by_class,batch_size,dataset,K=2,label_random_list = [0,0,1,1,]):
    # images_by_class = index_dataset(dataset)
    a = label_random_list[random.randrange(len(label_random_list))]
    # print(a)
    example_indices = [sample_from_class(images_by_class, a) for _ in range(batch_size)
                           for class_label_ind in [a]
                           ]
    return example_indices[:batch_size]

class cusSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, dataset, batchsize, label_random_list=[0,1,1,1,2,2,2]):
        self.images_by_class = index_dataset(dataset)
        self.batch_size = batchsize
        self.dataset = dataset
        self.label_random_list = label_random_list
        self.len = int(math.ceil(len(dataset) * 1.0 / batchsize))

    def __iter__(self):
        # return [sample_from_class(self.images_by_class, class_label_ind) for _ in range(self.batchsize)
        #                    for class_label_ind in [self.label_random_list[random.randrange(len(self.label_random_list))]]
        #                    ]
        # print(sampler_(self.images_by_class,self.batch_size,self.dataset))
        return iter(sampler_(self.images_by_class,self.batch_size,self.dataset,self.label_random_list))

    def __len__(self):
        return self.len

# def shuffle_cus(d1=20,d2=10,d3=5,batch=2):
#     return_list = []
#     total_num = d1 + d2 + d3
#     list1 = list(range(d1))
#     batch1 = d1//batch
#     list2 = list(range(d1,d1+d2))
#     batch2 = d2//batch
#     list3 = list(range(d1+d2,d1+d2+d3))
#     batch3 = d3// batch
#     random.shuffle(list1)
#     random.shuffle(list2)
#     random.shuffle(list3)
#     random_list = list(range(batch1+batch2+batch3))
#     random.shuffle(random_list)
#     for random_batch_index in random_list:
#         if random_batch_index < batch1:
#             random_batch_index1 = random_batch_index
#             return_list += list1[random_batch_index1*batch : (random_batch_index1+1)*batch]
#         elif random_batch_index < batch1 + batch2:
#             random_batch_index1 = random_batch_index - batch1
#             return_list += list2[random_batch_index1*batch : (random_batch_index1+1)*batch]
#         else:
#             random_batch_index1 = random_batch_index - batch1 - batch2
#             return_list += list3[random_batch_index1*batch : (random_batch_index1+1)*batch]
#     return return_list


def shuffle_cus(cate_size=500, cate_num=13, batch=2):
    return_list = []
    total_num = cate_size * cate_num

    lists = {}
    batches = {}
    batch_num = 0

    for i in range(cate_num):
        lists[i] = list(range(i * cate_size, (i+1) * cate_size))
        batches[i] = cate_size//batch
        random.shuffle(lists[i])

        batch_num += batches[i]

    # print(lists[1])
    # print('batch_num: ', batch_num)
    # list1 = list(range(d1))
    # batch1 = d1//batch
    # list2 = list(range(d1,d1+d2))
    # batch2 = d2//batch
    # list3 = list(range(d1+d2,d1+d2+d3))
    # batch3 = d3// batch
    # random.shuffle(list1)
    # random.shuffle(list2)
    # random.shuffle(list3)
    random_list = list(range(batch_num))
    random.shuffle(random_list)

    # print("batch_num: ", batch_num)
    for random_batch_index in random_list:

        # print(random_batch_index)

        cate = random_batch_index // batches[0]

        random_batch_index = random_batch_index % batches[0]

        # random_batch_index = total_index - cate * batch_num

        # print(cate, random_batch_index)
        return_list += lists[cate][random_batch_index*batch : (random_batch_index+1)*batch]

        # print("index: ", cate, random_batch_index*batch, (random_batch_index+1)*batch - 1)
        # print(lists[cate][random_batch_index*batch], lists[cate][(random_batch_index+1)*batch - 1])

        # The -1 here means the index of (random_batch_index+1)*batch] - 1) is the last one in each batch
        assert abs(lists[cate][random_batch_index*batch] - lists[cate][(random_batch_index+1)*batch - 1]) <= cate_size - 1
        # print(cate)

        # if cate == 1:
        #     print('lists', lists[cate][random_batch_index*batch : (random_batch_index+1)*batch])

    # print('return_list: ', return_list)
    # print("len(return_list): ", len(return_list))
    return return_list


def shuffle_cus_imb(img_num_lis, cate_num=13, batch=2):
    return_list = []

    lists = {}

    # The first batch idx is 0
    batches = [0]
    batch_num = 0

    for i in range(cate_num):
        lists[i] = list(range(img_num_lis[i], img_num_lis[i + 1]))

        cate_batch = (img_num_lis[i + 1] - img_num_lis[i])//batch
        random.shuffle(lists[i])
        batch_num += cate_batch
        batches.append(batch_num)

    random_list = list(range(batch_num))
    random.shuffle(random_list)

    # print("batch_num: ", batch_num)
    for random_batch_index in random_list:

        # print(random_batch_index)
        for batch_idx in range(cate_num):
            if batches[batch_idx] <= random_batch_index < batches[batch_idx + 1]:
                break

            # If batch_idx == range(cate_num), break should take effect
            assert batch_idx != range(cate_num)

        cate = batch_idx

        random_batch_index = random_batch_index - batches[batch_idx]

        # random_batch_index = total_index - cate * batch_num

        # print(cate, random_batch_index)
        return_list += lists[cate][random_batch_index*batch : (random_batch_index+1)*batch]

    # print('return_list: ', return_list)
    # print("len(return_list): ", len(return_list))
    return return_list


def shuffle_cus_balance(d1=20,d2=10,d3=5,batch=2,balance_index=1):
    return_list = []
    total_num = d1 + d2 + d3
    list1 = list(range(d1))
    # batch1 = d1//batch
    list2 = list(range(d1,d1+d2))
    # batch2 = d2//batch
    list3 = list(range(d1+d2,d1+d2+d3))
    # batch3 = d3// batch
    random.shuffle(list1)
    random.shuffle(list2)
    random.shuffle(list3)
    total_list = [list1,list2,list3]
    target_list = total_list[balance_index]
    for index,list_item in enumerate(total_list):
        if index == balance_index:
            continue
        if len(list_item) > len(target_list):
            list_item = list_item[:len(target_list)]
            total_list[index] = list_item
    list1 = total_list[0]
    list2 = total_list[1]
    list3 = total_list[2]
    # list1 = list(range(d1))
    d1 = len(list1)
    batch1 = d1 // batch
    # list2 = list(range(d1, d1 + d2))
    d2 = len(list2)
    batch2 = d2 // batch
    # list3 = list(range(d1 + d2, d1 + d2 + d3))
    d3 = len(list3)
    batch3 = d3 // batch

    random_list = list(range(batch1+batch2+batch3))
    random.shuffle(random_list)
    for random_batch_index in random_list:
        if random_batch_index < batch1:
            random_batch_index1 = random_batch_index
            return_list += list1[random_batch_index1*batch : (random_batch_index1+1)*batch]
        elif random_batch_index < batch1 + batch2:
            random_batch_index1 = random_batch_index - batch1
            return_list += list2[random_batch_index1*batch : (random_batch_index1+1)*batch]
        else:
            random_batch_index1 = random_batch_index - batch1 - batch2
            return_list += list3[random_batch_index1*batch : (random_batch_index1+1)*batch]
    return return_list

class Sampler_uni(torch.utils.data.sampler.Sampler):
    def __init__(self, cate_size=500, cate_num=13, batchsize=2):
        self.cate_size = cate_size
        self.cate_num = cate_num
        self.batchsize = batchsize

    def __iter__(self):

        rlist = shuffle_cus(cate_size=self.cate_size, cate_num=self.cate_num, batch=self.batchsize)

        return iter(rlist)

    def __len__(self):

        return self.cate_size * self.cate_num


class Sampler_imbalance(torch.utils.data.sampler.Sampler):
    def __init__(self, img_num_lis, cate_num=13, batchsize=2):
        self.img_num_lis = img_num_lis
        self.cate_num = cate_num
        self.batchsize = batchsize

        self.batch_num = 0

        for i in range(cate_num):
            cate_batch = (img_num_lis[i + 1] - img_num_lis[i]) // batchsize
            self.batch_num += cate_batch

    def __iter__(self):

        rlist = shuffle_cus_imb(self.img_num_lis, cate_num=self.cate_num, batch=self.batchsize)

        return iter(rlist)

    def __len__(self):

        return self.batch_num
