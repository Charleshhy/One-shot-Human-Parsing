import torch
import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps
from torchvision import transforms


class RandomCrop_new(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        query, query_mask, query_fg = sample['query'], sample['query_mask'], sample['query_fg']
        support, support_mask, support_fg = sample['support'], sample['support_mask'], sample['support_fg']

        if self.padding > 0:
            query = ImageOps.expand(query, border=self.padding, fill=0)
            query_mask = ImageOps.expand(query_mask, border=self.padding, fill=0)
            query_fg = ImageOps.expand(query_fg, border=self.padding, fill=0)

            support = ImageOps.expand(support, border=self.padding, fill=0)
            support_mask = ImageOps.expand(support_mask, border=self.padding, fill=0)
            support_fg = ImageOps.expand(support_fg, border=self.padding, fill=0)

        assert query.size == query_mask.size
        assert support.size == support_mask.size

        w, h = query.size
        th, tw = self.size # target size

        sw, sh = support.size
        if w == tw and h == th and sw==tw and sh==tw:
            return {'query': query,
                    'query_mask': query_mask,
                    'query_fg': query_fg,
                    'support': support,
                    'support_mask': support_mask,
                    'support_fg': support_fg}

        new_query = Image.new('RGB',(tw,th),'black')  # size is w x h; and 'white' is 255
        new_query_mask = Image.new('L',(tw,th),'white')  # same above
        new_query_fg = Image.new('L',(tw,th),'white')  # same above

        new_support = Image.new('RGB',(tw,th),'black')  # size is w x h; and 'white' is 255
        new_support_mask = Image.new('L',(tw,th),'white')  # same above
        new_support_fg = Image.new('L',(tw,th),'white')  # same above

        # if w > tw or h > th
        x1 = y1 = 0
        if w > tw:
            x1 = random.randint(0,w - tw)
        if h > th:
            y1 = random.randint(0,h - th)
        # crop
        query = query.crop((x1, y1, x1 + tw, y1 + th))
        query_mask = query_mask.crop((x1, y1, x1 + tw, y1 + th))
        query_fg = query_fg.crop((x1,y1, x1 + tw, y1 + th))

        new_query.paste(query,(0,0))
        new_query_mask.paste(query_mask,(0,0))
        new_query_fg.paste(query_fg, (0, 0))

        x2 = y2 = 0
        if sw > tw:
            x2 = random.randint(0,sw - tw)
        if sh > th:
            y2 = random.randint(0,sh - th)

        support = support.crop((x2,y2, x2 + tw, y2 + th))
        support_mask = support_mask.crop((x2,y2, x2 + tw, y2 + th))
        support_fg = support_fg.crop((x2,y2, x2 + tw, y2 + th))

        new_support.paste(support,(0,0))
        new_support_mask.paste(support_mask,(0,0))
        new_support_fg.paste(support_fg, (0, 0))

        return {'query': new_query,
                    'query_mask': new_query_mask,
                    'query_fg': new_query_fg,
                    'support': new_support,
                    'support_mask': new_support_mask,
                    'support_fg': new_support_fg}


class Paste(object):
    def __init__(self, size,):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size # target size
        assert (w <=tw) and (h <= th)
        if w == tw and h == th:
            return {'image': img,
                    'label': mask}

        new_img = Image.new('RGB',(tw,th),'black')  # size is w x h; and 'white' is 255
        new_mask = Image.new('L',(tw,th),'white')  # same above

        new_img.paste(img,(0,0))
        new_mask.paste(mask,(0,0))

        return {'image': new_img,
                'label': new_mask}

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class HorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        support = sample['support'].transpose(Image.FLIP_LEFT_RIGHT)
        support_mask = sample['support_mask'].transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask,
                'support': support,
                'support_mask:': support_mask}

class HorizontalFlip_only_img(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class RandomHorizontalFlip_cihp(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # mask = Image.open()

        return {'image': img,
                'label': mask}

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}

class Normalize_255(object):
    """Normalize a tensor image with mean and standard deviation. tf use 255.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(123.15, 115.90, 103.06), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        # img = 255.0
        img -= self.mean
        img /= self.std
        img = img
        img = img[[0,3,2,1],...]
        return {'image': img,
                'label': mask}


class Normalize_xception_tf(object):

    def __call__(self, sample):
        query = np.array(sample['query']).astype(np.float32)
        query_mask = np.array(sample['query_mask']).astype(np.float32)
        query_fg = np.array(sample['query_fg']).astype(np.float32)

        support = np.array(sample['support']).astype(np.float32)
        support_mask = np.array(sample['support_mask']).astype(np.float32)
        support_fg = np.array(sample['support_fg']).astype(np.float32)

        query = (query*2.0)/255.0 - 1
        support = (support * 2.0) / 255.0 - 1

        return {'query': query,
                    'query_mask': query_mask,
                    'query_fg': query_fg,
                    'support': support,
                    'support_mask': support_mask,
                    'support_fg': support_fg}


class ToTensor_(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.rgb2bgr = transforms.Lambda(lambda x:x[[2,1,0],...])

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        query = np.array(sample['query']).astype(np.float32).transpose((2, 0, 1))
        query_mask = np.expand_dims(np.array(sample['query_mask']).astype(np.float32), -1).transpose((2, 0, 1))
        query_fg = np.expand_dims(np.array(sample['query_fg']).astype(np.float32), -1).transpose((2, 0, 1))

        support = np.array(sample['support']).astype(np.float32).transpose((2, 0, 1))
        support_mask = np.expand_dims(np.array(sample['support_mask']).astype(np.float32), -1).transpose((2, 0, 1))
        support_fg = np.expand_dims(np.array(sample['support_fg']).astype(np.float32), -1).transpose((2, 0, 1))

        query = torch.from_numpy(query).float()
        query = self.rgb2bgr(query)
        query_fg = torch.from_numpy(query_fg).float()
        query_mask = torch.from_numpy(query_mask).float()

        support = torch.from_numpy(support).float()
        support = self.rgb2bgr(support)
        support_fg = torch.from_numpy(support_fg).float()
        support_mask = torch.from_numpy(support_mask).float()

        return {'query': query,
                    'query_mask': query_mask,
                    'query_fg': query_fg,
                    'support': support,
                    'support_mask': support_mask,
                    'support_fg': support_fg}

class ToTensor_original_(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.rgb2bgr = transforms.Lambda(lambda x:x[[2,1,0],...])

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample['image_original']).astype(np.float32).transpose((2, 0, 1))

        support = np.array(sample['image_original']).astype(np.float32).transpose((2, 0, 1))


        img = torch.from_numpy(img).float()
        img = self.rgb2bgr(img)

        support = torch.from_numpy(support).float()
        support = self.rgb2bgr(support)

        sample['image_original'] = img
        sample['support_original'] = support

        return sample

class ToTensor_only_img(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.rgb2bgr = transforms.Lambda(lambda x:x[[2,1,0],...])

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        # mask = np.expand_dims(np.array(sample['label']).astype(np.float32), -1).transpose((2, 0, 1))
        # mask[mask == 255] = 0

        img = torch.from_numpy(img).float()
        img = self.rgb2bgr(img)
        # mask = torch.from_numpy(mask).float()


        return {'image': img,
                'label': sample['label']}

class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}

class FixedResizeOriginal(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image_original']
        sup = sample['support_original']

        # assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        sup = sup.resize(self.size, Image.BILINEAR)

        sample['image_original'] = img
        sample['support_original'] = sup

        return sample

class Keep_origin_size_Resize(object):
    def __init__(self, max_size, scale=1.0):
        self.size = tuple(reversed(max_size))  # size: (h, w)
        self.scale = scale
        self.paste = Paste(int(max_size[0]*scale))

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size
        h, w = self.size
        h = int(h*self.scale)
        w = int(w*self.scale)
        img = img.resize((h, w), Image.BILINEAR)
        mask = mask.resize((h, w), Image.NEAREST)

        return self.paste({'image': img,
                'label': mask})

class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        human = sample['human']

        img_parent = sample['img_parent']
        support_mask = sample['support_mask']

        assert img.size == mask.size
        # w, h = img.size

        # if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
        #     return {'image': img,
        #             'label': mask}
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        img_parent = img_parent.resize((ow, oh), Image.NEAREST)
        human = human.resize((ow, oh), Image.NEAREST)

        support = sample['support']
        sup_parent = sample['sup_parent']
        sup_seen = sample['sup_seen']

        oh, ow = self.size
        support = support.resize((ow, oh), Image.BILINEAR)
        sup_mask = support_mask.resize((ow, oh), Image.NEAREST)
        sup_parent = sup_parent.resize((ow, oh), Image.NEAREST)
        sup_seen = sup_seen.resize((ow, oh), Image.NEAREST)

        # assert support.size == sup_mask.size

        return {'image': img,
                'label': mask,
                'support': support,
                'support_mask': sup_mask,
                'img_parent': img_parent,
                'sup_parent': sup_parent,
                'sup_seen': sup_seen,
                'human': human}

class Scale_(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size
        ow = int(w*self.scale)
        oh = int(h*self.scale)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)


        return {'image': img,
                'label': mask}

class Scale_only_img(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # assert img.size == mask.size
        w, h = img.size
        ow = int(w*self.scale)
        oh = int(h*self.scale)
        img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask}

class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)

                return {'image': img,
                        'label': mask}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}

class RandomSized_new(object):
    '''what we use is this class to aug'''
    def __init__(self, size, scale1=0.5, scale2=2):
        self.size = size
        self.crop = RandomCrop_new(self.size)
        self.small_scale = scale1
        self.big_scale = scale2

    def __call__(self, sample):
        query = sample['query']
        query_mask = sample['query_mask']
        query_fg = sample['query_fg']
        assert query.size == query_mask.size

        support = sample['support']
        support_mask = sample['support_mask']
        support_fg = sample['support_fg']
        assert support.size == support_mask.size

        w = int(random.uniform(self.small_scale, self.big_scale) * query.size[0])
        h = int(random.uniform(self.small_scale, self.big_scale) * query.size[1])

        query, query_mask, query_fg = query.resize((w, h), Image.BILINEAR), query_mask.resize((w, h), Image.NEAREST), \
                            query_fg.resize((w, h), Image.NEAREST)

        w = int(random.uniform(self.small_scale, self.big_scale) * support.size[0])
        h = int(random.uniform(self.small_scale, self.big_scale) * support.size[1])

        support, support_mask, support_fg = support.resize((w, h), Image.BILINEAR), \
                                            support_mask.resize((w, h), Image.NEAREST), support_fg.resize((w, h), Image.NEAREST)


        sample = {'query': query, 'query_mask': query_mask, 'support': support,
                  'support_mask': support_mask, 'query_fg': query_fg,
                  'support_fg': support_fg,
                  }
        # finish resize
        return self.crop(sample)
# class Random

class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return {'image': img, 'label': mask}


class RandomAggregation(object):
    def __init__(self, rate, dataset):
        self.rate = rate
        self.dataset = dataset

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        support = sample['support']
        sup_mask = sample['support_mask']

        if self.dataset == 'cihp':
            lis = [[0], [1, 2, 4, 13], [5, 6, 7, 10, 11, 12], [3, 14, 15], [8, 9, 16, 17, 18, 19]]
        elif self.dataset == 'pascal':
            lis = [[0], [1], [2], [3, 4], [5, 6]]
        elif self.dataset == 'atr':
            lis = [[0], [1, 2, 3, 11], [4, 5, 7, 8, 16, 17], [14, 15], [6, 9, 10, 12, 13]]
        else:
            raise NotImplementedError

        randlis = [random.randint(0, 1) for b in range(1, 5)]
        print(randlis)

        for i in range(5):
            if randlis[i]:
                for j in lis[i]:
                    mask = torch.where(mask == j, torch.tensor([i + 20]).cuda(), mask)
                    sup_mask = torch.where(sup_mask == j, torch.tensor([i + 20]).cuda(), sup_mask)

        return {'image': img,
                'label': mask,
                'support': support,
                'support_mask:': sup_mask}
