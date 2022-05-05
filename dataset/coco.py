import os
import numpy as np
import random
import json
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import cv2


class CocoPanopticDataset(Dataset):
    def __init__(self,
                 imgdir: str,
                 anndir: str,
                 annfile: str,
                 transform=None):
        with open(annfile) as f:
            self.data = json.load(f)['annotations']
            self.data = list(filter(lambda data: any(info['category_id'] == 1 for info in data['segments_info']), self.data))
        self.imgdir = imgdir
        self.anndir = anndir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        img = self._load_img(data)
        seg = self._load_seg(data)
        
        if self.transform is not None:
            img, msk, seg = self.transform(img, seg)
            
        return img, msk, seg

    def _load_img(self, data):
        with Image.open(os.path.join(self.imgdir, data['file_name'].replace('.png', '.jpg'))) as img:
            return img.convert('RGB')
    
    def _load_seg(self, data):
        with Image.open(os.path.join(self.anndir, data['file_name'])) as ann:
            ann.load()
            
        ann = np.array(ann, copy=False).astype(np.int32)
        ann = ann[:, :, 0] + 256 * ann[:, :, 1] + 256 * 256 * ann[:, :, 2]
        seg = np.zeros(ann.shape, np.uint8)
        
        for segments_info in data['segments_info']:
            if segments_info['category_id'] in [1, 27, 32]: # person, backpack, tie
                seg[ann == segments_info['id']] = 255
        
        return Image.fromarray(seg)
    

class CocoPanopticTrainAugmentation:
    def __init__(self, size):
        self.size = size
        self.jitter = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
        self.kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,100)]
    
    def __call__(self, img, seg):
        # Affine
        params = transforms.RandomAffine.get_params(degrees=(-20, 20), translate=(0.1, 0.1),
                                                    scale_ranges=(1, 1), shears=(-10, 10), img_size=img.size)
        img = F.affine(img, *params, interpolation=F.InterpolationMode.BILINEAR)
        seg = F.affine(seg, *params, interpolation=F.InterpolationMode.NEAREST)
        
        # Resize
        params = transforms.RandomResizedCrop.get_params(img, scale=(0.5, 1), ratio=(0.7, 1.3))
        img = F.resized_crop(img, *params, self.size, interpolation=F.InterpolationMode.BILINEAR)
        seg = F.resized_crop(seg, *params, self.size, interpolation=F.InterpolationMode.NEAREST)
        
        # Horizontal flip
        if random.random() < 0.5:
            img = F.hflip(img)
            seg = F.hflip(seg)
        
        # Color jitter
        img = self.jitter(img)

        # Gen Mask
        msk = np.array(seg)
        random_num = random.randint(0,3)
        if random_num == 0:
            msk = cv2.erode(msk, self.kernels[np.random.randint(1, 30)])
        elif random_num == 1:
            msk = cv2.dilate(msk, self.kernels[np.random.randint(1, 30)])
        elif random_num == 2:
            msk = cv2.erode(msk, self.kernels[np.random.randint(1, 30)])
            msk = cv2.dilate(msk, self.kernels[np.random.randint(1, 30)])
        else:
            msk = cv2.dilate(msk, self.kernels[np.random.randint(1, 30)])
            msk = cv2.erode(msk, self.kernels[np.random.randint(1, 30)])

        # Cut Mask
        if random.random() < 0.25:
            h, w = msk.shape
            patch_size_h, patch_size_w = random.randint(h // 4, h // 2), random.randint(w // 4, w // 2)
            x1 = random.randint(0, w - patch_size_w)
            y1 = random.randint(0, h - patch_size_h)
            x2 = random.randint(0, w - patch_size_w)
            y2 = random.randint(0, h - patch_size_h)
            msk[y1:y1+patch_size_h, x1:x1+patch_size_w] = msk[y2:y2+patch_size_h, x2:x2+patch_size_w].copy()
        
        # To tensor
        img = F.to_tensor(img)
        msk = F.to_tensor(msk)
        seg = F.to_tensor(seg)
        
        return img, msk, seg
    

class CocoPanopticValidAugmentation:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img, seg):
        # Resize
        params = transforms.RandomResizedCrop.get_params(img, scale=(1, 1), ratio=(1., 1.))
        img = F.resized_crop(img, *params, self.size, interpolation=F.InterpolationMode.BILINEAR)
        seg = F.resized_crop(seg, *params, self.size, interpolation=F.InterpolationMode.NEAREST)
        
        # To tensor
        img = F.to_tensor(img)
        seg = F.to_tensor(seg)
        
        return img, seg