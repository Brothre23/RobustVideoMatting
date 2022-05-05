import torch
import os
import json
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
import cv2


class YouTubeVISDataset(Dataset):
    def __init__(self, videodir, annfile, size, seq_length, seq_sampler, transform=None):
        self.videodir = videodir
        self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform
        
        with open(annfile) as f:
            data = json.load(f)

        self.masks = {}
        for ann in data['annotations']:
            if ann['category_id'] == 26: # person
                video_id = ann['video_id']
                if video_id not in self.masks:
                    self.masks[video_id] = [[] for _ in range(len(ann['segmentations']))]
                for frame, mask in zip(self.masks[video_id], ann['segmentations']):
                    if mask is not None:
                        frame.append(mask)
        
        self.videos = {}
        for video in data['videos']:
            video_id = video['id']
            if video_id in self.masks:
                self.videos[video_id] = video
        
        self.index = []
        for video_id in self.videos.keys():
            for frame in range(len(self.videos[video_id]['file_names'])):
                self.index.append((video_id, frame))
                
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        video_id, frame_id = self.index[idx]
        video = self.videos[video_id]
        frame_count = len(self.videos[video_id]['file_names'])
        H, W = video['height'], video['width']
        
        imgs, segs = [], []
        for t in self.seq_sampler(self.seq_length):
            frame = (frame_id + t) % frame_count

            filename = video['file_names'][frame]
            masks = self.masks[video_id][frame]
        
            with Image.open(os.path.join(self.videodir, filename)) as img:
                imgs.append(self._downsample_if_needed(img.convert('RGB'), Image.BILINEAR))
        
            seg = np.zeros((H, W), dtype=np.uint8)
            for mask in masks:
                seg |= self._decode_rle(mask)
            segs.append(self._downsample_if_needed(Image.fromarray(seg), Image.NEAREST))
            
        if self.transform is not None:
            imgs, msks, segs = self.transform(imgs, segs)
        
        return imgs, msks, segs
    
    def _decode_rle(self, rle):
        H, W = rle['size']
        msk = np.zeros(H * W, dtype=np.uint8)
        encoding = rle['counts']
        skip = 0
        for i in range(0, len(encoding) - 1, 2):
            skip += encoding[i]
            draw = encoding[i + 1]
            msk[skip : skip + draw] = 255
            skip += draw
        return msk.reshape(W, H).transpose()
    
    def _downsample_if_needed(self, img, resample):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h), resample)
        return img


class YouTubeVISAugmentation:
    def __init__(self, size):
        self.size = size
        self.jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.15)
        self.kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,100)]
    
    def __call__(self, imgs, segs):
        # Gen Mask
        msks = []
        for seg in segs:
            msk = np.array(seg.copy())
            msks.append(msk)
        for i in range(len(msks)):
            random_num = random.randint(0,3)
            if random_num == 0:
                msks[i] = cv2.erode(msks[i], self.kernels[np.random.randint(1, 30)])
            elif random_num == 1:
                msks[i] = cv2.dilate(msks[i], self.kernels[np.random.randint(1, 30)])
            elif random_num == 2:
                msks[i] = cv2.erode(msks[i], self.kernels[np.random.randint(1, 30)])
                msks[i] = cv2.dilate(msks[i], self.kernels[np.random.randint(1, 30)])
            else:
                msks[i] = cv2.dilate(msks[i], self.kernels[np.random.randint(1, 30)])
                msks[i] = cv2.erode(msks[i], self.kernels[np.random.randint(1, 30)])

        # Cut Mask
        for i in range(len(msks)):
            if random.random() < 0.25:
                h, w = msks[0].shape
                patch_size_h, patch_size_w = random.randint(h // 4, h // 2), random.randint(w // 4, w // 2)
                x1 = random.randint(0, w - patch_size_w)
                y1 = random.randint(0, h - patch_size_h)
                x2 = random.randint(0, w - patch_size_w)
                y2 = random.randint(0, h - patch_size_h)
                msks[i][y1:y1+patch_size_h, x1:x1+patch_size_w] = msks[i][y2:y2+patch_size_h, x2:x2+patch_size_w].copy()
        
        # To tensor
        imgs = torch.stack([F.to_tensor(img) for img in imgs])
        msks = torch.stack([F.to_tensor(msk) for msk in msks])
        segs = torch.stack([F.to_tensor(seg) for seg in segs])
        
        # Resize
        params = transforms.RandomResizedCrop.get_params(imgs, scale=(0.8, 1), ratio=(0.9, 1.1))
        imgs = F.resized_crop(imgs, *params, self.size, interpolation=F.InterpolationMode.BILINEAR)
        msks = F.resized_crop(segs, *params, self.size, interpolation=F.InterpolationMode.BILINEAR)
        segs = F.resized_crop(segs, *params, self.size, interpolation=F.InterpolationMode.BILINEAR)
        
        # Color jitter
        imgs = self.jitter(imgs)
        
        # Grayscale
        if random.random() < 0.05:
            imgs = F.rgb_to_grayscale(imgs, num_output_channels=3)
        
        # Horizontal flip
        if random.random() < 0.5:
            imgs = F.hflip(imgs)
            msks = F.hflip(segs)
            segs = F.hflip(segs)
        
        return imgs, msks, segs
