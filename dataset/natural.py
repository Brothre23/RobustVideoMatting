import os
import random
from torch.utils.data import Dataset
from PIL import Image

from .augmentation import MotionAugmentation


class NaturalImageDataset(Dataset):
    def __init__(self,
                 natural_image_dir,
                 background_image_dir,
                 background_video_dir,
                 size,
                 seq_length,
                 seq_sampler,
                 transform=None):
        self.background_image_dir = background_image_dir
        self.background_image_files = os.listdir(background_image_dir)
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted(os.listdir(background_video_dir))
        self.background_video_frames = [sorted(os.listdir(os.path.join(background_video_dir, clip)))
                                        for clip in self.background_video_clips]

        self.natural_image_dir = natural_image_dir
        self.natural_image_clips = sorted(os.listdir(natural_image_dir))
        self.natural_image_frames = [sorted(os.listdir(os.path.join(natural_image_dir, clip))) 
                                    for clip in self.natural_image_clips]
        self.natural_image_idx = [(clip_idx, frame_idx) 
                                    for clip_idx in range(len(self.natural_image_clips)) 
                                    for frame_idx in range(0, len(self.natural_image_frames[clip_idx]), seq_length)]

        self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform

    def __len__(self):
        return len(self.natural_image_idx)
    
    def __getitem__(self, idx):
        if random.random() < 0.5:
            bgrs = self._get_random_image_background()
        else:
            bgrs = self._get_random_video_background()
        
        srcs = self._get_natural_image(idx)
        
        if self.transform is not None:
            srcs, _, bgrs, _ = self.transform(srcs, srcs, bgrs, bgrs)
        
        return srcs, bgrs
    
    def _get_random_image_background(self):
        with Image.open(os.path.join(self.background_image_dir, random.choice(self.background_image_files))) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        bgrs = [bgr] * self.seq_length
        return bgrs
    
    def _get_random_video_background(self):
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count - self.seq_length)))
        clip = self.background_video_clips[clip_idx]
        bgrs = []
        for i in self.seq_sampler(self.seq_length):
            frame_idx_t = frame_idx + i
            frame = self.background_video_frames[clip_idx][frame_idx_t % frame_count]
            with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
                bgr = self._downsample_if_needed(bgr.convert('RGB'))
            bgrs.append(bgr)
        return bgrs
    
    def _get_natural_image(self, idx):
        clip_idx, frame_idx = self.natural_image_idx[idx]
        clip = self.natural_image_clips[clip_idx]
        frame_count = len(self.natural_image_frames[clip_idx])
        srcs = []
        for i in self.seq_sampler(self.seq_length):
            frame = self.natural_image_frames[clip_idx][(frame_idx + i) % frame_count]
            with Image.open(os.path.join(self.natural_image_dir, clip, frame)) as src:
                    src = self._downsample_if_needed(src.convert('RGB'))
            srcs.append(src)
        return srcs
    
    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img

class NaturalImageAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0.3,
            prob_bgr_affine=0.3,
            prob_noise=0.1,
            prob_color_jitter=0.3,
            prob_grayscale=0.02,
            prob_sharpness=0.1,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
        )
