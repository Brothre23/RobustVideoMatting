"""
Expected directory format:

VideoMatte Train/Valid:
    ├──fgr/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
    ├── pha/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
        
ImageMatte Train/Valid:
    ├── fgr/
      ├── sample1.jpg
      ├── sample2.jpg
    ├── pha/
      ├── sample1.jpg
      ├── sample2.jpg

Background Image Train/Valid
    ├── sample1.png
    ├── sample2.png

Background Video Train/Valid
    ├── 0000/
      ├── 0000.jpg/
      ├── 0001.jpg/

"""

DATA_PATHS = {
    'videomatte_sd': {
        'train':
        '../dataset/VideoMatte240K/VideoMatte240K_JPEG_SD/train',
        'valid':
        '../dataset/VideoMatte240K/VideoMatte240K_JPEG_SD/valid',
    },
    'videomatte_hd': {
        'train':
        '../dataset/VideoMatte240K/VideoMatte240K_JPEG_HD/train',
        'valid':
        '../dataset/VideoMatte240K/VideoMatte240K_JPEG_HD/valid',
    },
    'background_images': {
        'train':
        '../dataset/Background_Images/train',
        'valid':
        '../dataset/Background_Images/valid',
    },
    'background_videos': {
        'train':
        '../dataset//Background_Videos/train',
        'valid':
        '../dataset//Background_Videos/valid',
    },
    'natural_images': '../dataset/Real_Human/image_allframe'
    ,
    'imagematte': {
        'train':
        '../dataset/ImageMatte/train',
        'valid':
        '../dataset/ImageMatte/valid',
    },
    'coco_panoptic': {
        'imgdir':
        '../dataset/coco/train2017/',
        'anndir':
        '../dataset/coco/panoptic_train2017/',
        'annfile':
        '../dataset/coco/annotations/panoptic_train2017.json',
    },
    'spd': {
        'imgdir':
        '../dataset/SuperviselyPersonDataset/img',
        'segdir':
        '../dataset/SuperviselyPersonDataset/seg',
    },
    'youtubevis': {
        'videodir':
        '../dataset/YouTubeVIS/train/JPEGImages',
        'annfile':
        '../dataset/YouTubeVIS/train/instances.json',
    }
}