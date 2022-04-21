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
        '../../下載/RobustVideoMatting/VideoMatte240K/VideoMatte240K_JPEG_SD/train',
        'valid':
        '../../下載/RobustVideoMatting/VideoMatte240K/VideoMatte240K_JPEG_SD/valid',
    },
    'videomatte_hd': {
        'train':
        '../../下載/RobustVideoMatting/VideoMatte240K/VideoMatte240K_JPEG_HD/train',
        'valid':
        '../../下載/RobustVideoMatting/VideoMatte240K/VideoMatte240K_JPEG_HD/valid',
    },
    'background_images': {
        'train':
        '../../下載/RobustVideoMatting/Background_Images/train',
        'valid':
        '../../下載/RobustVideoMatting/Background_Images/valid',
    },
    'background_videos': {
        'train':
        '../../下載/RobustVideoMatting//Background_Videos/train',
        'valid':
        '../../下載/RobustVideoMatting//Background_Videos/valid',
    },
    'natural_images': '../../下載/RobustVideoMatting/Real_Human/image_allframe'
    ,
    'imagematte': {
        'train':
        '../../下載/RobustVideoMatting/ImageMatte/train',
        'valid':
        '../../下載/RobustVideoMatting/ImageMatte/valid',
    },
    'coco_panoptic': {
        'imgdir':
        '../../下載/RobustVideoMatting/coco/train2017/',
        'anndir':
        '../../下載/RobustVideoMatting/coco/panoptic_train2017/',
        'annfile':
        '../../下載/RobustVideoMatting/coco/annotations/panoptic_train2017.json',
    },
    'spd': {
        'imgdir':
        '../../下載/RobustVideoMatting/SuperviselyPersonDataset/img',
        'segdir':
        '../../下載/RobustVideoMatting/SuperviselyPersonDataset/seg',
    },
    'youtubevis': {
        'videodir':
        '../../下載/RobustVideoMatting/YouTubeVIS/train/JPEGImages',
        'annfile':
        '../../下載/RobustVideoMatting/YouTubeVIS/train/instances.json',
    }
}