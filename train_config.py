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
    'videomatte': {
        'train':
        '/media/brothre23/1T-HDD/RobustVideoMatting/VideoMatte240K_JPEG_SD/train',
        'valid':
        '/media/brothre23/1T-HDD/RobustVideoMatting/VideoMatte240K_JPEG_SD/valid',
    },
    'imagematte': {
        'train':
        '/media/brothre23/1T-HDD/RobustVideoMatting/ImageMatte/train',
        'valid':
        '/media/brothre23/1T-HDD/RobustVideoMatting/ImageMatte/valid',
    },
    'background_images': {
        'train':
        '/media/brothre23/1T-HDD/RobustVideoMatting/Background_Images/train',
        'valid':
        '/media/brothre23/1T-HDD/RobustVideoMatting/Background_Images/valid',
    },
    'background_videos': {
        'train':
        '/media/brothre23/1T-HDD/RobustVideoMatting/Background_Videos/train',
        'valid':
        '/media/brothre23/1T-HDD/RobustVideoMatting/Background_Videos/valid',
    },
    'natural_images': '/media/brothre23/1T-HDD/RobustVideoMatting/Real_Human/image_allframe'
    ,
    'coco_panoptic': {
        'imgdir':
        '/media/brothre23/1T-HDD/RobustVideoMatting/coco/train2017/',
        'anndir':
        '/media/brothre23/1T-HDD/RobustVideoMatting/coco/panoptic_train2017/',
        'annfile':
        '/media/brothre23/1T-HDD/RobustVideoMatting/coco/annotations/panoptic_train2017.json',
    },
    'spd': {
        'imgdir':
        '/media/brothre23/1T-HDD/RobustVideoMatting/SuperviselyPersonDataset/img',
        'segdir':
        '/media/brothre23/1T-HDD/RobustVideoMatting/SuperviselyPersonDataset/seg',
    },
    'youtubevis': {
        'videodir':
        '/media/brothre23/1T-HDD/RobustVideoMatting/YouTubeVIS/train/JPEGImages',
        'annfile':
        '/media/brothre23/1T-HDD/RobustVideoMatting/YouTubeVIS/train/instances.json',
    }
}