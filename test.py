# import torch

# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True).to('cuda:0')
# model.eval()

# from PIL import Image
# from torchvision import transforms, utils
# input_image = Image.open('input.png')
# input_image = input_image.convert("RGB")
# preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0)
# input_batch = input_batch.to('cuda:0')

# with torch.no_grad():
#     output = model(input_batch)['out'][0]

# output = output.softmax(dim=0)
# output = output[15, :, :]
# output = (output > 0.5).float()

# utils.save_image(output, 'output.jpg')

import segmentation_models_pytorch as smp
import torch

model = smp.DeepLabV3Plus(
    encoder_name='resnet50',
    in_channels=3,
    classes=20
)

model.load_state_dict(torch.load('/home/caig/下載/best_deeplabv3plus_resnet50_voc_os16.pth')['model_state'])
