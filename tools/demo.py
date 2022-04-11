import os
import cv2
import torch
import random
import numpy as np
import sys
sys.path.append("./")
from PIL import Image
from core import enbisenetv2
from utils import get_argparser 
import torch.nn.functional as F
from utils import transforms as et 
from torchvision import transforms

def load_weight(model, weight_file):
    checkpoints = torch.load(weight_file, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoints["model_state"])
    
    return model

class ExtToTensor(object):
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type
    def __call__(self, pic):
        import torchvision.transforms.functional as ttf
        if self.normalize:
            return ttf.to_tensor(pic)
        else:
            return torch.from_numpy( np.array( pic, dtype=np.float32).transpose(2, 0, 1) )

class ExtNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        import torchvision.transforms.functional as ttf
        return ttf.normalize(tensor, self.mean, self.std)
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ExtLetterResize(object):
    def __init__(self, new_shape=(640, 640), color=(114, 114, 114), stride=32):
        self.new_shape = new_shape
        self.color = color
        self.stride = stride
    
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.asarray(img)

        shape = img.shape[:2] # h, w
        if isinstance(self.new_shape, int):
            self.new_shape = (self.new_shape, self.new_shape)
        
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        new_unpad = ( int( round(shape[1] * r) ), int( round(shape[0] * r) ) ) # w, h
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1] # w, h
        dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = 0, dh
        left, right = 0, dw
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)
        return img

def demo(image:Image):
    opts = get_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_transform = transforms.Compose([
        ExtLetterResize((opts.crop_size, opts.crop_size)), 
        ExtToTensor(),
        ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    
    model = enbisenetv2(opts.num_classes)
    model = load_weight(model, opts.weight_path)
    model.to(device)
    model.eval()

    image = image.convert("RGB")
    input_tensor = val_transform(image).unsqueeze(0).to(device)

    outputs = model(input_tensor) 
    outputs = F.softmax(outputs, dim=1)
    _, outputs = torch.max(outputs, dim=1)

    output = outputs.data.cpu().numpy()[0]
    output = output.astype(np.uint8) * 50
    cv2.imshow("output", output)
    cv2.waitKey()

if __name__ == "__main__":
    # run demo : python .\tools\demo.py --num_classes 4 --crop_size 640
    image = Image.open("C:/Jyd/test/jyd_test/images2/detection/image--002a173d320242158e79276feb9b144d.jpg")
    demo(image)