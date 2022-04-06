from utils import transforms as et 
from dataset import VOCSegmentation
import numpy as np
import os
import torch
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('agg')

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        et.ExtResize(size=opts.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        # et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        et.ExtRandomHorizontalFlip(),
        # et.PerspectiveTransform(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    train_dst = VOCSegmentation(args=opts, image_set='train', transform=train_transform)
    val_dst = VOCSegmentation(args=opts, image_set='val', transform=val_transform)

    return train_dst, val_dst

def get_params(model):
        backbone, heads = [], []
        for name, module in model.named_children():
            if name == "detail" or name == "segment":
                for name, param in module.named_parameters():
                    backbone.append(param)
            else:
                for name, param in module.named_parameters():
                    heads.append(param)
            
        return backbone, heads
    
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    # 网络前向传播的时候不会保存梯度信息，节约显存
    with torch.no_grad():
        for i, (images, labels, _) in tqdm(enumerate(loader)):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy() #[batch_size,513,513]
            targets = labels.cpu().numpy() #[batch_size,513,513]

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

        score = metrics.get_results()
    return score, ret_samples