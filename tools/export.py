import os
import cv2
import torch
import random
import numpy as np
import sys
sys.path.append("./")
from core import enbisenetv2
from utils import get_argparser 

def load_weight(model, weight_file):
    checkpoints = torch.load(weight_file, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoints["model_state"])
    
    return model

def serialize2onnx():
    opts = get_argparser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = enbisenetv2(opts.num_classes)
    model = load_weight(model, opts.weight_path)
    model.eval()

    dummy_inputs = torch.randn(1, 3, 640, 640).to(device)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model, dummy_inputs, opts.onnx_saved_path + opts.onnx_name, 
                      verbose=True, input_names=input_names, do_constant_folding=True, 
                      output_names=output_names, opset_version=11, 
                      dynamic_axes={'input':{2:'height', 3:'width'}, 
                                    'output':{2:'height', 3:'width'}})

if __name__ == "__main__":
    # run serialize2onnx : python ./tools/export.py --num_classes 4 
    serialize2onnx()