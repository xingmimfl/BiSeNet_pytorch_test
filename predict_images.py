import os
import sys
import torch
from bisenet import bisenet
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
import cv2

device = torch.device("cuda:2")

if __name__=="__main__":
    with open('./cityscapes_info.json', 'r') as fr:
        labels_info = json.load(fr)
        color_map = {el['trainId']:el['color'] for el in labels_info}
    

    model_file_path = "models/bisenet_20190527_version1_iter_79900_.model"
    pretrained_model = torch.load(model_file_path)
    pretrained_model.eval()
    pretrained_state_dict = pretrained_model.state_dict()
    #print(pretrained_state_dict.keys())

    model = bisenet(19, training=False)
    state_dict = model.state_dict()
    model.load_state_dict({k[7:]:v for k, v in pretrained_state_dict.items() if k[7:] in state_dict}) 

    model.eval()
    model = model.to(device) 

    image_path = "lindau_000012_000019_leftImg8bit.png"    
    a_image = Image.open(image_path) 
    img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    a_image = img_transform(a_image)
    a_image = a_image.to(device)
    a_image = a_image.unsqueeze(dim=0)
    _, _, img_h, img_w = a_image.size()

    out_mask = model(a_image) 
    out_mask = out_mask.squeeze().argmax(dim=0)
    print("out_mask.size:\t", out_mask.size())
    out_mask = out_mask.cpu().numpy()
    
    a_new_image = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    for i in range(img_h):
        for j in range(img_w):
            value = out_mask[i,j]   
            a_new_image[i][j] = color_map[value]
    cv2.imwrite("hh.jpg", a_new_image)
