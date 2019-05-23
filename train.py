import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transform
from bisenet import bisenet
from cityscapes import CityScapes
from config import *

device = torch.device('cuda:%d' % device_id)

def train(model, optimizer, dataloader_train):
    for epoch in range(MAX_EPOCHS):
        #lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        for i, (images, masks) in enumerate(dataloader_train):
            images = images.to(device) 
            masks = masks.to(device).squeeze(t)
        
            output, output_sup1, output_sup2 = model(images)
            loss_p, loss_16, loss_32 = model.get_loss(output, output_sup1, output_sup2, masks)
            loss = loss_p + loss_16 + loss_32
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
          


if __name__=="__main__":
    #----init dataloader----
    crop_size = [CROP_SIZE, CROP_SIZE]
    train_dataset = CityScapes(DATA_PATH, cropsize=crop_size, mode = 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, num_workers=2, shuffle=True, pin_memory=True)

    #-----init model-----
    model = bisenet(19, training=True)
    model = model.to(device)
    model.train() #---set status to training---- 
    resnet_state_dict = torch.load(RESNET_MODEL_PATH) #---load pretrained resnet model---
    #model.cp.res18.load_state_dict({k:v for k, v in resnet_state_dict.items() if k in model.cp.res18.state_dict().keys()})
    res18_dict = {k:v for k, v in resnet_state_dict.items() if k in model.cp.res18.state_dict().keys()}
    model.cp.res18.load_state_dict(res18_dict) 
    
    #----fix first layers in resnet18----
    #optimizer = torch.optim.RMSprop(model.parameters(), LEARNING_RATE)

    #----learning rate in resnet18 is smaller than other layers-----
    resnet_params = []
    model_other_params = []
    for name, param in model.named_parameters():
        if "res" in name:
            resnet_params.append(param)
        else:
            model_other_params.append(param)
    optimizer = torch.optim.RMSprop([
        {'params':resnet_params, 'lr':LEARNING_RATE * 0.1},
        {'params':model_other_params}],
        lr = LEARNING_RATE)
        
    train(model, optimizer, train_dataloader) 
