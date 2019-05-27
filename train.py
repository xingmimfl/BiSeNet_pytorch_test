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
    data_iter = iter(dataloader_train)
    for epoch in range(MAX_ITERS):
        lr = poly_lr_scheduler(optimizer, LEARNING_RATE, iter_count=epoch, max_iter=MAX_ITERS)
        try:
            images, masks = data_iter.next()
        except StopIteration:
            data_iter = iter(dataloader_train) 
            images, masks = data_iter.next()
        images = images.to(device) 
        masks = masks.to(device).squeeze()
        
        output, output_sup1, output_sup2 = model(images)
        loss_p, loss_16, loss_32 = model.get_loss(output, output_sup1, output_sup2, masks)
        loss = loss_p + loss_16 + loss_32
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #----save model----
        if epoch % TRAIN_SAVE_INTERVAL == 0:
            save_name = '_'.join([SUFFIX, "iter", str(epoch), '.model'])
            save_path = os.path.join(SNAPSHOT_PATH, save_name)
            torch.save(model, save_path)
            print("save model:\t", save_path)

def poly_lr_scheduler(optimizer, init_lr, iter_count, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    lr = init_lr*(1 - iter_count/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
          

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

    optimizer = torch.optim.SGD(
        [
            {'params':resnet_params, 'lr':LEARNING_RATE * 0.1},
            {'params':model_other_params}
        ],
        lr = LEARNING_RATE, 
        momentum = MOMENTUM,
        weight_decay = WEIGHT_DECAY,
    )
        
    train(model, optimizer, train_dataloader) 
