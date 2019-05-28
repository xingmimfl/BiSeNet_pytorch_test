import os
import sys
import time
import torch
import torch.nn as nn
import torchvision.transforms as transform
from bisenet import bisenet
from cityscapes import CityScapes
from config import *

device = torch.device("cuda:%d" % DEVICE_IDS[0])

def train(model, optimizer, dataloader_train):
    start_time = time.time()
    iter_time = start_time

    loss_avg = AverageMeter()
    data_iter = iter(dataloader_train)
    for epoch in range(MAX_ITERS):
        lr = poly_lr_scheduler(optimizer, LEARNING_RATE, iter_count=epoch, max_iter=MAX_ITERS)
        try:
            images, masks = data_iter.next()
        except StopIteration:
            data_iter = iter(dataloader_train) 
            images, masks = data_iter.next()
        #images = images.cuda()
        images = images.to(device)
        #masks = masks.cuda().squeeze()
        masks = masks.to(device).squeeze()
        
        output, output_sup1, output_sup2 = model(images)
        if len(DEVICE_IDS) >=2:
            loss_p, loss_16, loss_32 = model.module.get_loss(output, output_sup1, output_sup2, masks)
        else:
            loss_p, loss_16, loss_32 = model.get_loss(output, output_sup1, output_sup2, masks)
        loss = loss_p + loss_16 + loss_32
        loss_avg.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #----save model----
        if epoch % TRAIN_SAVE_INTERVAL == 0:
            save_name = '_'.join([SUFFIX, "iter", str(epoch), '.model'])
            save_path = os.path.join(SNAPSHOT_PATH, save_name)
            torch.save(model, save_path)
            print("save model:\t", save_path)
            print("[iter %d]" % epoch + " loss: %.4e" % loss_avg.val)
            loss_avg = AverageMeter()

            curr_time = time.time()
            print(" time cost: {:f}".format(curr_time - iter_time))
            iter_time = curr_time
            print(' Total time used: {:f}'.format(curr_time - start_time))
            print("")


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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
          

if __name__=="__main__":
    #----init dataloader----
    crop_size = [CROP_SIZE, CROP_SIZE]
    train_dataset = CityScapes(DATA_PATH, cropsize=crop_size, mode = 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=True)

    #-----init model-----
    model = bisenet(19, training=True)
    model.train() #---set status to training---- 
    resnet_state_dict = torch.load(RESNET_MODEL_PATH) #---load pretrained resnet model---
    #model.cp.res18.load_state_dict({k:v for k, v in resnet_state_dict.items() if k in model.cp.res18.state_dict().keys()})
    res18_dict = {k:v for k, v in resnet_state_dict.items() if k in model.cp.res18.state_dict().keys()}
    model.cp.res18.load_state_dict(res18_dict) 
    if len(DEVICE_IDS) >=2:
        #model = nn.DataParallel(model, device_ids=DEVICE_IDS).cuda()
        model = nn.DataParallel(model, device_ids=DEVICE_IDS).to(device)
    
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
