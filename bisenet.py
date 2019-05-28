import os
import torch
import torch.nn as nn
from loss import OhemCELoss
from resnet import resnet18
from config import DEVICE_IDS
device = torch.device("cuda:%d" % DEVICE_IDS[0])


def conv_bn(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv_sigmoid(in_channels, out_channels, kernel_size=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
        #nn.BatchNorm2d(out_channels),
        nn.Sigmoid()
    )


class SpartialPath(nn.Module):
    def __init__(self):
        super(SpartialPath, self).__init__()
        self.conv_block1 = conv_bn(in_channels=3, out_channels=64)
        self.conv_block2 = conv_bn(in_channels=64, out_channels=128)
        self.conv_block3 = conv_bn(in_channels=128, out_channels=256)
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x) 
        return x

class ContextPath(nn.Module):
    def __init__(self, num_classes):
        super(ContextPath, self).__init__()
        self.res18 = resnet18()
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.arm1 = AttentionRefinementModule(in_channels=256, out_channels=256)
        self.arm2 = AttentionRefinementModule(in_channels=512, out_channels=512)
 
    def forward(self, x):         
        #x3 channel 256; x4 channel 512
        x3, x4 = self.res18(x)
        #print("before arm x3.size():\t", x3.size())
        #print("before arm x4.size():\t", x4.size())
        tailf =  self.global_pool(x4)
        x3 = self.arm1(x3)
        x4 = self.arm2(x4)
        x4 = torch.mul(x4, tailf)
        #print("after arm x3.size():\t", x3.size())
        #print("after arm x4.size():\t", x4.size())

        #----upsampling-----
        x3 = torch.nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x4 = torch.nn.functional.interpolate(x4, scale_factor=4, mode='bilinear', align_corners=True)
        cx = torch.cat((x3, x4), dim=1)
        return cx, x3, x4 
        
class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_block = conv_sigmoid(in_channels, out_channels) 
        
    def forward(self, x):
        branch_x = self.global_pool(x)
        branch_x = self.conv_block(branch_x)
        x = torch.mul(x, branch_x)
        return x

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv_block1 = conv_bn(in_channels, out_channels, stride=1)         
        self.conv_block2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_block1(x)
        branch_x = self.conv_block2(x)
        mul_x = torch.mul(x, branch_x)
        result = torch.add(x, mul_x)
        return result 
         
class bisenet(nn.Module):
    def __init__(self, num_classes, training=False):
        super(bisenet, self).__init__()
        self.sp = SpartialPath()
        self.cp = ContextPath(num_classes)
        self.ffm = FeatureFusionModule(1024, num_classes)
        self.training = training
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1) 
        self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        #self.criterion_p = nn.CrossEntropyLoss(ignore_index=255).cuda()
        #self.criterion_16 = nn.CrossEntropyLoss(ignore_index=255).cuda()
        #self.criterion_32 = nn.CrossEntropyLoss(ignore_index=255).cuda()
        self.criterion_p = nn.CrossEntropyLoss(ignore_index=255).to(device)
        self.criterion_16 = nn.CrossEntropyLoss(ignore_index=255).to(device)
        self.criterion_32 = nn.CrossEntropyLoss(ignore_index=255).to(device)

    def forward(self, input):
        sp_x = self.sp(input)
        cp_x, cx1, cx2 = self.cp(input)
      
        if self.training: #---auxilory loss
            cx1_up = self.supervision1(cx1)                
            cx2_up = self.supervision2(cx2)
            cx1_up = torch.nn.functional.interpolate(cx1_up, scale_factor=8, mode='bilinear', align_corners=True)
            cx2_up = torch.nn.functional.interpolate(cx2_up, scale_factor=8, mode='bilinear', align_corners=True)
       
        #print("sp_x.size():\t", sp_x.size())
        #print("cp_x.size():\t", cp_x.size()) 
        result = self.ffm(sp_x, cp_x)
        #----upsampling-----
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear', align_corners=True)
        result = self.conv(result)
        #print("result.size():\t", result.size())
        #print("cx1_up.size():\t", cx1_up.size())
        #print("cx2_up.size():\t", cx2_up.size())
        if self.training:
            return result, cx1_up, cx2_up

        return result


    def get_loss(self, output, output_16, output_32, gt_mask):
        loss_p = self.criterion_p(output, gt_mask) 
        loss_16 = self.criterion_16(output_16, gt_mask)
        loss_32 = self.criterion_32(output_32, gt_mask)
        return loss_p, loss_16, loss_32

if __name__=="__main__":
    #x = torch.rand(1, 3, 640, 640)
    model = bisenet(19, training=True)
    #result, cx1, cx2 = model(x)
    state_dict = model.state_dict()
    print(len(state_dict.keys()))
