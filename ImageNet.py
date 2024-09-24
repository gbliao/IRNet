import torch.nn as nn
from torch.nn import functional as F
from resnet import * 



class ImageNet(nn.Module):
    def __init__(self, args=None):
        super(ImageNet, self).__init__()

         ################################ backbone ################################
        self.rgb_backbone = ResNet(BasicBlock, [3, 4, 6, 3])
        self.fs_backbone = ResNet(BasicBlock, [3, 4, 6, 3])
        self.load_pretrained_model('./resnet34-333f7ec4.pth')

        self.relu = nn.ReLU()
        k_channel = 128
        cp = []
        cp.append(nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), self.relu, nn.Conv2d(64, 64, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), self.relu, nn.Conv2d(128, k_channel, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), self.relu, nn.Conv2d(128, k_channel, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), self.relu, nn.Conv2d(128, k_channel, 3, 1, 1), self.relu))
        self.CP = nn.ModuleList(cp)

        cp_fs = []
        cp_fs.append(nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), self.relu, nn.Conv2d(64, 64, 3, 1, 1), self.relu))
        cp_fs.append(nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), self.relu, nn.Conv2d(128, k_channel, 3, 1, 1), self.relu))
        cp_fs.append(nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), self.relu, nn.Conv2d(128, k_channel, 3, 1, 1), self.relu))
        cp_fs.append(nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), self.relu, nn.Conv2d(128, k_channel, 3, 1, 1), self.relu))
        self.CP_FS = nn.ModuleList(cp_fs)
        
        self.Exploit_3D_layer3 = Exploit_3D(k_channel,k_channel)
        self.Exploit_3D_layer4 = Exploit_3D(k_channel,k_channel)
        self.Exploit_3D_layer5 = Exploit_3D(k_channel,k_channel)
        
        self.agg_layer3 = Agg_layer(k_channel*12)
        self.agg_layer4 = Agg_layer(k_channel*12)
        self.agg_layer5 = Agg_layer(k_channel*12)
        
        self.R_sal_fea5 = nn.Sequential(
            nn.Conv2d(k_channel, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU())
        self.R_sal_pre5 = nn.Sequential(nn.Conv2d(k_channel, 1, kernel_size=1))        
        self.R_edge_fea5 = nn.Sequential(
            nn.Conv2d(1, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU(),
            nn.Conv2d(k_channel, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU())
        self.R_edge_pre5 = nn.Sequential(nn.Conv2d(k_channel, 1, kernel_size=1))
        self.ContourFusionRF_54 = ContourFusion(k_channel)
        self.fuseRF_54 = FuseGate(k_channel)
      
        self.R_sal_fea4 = nn.Sequential(
            nn.Conv2d(k_channel, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU())
        self.R_sal_pre4 = nn.Sequential(nn.Conv2d(k_channel, 1, kernel_size=1))        
        self.R_edge_fea4 = nn.Sequential(
            nn.Conv2d(1, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU(),
            nn.Conv2d(k_channel, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU())
        self.R_edge_pre4 = nn.Sequential(nn.Conv2d(k_channel, 1, kernel_size=1))
        self.ContourFusionRF_43 = ContourFusion(k_channel)
        self.fuseRF_43 = FuseGate(k_channel)
             
        self.T_sal_fea5 = nn.Sequential(
            nn.Conv2d(k_channel, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU())
        self.T_sal_pre5 = nn.Sequential(nn.Conv2d(k_channel, 1, kernel_size=1))        
        self.T_edge_fea5 = nn.Sequential(
            nn.Conv2d(1, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU(),
            nn.Conv2d(k_channel, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU())
        self.T_edge_pre5 = nn.Sequential(nn.Conv2d(k_channel, 1, kernel_size=1))
        self.ContourFusionFR_54 = ContourFusion(k_channel)
        self.fuseFR_54 = FuseGate(k_channel)
        
        self.T_sal_fea4 = nn.Sequential(
            nn.Conv2d(k_channel, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU())
        self.T_sal_pre4 = nn.Sequential(nn.Conv2d(k_channel, 1, kernel_size=1))        
        self.T_edge_fea4 = nn.Sequential(
            nn.Conv2d(1, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU(),
            nn.Conv2d(k_channel, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU())
        self.T_edge_pre4 = nn.Sequential(nn.Conv2d(k_channel, 1, kernel_size=1))
        self.fuseFR_43 = FuseGate(k_channel)
        self.ContourFusionFR_43 = ContourFusion(k_channel)

        self.fusion_sal_fea = nn.Sequential(
            nn.Conv2d(k_channel, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU(),
            nn.Conv2d(k_channel, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU())
        self.fusion_sal_pre = nn.Sequential(nn.Conv2d(k_channel, 1, kernel_size=1))     
        self.fusion_edge_fea = nn.Sequential(
            nn.Conv2d(1, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU(),
            nn.Conv2d(k_channel, k_channel, kernel_size=3, padding=1), nn.GroupNorm(32, k_channel), nn.PReLU())
        self.fusion_edge_pre = nn.Sequential(nn.Conv2d(k_channel, 1, kernel_size=1))    
        
    def load_pretrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        print('Loading the pretrained weight for backbone!')
        # rgb
        model_dict = self.rgb_backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.rgb_backbone.load_state_dict(model_dict)
        print('Loaded the pretrained weight for rgb_backbone!')
        # Focal stack  self.fs_backbone
        self.fs_backbone.load_state_dict(model_dict)    
        print('Loaded the pretrained weight for fs_backbone!')

    def forward(self, image_Input, FS_Input):
        B = image_Input.shape[0]
        rgb_feature_extract = []
        fs_feature_extract = []

        tmp_rgb = self.rgb_backbone(image_Input) # [B,64,56*56] [B,128,28*28] [B,320,14*14] [B,512,7*7]
        tmp_FS = self.fs_backbone(FS_Input)

        for i in range(4):
            rgb_feature_extract.append(self.CP[i](tmp_rgb[i+1])) 
        rgb_fea_1_4, rgb_fea_1_8, rgb_fea_1_16, rgb_fea_1_32 = rgb_feature_extract # 1*C*H*W
        Teacher_layers = [rgb_fea_1_4, rgb_fea_1_8, rgb_fea_1_16, rgb_fea_1_32]
        
        for i in range(4):
            fs_feature_extract.append(self.CP_FS[i](tmp_FS[i+1])) 
        FS_fea_1_4, FS_fea_1_8, FS_fea_1_16, FS_fea_1_32 = fs_feature_extract # 12*C*H*W
        Student_layers = [FS_fea_1_4, FS_fea_1_8, FS_fea_1_16, FS_fea_1_32]

        FS_fea_1_8 = FS_fea_1_8.unsqueeze(2)  # 12*C*1*H*W
        FS_fea_1_8 = torch.cat(torch.chunk(FS_fea_1_8, 12, dim=0), dim=2)  # 1*C*12*H*W
        FS_fea_1_16 = FS_fea_1_16.unsqueeze(2)  
        FS_fea_1_16 = torch.cat(torch.chunk(FS_fea_1_16, 12, dim=0), dim=2)    
        FS_fea_1_32 = FS_fea_1_32.unsqueeze(2)  
        FS_fea_1_32 = torch.cat(torch.chunk(FS_fea_1_32, 12, dim=0), dim=2)    
        
        # Exploit_3D 
        FS_1_8 = self.Exploit_3D_layer3(FS_fea_1_8)
        FS_1_16 = self.Exploit_3D_layer4(FS_fea_1_16)
        FS_1_32 = self.Exploit_3D_layer5(FS_fea_1_32)

        
        FS_fea_1_8 = FS_1_8.view(FS_1_8.shape[0], FS_1_8.shape[1] * FS_1_8.shape[2], FS_1_8.shape[3], FS_1_8.shape[4]).contiguous()
        FS_fea_1_16 = FS_1_16.view(FS_1_16.shape[0], FS_1_16.shape[1] * FS_1_16.shape[2], FS_1_16.shape[3], FS_1_16.shape[4]).contiguous()        
        FS_fea_1_32 = FS_1_32.view(FS_1_32.shape[0], FS_1_32.shape[1] * FS_1_32.shape[2], FS_1_32.shape[3], FS_1_32.shape[4]).contiguous()
        
        FS_fea_1_8 = self.agg_layer3(FS_fea_1_8)
        FS_fea_1_16 = self.agg_layer4(FS_fea_1_16)
        FS_fea_1_32 = self.agg_layer5(FS_fea_1_32)        
           
        R_sal_fea5 = self.R_sal_fea5(rgb_fea_1_32)  # sal feature
        R_sal_pre5 = self.R_sal_pre5(R_sal_fea5)      # sal pre
        R_edge_fea5 = self.R_edge_fea5(R_sal_pre5)   # edge feature
        R_edge_pre5 = self.R_edge_pre5(R_edge_fea5)   # edge pre
        R_sal_fea5 = F.interpolate(R_sal_fea5, size=rgb_fea_1_16.size()[2:], mode='bilinear', align_corners=True)
        R_edge_fea5 = F.interpolate(R_edge_fea5, size=rgb_fea_1_16.size()[2:], mode='bilinear', align_corners=True)
        R_edge_pre5 = F.interpolate(R_edge_pre5, size=rgb_fea_1_16.size()[2:], mode='bilinear', align_corners=True)
        fuseRF_54 = self.fuseRF_54(torch.cat((FS_fea_1_16, rgb_fea_1_16, R_edge_fea5, R_sal_fea5), 1))
        fusionRF_54 = self.ContourFusionRF_54(fuseRF_54, R_edge_pre5)
        
        R_sal_fea4 = self.R_sal_fea4(fusionRF_54)  
        R_sal_pre4 = self.R_sal_pre4(R_sal_fea4)      
        R_edge_fea4 = self.R_edge_fea4(R_sal_pre4)   
        R_edge_pre4 = self.R_edge_pre4(R_edge_fea4)   
        R_sal_fea4 = F.interpolate(R_sal_fea4, size=rgb_fea_1_8.size()[2:], mode='bilinear', align_corners=True)
        R_edge_fea4 = F.interpolate(R_edge_fea4, size=rgb_fea_1_8.size()[2:], mode='bilinear', align_corners=True)
        R_edge_pre4 = F.interpolate(R_edge_pre4, size=rgb_fea_1_8.size()[2:], mode='bilinear', align_corners=True)
        fuseRF_43 = self.fuseRF_43(torch.cat((FS_fea_1_8, rgb_fea_1_8, R_edge_fea4, R_sal_fea4), 1))
        fusionRF_43 = self.ContourFusionRF_43(fuseRF_43, R_edge_pre4)        

        T_sal_fea5 = self.T_sal_fea5(FS_fea_1_32)  
        T_sal_pre5 = self.T_sal_pre5(T_sal_fea5)      
        T_edge_fea5 = self.T_edge_fea5(T_sal_pre5)  
        T_edge_pre5 = self.T_edge_pre5(T_edge_fea5)  
        T_sal_fea5 = F.interpolate(T_sal_fea5, size=rgb_fea_1_16.size()[2:], mode='bilinear', align_corners=True)
        T_edge_fea5 = F.interpolate(T_edge_fea5, size=rgb_fea_1_16.size()[2:], mode='bilinear', align_corners=True)
        T_edge_pre5 = F.interpolate(T_edge_pre5, size=rgb_fea_1_16.size()[2:], mode='bilinear', align_corners=True)
        fuseFR_54 = self.fuseFR_54(torch.cat((rgb_fea_1_16, FS_fea_1_16, T_edge_fea5, T_sal_fea5), 1))
        fusionFR_54 = self.ContourFusionFR_54(fuseRF_54, T_edge_pre5)        
        
        T_sal_fea4 = self.T_sal_fea4(fusionFR_54)  
        T_sal_pre4 = self.T_sal_pre4(T_sal_fea4)      
        T_edge_fea4 = self.T_edge_fea4(T_sal_pre4)   
        T_edge_pre4 = self.T_edge_pre4(T_edge_fea4)   
        T_sal_fea4 = F.interpolate(T_sal_fea4, size=rgb_fea_1_8.size()[2:], mode='bilinear', align_corners=True)
        T_edge_fea4 = F.interpolate(T_edge_fea4, size=rgb_fea_1_8.size()[2:], mode='bilinear', align_corners=True)
        T_edge_pre4 = F.interpolate(T_edge_pre4, size=rgb_fea_1_8.size()[2:], mode='bilinear', align_corners=True)
        fuseFR_43 = self.fuseFR_43(torch.cat((rgb_fea_1_8, FS_fea_1_8, T_edge_fea4, T_sal_fea4), 1))
        fusionFR_43 = self.ContourFusionFR_43(fuseRF_43, T_edge_pre4)  
        
        
        fusion_sal_fea = fusionRF_43 + fusionFR_43
        fusion_sal_pre = self.fusion_sal_pre(self.fusion_sal_fea(fusion_sal_fea))
        fusion_edge_pre = self.fusion_edge_pre(self.fusion_edge_fea(fusion_sal_pre))  
        
        
        R_sal_pre5 = F.interpolate(R_sal_pre5, size=image_Input.size()[2:], mode='bilinear', align_corners=True)
        R_sal_pre4 = F.interpolate(R_sal_pre4, size=image_Input.size()[2:], mode='bilinear', align_corners=True)
        T_sal_pre5 = F.interpolate(T_sal_pre5, size=image_Input.size()[2:], mode='bilinear', align_corners=True)
        T_sal_pre4 = F.interpolate(T_sal_pre4, size=image_Input.size()[2:], mode='bilinear', align_corners=True)
        fusion_sal_pre = F.interpolate(fusion_sal_pre, size=image_Input.size()[2:], mode='bilinear', align_corners=True)  

        R_edge_pre5 = F.interpolate(R_edge_pre5, size=image_Input.size()[2:], mode='bilinear', align_corners=True)
        R_edge_pre4 = F.interpolate(R_edge_pre4, size=image_Input.size()[2:], mode='bilinear', align_corners=True)
        T_edge_pre5 = F.interpolate(T_edge_pre5, size=image_Input.size()[2:], mode='bilinear', align_corners=True)
        T_edge_pre4 = F.interpolate(T_edge_pre4, size=image_Input.size()[2:], mode='bilinear', align_corners=True)
        fusion_edge_pre = F.interpolate(fusion_edge_pre, size=image_Input.size()[2:], mode='bilinear', align_corners=True)  
        

        sal_pre = [fusion_sal_pre, R_sal_pre5, R_sal_pre4, T_sal_pre5, T_sal_pre4]
        edge_pre = [fusion_edge_pre, R_edge_pre5, R_edge_pre4, T_edge_pre5, T_edge_pre4]
        
        return Teacher_layers, Student_layers, sal_pre, edge_pre

    

class Exploit_3D(nn.Module):
    def __init__(self, inp, out_channel, dilation=[1,2,3]):
        super(Exploit_3D, self).__init__()
        mid_dim = out_channel//2
        
        self.conv1 = nn.Sequential(nn.Conv3d(inp, mid_dim, [1, 1, 1]), nn.BatchNorm3d(mid_dim), nn.ReLU())
        self.mid_conv1 = nn.Conv3d(mid_dim, mid_dim, kernel_size=[3, 3, 3], padding=[1,dilation[0],dilation[0]], groups=mid_dim, dilation=[1,dilation[0],dilation[0]])
        self.mid_conv2 = nn.Conv3d(mid_dim, mid_dim, kernel_size=[3, 3, 3], padding=[1,dilation[1],dilation[1]], groups=mid_dim, dilation=[1,dilation[1],dilation[1]])
        self.mid_conv3 = nn.Conv3d(mid_dim, mid_dim, kernel_size=[3, 3, 3], padding=[1,dilation[2],dilation[2]], groups=mid_dim, dilation=[1,dilation[2],dilation[2]])
        
        self.mid_bnact = nn.Sequential(nn.BatchNorm3d(mid_dim), nn.ReLU(inplace=True))
        self.out_conv = nn.Sequential(nn.Conv3d(mid_dim, out_channel, [1, 1, 1], bias=False))

    def forward(self, input_x):
        x = self.conv1(input_x)
        x = self.mid_conv1(x) + self.mid_conv2(x) + self.mid_conv3(x)
        x = self.mid_bnact(x)
        return input_x + self.out_conv(x)

    
class Agg_layer(nn.Module):
    def __init__(self, channel, reduction=12):
        super(Agg_layer, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.agg = nn.Sequential(nn.Conv2d(channel, channel//12, 3, 1, 1), self.relu)
        
    def forward(self, x):
        
        return self.agg(x)
    
    
class ContourFusion(nn.Module):
    def __init__(self, channel):
        super(ContourFusion, self).__init__()
        self.gate = nn.Sequential(nn.Conv2d(channel , channel, kernel_size=3, padding=1),
                                  nn.GroupNorm(32, channel), nn.PReLU(),
                                   nn.Conv2d(channel , 1, kernel_size=1),
                                   nn.Sigmoid())

        self.conv = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                                  nn.GroupNorm(32, channel), nn.PReLU())

    def forward(self, x, edge):
        cm1 = self.gate(x)                  
        e1 = cm1 * torch.sigmoid(edge)      
        x = x * e1
        return self.conv(x)
    
    
class FuseGate(nn.Module):
    def __init__(self, in_planes, reduction=16):
        self.init__ = super(FuseGate, self).__init__()
        self.in_planes = in_planes
        
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        self.fc1 = nn.Conv2d(in_planes*4, in_planes//2, kernel_size = 1)  
        self.fc2 = nn.Conv2d(in_planes//2, in_planes*4, kernel_size = 1)
        self.merge_conv = nn.Sequential(
            nn.Conv2d(in_planes*4, in_planes, 3, 1, 1), self.relu,
            nn.Conv2d(in_planes, in_planes, 3, 1, 1), self.relu)   
        
    def forward(self, x):
        b, c, h, w = x.size()
        x1, x2, x3, x4 = torch.split(x, c // 4, dim=1)
        
        x1_gap = nn.AvgPool2d(x1.shape[2:])(x1).view(len(x1), c // 4, 1, 1)
        x2_gap = nn.AvgPool2d(x2.shape[2:])(x2).view(len(x2), c // 4, 1, 1)
        x3_gap = nn.AvgPool2d(x3.shape[2:])(x3).view(len(x3), c // 4, 1, 1)
        x4_gap = nn.AvgPool2d(x4.shape[2:])(x4).view(len(x4), c // 4, 1, 1)
        
        stack_gap = torch.cat([x1_gap, x2_gap, x3_gap, x4_gap], dim=1)   # bx4cx1x1
        
        stack_gap = self.fc1(stack_gap)
        stack_gap = self.relu(stack_gap)
        stack_gap = self.fc2(stack_gap)

        w1, w2, w3, w4 = torch.split(stack_gap, c // 4, dim=1)
        nx1 = x1 * w1 
        nx2 = x2 * w2 
        nx3 = x3 * w3 
        nx4 = x4 * w4 

        merge_feature = torch.cat( [nx1, nx2, nx3, nx4], dim=1)
        merge_feature = self.merge_conv(merge_feature)

        return merge_feature

if __name__ == '__main__':
    net = ImageNet()
    print("##############")
    
    