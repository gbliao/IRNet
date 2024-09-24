import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn import functional as F
import transforms as trans
from torchvision import transforms

from dataset import vaild_dataset
import math
from ImageNet import ImageNet
import os
import datetime

import numpy as np
import cv2
from tqdm import tqdm
from skimage import img_as_ubyte
import logging
from imageio import imsave

from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
FM = Fmeasure()
WFM = WeightedFmeasure()
SM = Smeasure()
EM = Emeasure()
MAE = MAE()


def vaild(vaild_loader, model, vaild_save_path, vaild_save_path_edge):
    model.eval()
    with torch.no_grad():
        for i in range(vaild_loader.size):
            image, focals, gt, name = vaild_loader.load_data()  
            
            # Image 
            inputs_image = Variable(image.cuda())   
            # Focal stack 
            focals = F.interpolate(focals, (256,256), mode='bilinear', align_corners=True)
            inputs_focal = Variable(focals.cuda())   
            basize, dime, height, width = inputs_focal.size()
            inputs_focal = inputs_focal.view(1,basize,dime,height,width).transpose(0,1)
            inputs_focal = torch.cat(torch.chunk(inputs_focal, 12, dim=2), dim=1)
            inputs_focal = torch.cat(torch.chunk(inputs_focal, basize, dim=0), dim=1).squeeze()

            with torch.no_grad():
                _, _, sal_pre, edge_pre = model(inputs_image, inputs_focal)
                outputs_saliency = sal_pre[0]
                outputs_edge = edge_pre[0]
                
                res_ = F.interpolate(outputs_saliency, (256,256), mode='bilinear', align_corners=False)
                res_ = res_.sigmoid().data.cpu().numpy().squeeze()
                imsave(vaild_save_path + name[:-4] + '.png', img_as_ubyte(res_))
                
                res_edge = F.interpolate(outputs_edge, (256,256), mode='bilinear', align_corners=False)
                res_edge = res_edge.sigmoid().data.cpu().numpy().squeeze()
                imsave(vaild_save_path_edge + name[:-4] + '.png', img_as_ubyte(res_edge))                

        
        mask_name_list = sorted(os.listdir(mask_root))
        for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
            mask_path = os.path.join(mask_root, mask_name)
            pred_path = os.path.join(pred_root, mask_name[:-4] + '.png')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            FM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            MAE.step(pred=pred, gt=mask)
        fm = FM.get_results()["fm"]
        em = EM.get_results()["em"]
        sm = SM.get_results()["sm"]
        mae = MAE.get_results()["mae"]
        FM.precisions.clear()
        FM.recalls.clear()
        FM.adaptive_fms.clear()
        FM.changeable_fms.clear()
        EM.adaptive_ems.clear()
        EM.changeable_ems.clear()
        SM.sms.clear()
        MAE.maes.clear()
        new_fm = fm
        new_em = em
        new_sm = sm
        new_mae = mae
        del fm, sm, mae, em
        sm, maxFm, mae, adpFm, meanFm, maxEm = new_sm, new_fm["curve"].max(), new_mae, new_fm["adp"], new_fm["curve"].mean(), new_em["curve"].max(),
        return sm, maxFm, mae, adpFm, meanFm, maxEm

    
################################ Models ################################     
torch.set_num_threads(4)
net = ImageNet()
net.cuda()
net.eval()


evluation_model = ['Final_Model']  
for model_weight in evluation_model:
    model_path = './checkpoint/' + model_weight + '.pth'
    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict)
    print('Model loaded from {}'.format(model_path))

    ################################ Vaild ################################ 
    vaild_output_dir = './preds_vaild/'
    data_mask_root = '/dataset/SOD/Dataset_LightField'

    for vaild_dataset_name in ['HFUT_Test155','DUTLF_Test','LFSD_DLG','LFSD']:  
        vaild_image_root = '/dataset/SOD/Dataset_LightField/' + vaild_dataset_name + '/test_images/'
        vaild_focal_root = '/dataset/SOD/Dataset_LightField/' + vaild_dataset_name + '/test_focal/'
        vaild_gt_root = '/dataset/SOD/Dataset_LightField/' + vaild_dataset_name + '/test_masks/'
        vaild_save_path = './preds_vaild/' + model_weight + '/' + vaild_dataset_name + '/'
        vaild_save_path_edge = './preds_vaild/' + model_weight + '/' + vaild_dataset_name + '_edge/'
        vaild_loader = vaild_dataset(vaild_image_root,vaild_focal_root,vaild_gt_root)

        pred_root = os.path.join(vaild_output_dir, model_weight, vaild_dataset_name)
        mask_root = os.path.join(data_mask_root, vaild_dataset_name, 'test_masks')
        if not os.path.exists(vaild_output_dir):
            os.makedirs(vaild_output_dir)
        if not os.path.exists(vaild_save_path):
            os.makedirs(vaild_save_path)
        if not os.path.exists(vaild_save_path_edge):
            os.makedirs(vaild_save_path_edge)

        ################################ Log ################################ 
        logging.basicConfig(filename=vaild_output_dir+'Results.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO,filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

        sm, maxFm, mae, adpFm, meanFm, maxEm = vaild(vaild_loader, net, vaild_save_path, vaild_save_path_edge)

        print(' Sm:{:.3f} maxFm:{:.3f} MAE:{:.3f} adpFm:{:.3f} meanFm:{:.3f} maxEm:{:.3f} || [{}/{}] '
              .format(sm, maxFm, mae, adpFm, meanFm, maxEm, vaild_dataset_name, model_weight))
        logging.info(' Sm:{:.3f} maxFm:{:.3f} MAE:{:.3f} adpFm:{:.3f} meanFm:{:.3f} maxEm:{:.3f} || [{}/{}] '
                     .format(sm, maxFm, mae, adpFm, meanFm, maxEm, vaild_dataset_name, model_weight))
