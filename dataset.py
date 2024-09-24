from PIL import Image
from torch.utils import data
from torch.nn import functional as F
import transforms as trans
from torchvision import transforms
import random
import os
import numpy as np
import torch
import scipy.io as sio
import glob


def load_list(dataset_name, data_root):    # dataset_name: TrainSet_DUTLF_HFUT, data_root: /dataset/Dataset_LightField/
    images, labels, contours, focal_image = [], [], [], []
    
    img_root = data_root + dataset_name + '/DUTLF_Train/' +'/train_images/'
    HFUT_root = data_root + dataset_name + '/HFUT_Train/' +'/train_images/'
    img_files = os.listdir(img_root)
    HFUT_files = os.listdir(HFUT_root) * 11
    
    for img in img_files:
        images.append(img_root + img[:-4]+'.jpg')
        labels.append(img_root.replace('/train_images/', '/train_masks/') + img[:-4]+'.png') 
        contours.append(img_root.replace('/train_images/', '/train_edges/') + img[:-4] + '.png')
        focal_image.append(img_root.replace('/train_images/', '/train_focal/') + img[:-4]+'.mat')
        
    for img in HFUT_files:
        images.append(HFUT_root + img[:-4]+'.jpg')
        labels.append(HFUT_root.replace('/train_images/', '/train_masks/') + img[:-4]+'.png') 
        contours.append(HFUT_root.replace('/train_images/', '/train_edges/') + img[:-4] + '.png')
        focal_image.append(HFUT_root.replace('/train_images/', '/train_focal/') + img[:-4]+'.mat')
        
    return images, labels, contours, focal_image
                           

def load_test_list(test_path, data_root):
    images, focal = [], []
    img_root = data_root + test_path + '/test_images/'  
    focal_root = data_root + test_path + '/test_focal/'
    img_files = os.listdir(img_root)
    focal_files = os.listdir(focal_root)
    for img in img_files:
        images.append(img_root + img)
    for focal_stacks in focal_files:
        focal.append(focal_root + focal_stacks)
    return images, focal


class ImageData(data.Dataset):
    def __init__(self, dataset_list, data_root, transform, mode, img_size=None, scale_size=None, t_transform=None, label_14_transform=None, label_28_transform=None, label_56_transform=None, label_112_transform=None):

        if mode == 'train':
            self.image_path, self.label_path, self.contour_path, self.focal_path = load_list(dataset_list, data_root)
        else:
            self.image_path, self.focal_path = load_test_list(dataset_list, data_root)

        self.transform = transform
        self.t_transform = t_transform
        self.label_14_transform = label_14_transform
        self.label_28_transform = label_28_transform
        self.label_56_transform = label_56_transform
        self.label_112_transform = label_112_transform
        self.mode = mode
        self.img_size = img_size
        self.scale_size = scale_size
                   
        # focal 
        mean_rgb = np.array([0.447, 0.407, 0.386])
        std_rgb = np.array([0.244, 0.250, 0.253])
        self.focal_mean_focal = np.tile(mean_rgb, 12)  # copy 
        self.focal_std_focal = np.tile(std_rgb, 12)
                           
    def __getitem__(self, item):
        fn = self.image_path[item].split('/')

        filename = fn[-1]

        image = Image.open(self.image_path[item]).convert('RGB')
        image_w, image_h = int(image.size[0]), int(image.size[1])

        if self.mode == 'train':
            label = Image.open(self.label_path[item]).convert('L')
            contour = Image.open(self.contour_path[item]).convert('L')
            
            focal_path = self.focal_path[item]
            focal = self.focal_file_load(focal_path)   # tensor 
     
            random_size = self.scale_size
                       
            new_img = image  
            new_label = label
            new_contour = contour
            new_img = self.transform(new_img)
            new_focal = focal 
            
            label_14 = self.label_14_transform(new_label)
            label_28 = self.label_28_transform(new_label)
            label_56 = self.label_56_transform(new_label)
            label_112 = self.label_112_transform(new_label)
            label_224 = self.t_transform(new_label)

            contour_14 = self.label_14_transform(new_contour)
            contour_28 = self.label_28_transform(new_contour)
            contour_56 = self.label_56_transform(new_contour)
            contour_112 = self.label_112_transform(new_contour)
            contour_224 = self.t_transform(new_contour)

            return new_img, new_focal, label_224, contour_224
                           
        else:
            image = self.transform(image)            
            focal_path = self.focal_path[item]
            focal = self.focal_file_load(focal_path)   
            return image, focal, image_w, image_h, self.focal_path[item]

    def __len__(self):
        return len(self.image_path)

    def focal_file_load(self, focal_path):
        focal = sio.loadmat(focal_path)
        focal = focal['img']
        focal = np.array(focal, dtype=np.int32)
        focal = focal.astype(np.float64)/255.0
        focal -= self.focal_mean_focal
        focal /= self.focal_std_focal
        focal = focal.transpose(2, 0, 1)
        focal = torch.from_numpy(focal).float()  
        return focal

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]       
        image, focal, mask, contour = [list(item) for item in zip(*batch)]

        image = torch.stack(image,dim=0)
        focal = torch.stack(focal,dim=0)
        mask = torch.stack(mask,dim=0)
        contour = torch.stack(contour,dim=0)
        
        image = F.interpolate(image, (size,size), mode='bilinear',align_corners=True)
        focal = F.interpolate(focal, (size,size), mode='bilinear',align_corners=True)
        mask = F.interpolate(mask, (size,size), mode='bilinear',align_corners=True)
        contour = F.interpolate(contour, (size,size), mode='bilinear',align_corners=True)
        return image, focal, mask, contour      

    
    
class vaild_dataset:
    def __init__(self, image_root, focal_root, gt_root):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.focal = [focal_root + f for f in os.listdir(focal_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.mat')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.focal = sorted(self.focal)

        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.focal)
        self.index = 0
        
        # focal 
        mean_rgb = np.array([0.447, 0.407, 0.386])
        std_rgb = np.array([0.244, 0.250, 0.253])
        self.focal_mean_focal = np.tile(mean_rgb, 12)  # copy 
        self.focal_std_focal = np.tile(std_rgb, 12)
            
    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        
        focal = self.focal_file_load(self.focal[self.index]).unsqueeze(0) 
        gt = self.binary_loader(self.gts[self.index])
        
        name = self.focal[self.index].split('/')[-1]

        self.index += 1
        self.index = self.index % self.size

        return image, focal, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')       
        
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        
    def focal_file_load(self, focal_path):
        focal = sio.loadmat(focal_path)
        focal = focal['img']
        focal = np.array(focal, dtype=np.int32)
        focal = focal.astype(np.float64)/255.0
        focal -= self.focal_mean_focal
        focal /= self.focal_std_focal
        focal = focal.transpose(2, 0, 1)
        focal = torch.from_numpy(focal).float()  
        return focal
    
    
def get_loader(dataset_list, data_root, img_size, mode='train'):

    if mode == 'train':
        # RGB
        transform = trans.Compose([
                trans.Scale((256, 256), interpolation=Image.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # label 256 size
        t_transform = trans.Compose([
            trans.Scale((256, 256), interpolation=Image.NEAREST),     # add 
            transforms.ToTensor(),
        ])                
        label_14_transform = trans.Compose([
            trans.Scale((img_size // 16, img_size // 16), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_28_transform = trans.Compose([
            trans.Scale((img_size//8, img_size//8), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_56_transform = trans.Compose([
            trans.Scale((img_size//4, img_size//4), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_112_transform = trans.Compose([
            trans.Scale((img_size//2, img_size//2), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        scale_size = 256
                           
    else:
        transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  
        ])

    if mode == 'train':
        dataset = ImageData(dataset_list, data_root, transform, mode, img_size, scale_size, t_transform, label_14_transform, label_28_transform, label_56_transform, label_112_transform)
    
    else:
        dataset = ImageData(dataset_list, data_root, transform, mode)
    return dataset