import os
import torch
import Training
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=True, type=bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33111', type=str, help='init_method')
    
    parser.add_argument('--data_root', default='/dataset/SOD/Dataset_LightField/', type=str, help='data path')
    parser.add_argument('--trainset', default='TrainSet_DUTLF_HFUT', type=str, help='Trainging set')
    
    parser.add_argument('--train_steps', default=50*12100, type=int, help='total training steps')    
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--pretrained_model', default='./resnet34-333f7ec4.pth', type=str, help='load Pretrained model')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
    parser.add_argument('--stepvalue1', default=200000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=400000, type=int, help='the step 2 for adjusting lr')
    
    parser.add_argument('--model_name', default='IRNet_Eval', type=str, help='save model path')
    parser.add_argument('--save_model_dir', default='/model/checkpoint/', type=str, help='save model path')
    parser.add_argument('--save_log_dir', default='/userhome/IRNet/', type=str, help='save log path')
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    if not os.path.exists(args.save_log_dir):
        os.makedirs(args.save_log_dir)
        
    num_gpus = torch.cuda.device_count()
    if args.Training:
        Training.train_net(num_gpus=num_gpus, args=args)
