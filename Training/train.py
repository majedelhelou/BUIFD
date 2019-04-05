import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from models import DnCNN_RL, BUIFD
from dataset import prepare_data, Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--weight_exponent", type=float, default=0, help="Exponent of noise level loss weight")
parser.add_argument("--color", type=int, default=0, help='1 for a color network, 0 for grayscale')

parser.add_argument("--net_mode", type=str, default="R", help='with DnCNN_RL (R) or BUIFD (F)')
parser.add_argument("--noise_max", type=float, default=75, help="Max training noise level")
opt = parser.parse_args()


def main():

    # Load dataset
    print('Loading dataset ...\n')
    ## F ##
    if opt.net_mode == 'F':
        if opt.color == 1:
            dataset_train = Dataset(train=True, aug_times=2, grayscale=False, scales=True)
        else:
            dataset_train = Dataset(train=True, aug_times=1, grayscale=True)

    ## R ##
    if opt.net_mode == 'R':
        if opt.color == 1:
            dataset_train = Dataset(train=True, aug_times=2, grayscale=False, scales=True)
        else:
            dataset_train = Dataset(train=True, aug_times=2, grayscale=True, scales=True)


    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))


    # Build model
    model_channels = 1 + 2 * opt.color

    if opt.net_mode == 'R':
        print('** Creating RL network: **')
        net = DnCNN_RL(channels=model_channels, num_of_layers=opt.num_of_layers)
    elif opt.net_mode == 'F':
        print('** Creating Fusion network: **')
        net = BUIFD(channels=model_channels, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    
    pretrained_dncnn = False
    if opt.net_mode == 'F' and pretrained_dncnn:
        print('***** PRETRAINED DnCNN in F *****')
        subnet = DnCNN_RL(channels=model_channels, num_of_layers=opt.num_of_layers)
        subnet = nn.DataParallel(subnet).cuda()
        subnet.load_state_dict(torch.load(os.path.join('/scratch/elhelou/real_denoise/saved_models/final_models/model%d_R_20_2_scales' %opt.noise_max, 'net_49.pth')))
        subnet = subnet.module
        for param in subnet.dncnn.dncnn.parameters():
            param.requires_grad = False
        net.dncnn.dncnn = subnet.dncnn.dncnn

    # Loss
    criterion = nn.MSELoss(size_average=False)

    # Move to GPU
    model = nn.DataParallel(net).cuda()
    criterion.cuda() # print(model)
    print('Trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # Training
    noiseL_B = [0,opt.noise_max]

    train_loss_log = np.zeros(opt.epochs)
    
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            # factor = epoch // opt.milestone
            current_lr = opt.lr / (10.)
        # Learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('\nlearning rate %f' % current_lr)

        # Train
        for i, data in enumerate(loader_train, 0):
            # Training
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            
            # ADD Noise
            img_train = data

            noise = torch.zeros(img_train.size())
            noise_level_train = torch.zeros(img_train.size())
            stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
            sizeN = noise[0,:,:,:].size()
            # Noise Level map preparation (each step)
            for n in range(noise.size()[0]):
                noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)

                noise_level_value = stdN[n] / noiseL_B[1]
                noise_level_train[n,:,:,:] = torch.FloatTensor( np.ones(sizeN) )
                noise_level_train[n,:,:,:] = noise_level_train[n,:,:,:] * noise_level_value
            noise_level_train = Variable(noise_level_train.cuda())


            imgn_train = img_train + noise


            # Training step
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            
            if opt.net_mode == 'R':
                out_train = model(imgn_train)
                loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
                loss_weight = -99
            elif opt.net_mode == 'F':
                out_train, out_noise_level_train = model(imgn_train)
                loss_img = criterion(out_train, noise) / (imgn_train.size()[0]*2)
                loss_noise_level = criterion(out_noise_level_train, noise_level_train) / (imgn_train.size()[0]*2)
                
                # Multi-components
                loss_weight = 10 ** (- opt.weight_exponent)
                loss = loss_img + loss_weight*loss_noise_level

            loss.backward()
            optimizer.step()
            

            # Results
            model.eval()
            if opt.net_mode == 'R':
                out_train = model(imgn_train)
            elif opt.net_mode == 'F':
                out_train, out_noise_level_train = model(imgn_train)
            
            img_out_train = imgn_train - out_train
            img_out_train = torch.clamp(img_out_train, 0., 1.)
            psnr_train = batch_PSNR(img_out_train, img_train, 1.)
            

            
            if opt.net_mode == 'R':
                train_loss_log[epoch] += loss.item()
            elif opt.net_mode == 'F':
                train_loss_log[epoch] += loss_img.item()

        train_loss_log[epoch] = train_loss_log[epoch] / len(loader_train)



        print('Epoch %d: loss=%.4f' %(epoch,train_loss_log[epoch]))

        if opt.color == 0:
            model_name = 'gray_%s_%d' % ('DnCNN' if opt.net_mode == 'R' else 'BUIFD', noiseL_B[1])
        else:
            model_name = 'color_%s_%d' % ('DnCNN' if opt.net_mode == 'R' else 'BUIFD', noiseL_B[1])


        model_dir = os.path.join('saved_models', model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), os.path.join(model_dir, 'net_%d.pth' % (epoch)) )
        
        

if __name__ == "__main__":
    if opt.preprocess:
        if opt.color == 0:
            grayscale = True
            stride = 10
            if opt.net_mode == 'F':
                prepare_data(data_path='training_data', patch_size=50, stride=stride, aug_times=1, grayscale=grayscale)
            else:
                prepare_data(data_path='training_data', patch_size=50, stride=stride, aug_times=2, grayscale=grayscale, scales_bool=True)
        else:
            stride = 25
            grayscale = False
            prepare_data(data_path='training_data', patch_size=50, stride=stride, aug_times=2, grayscale=grayscale, scales_bool=True)

    main()