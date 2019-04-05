import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN_RL, BUIFD
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("--color_mode", type=str, default='gray', help='Grayscale (gray) or color (color) model')
parser.add_argument("--model", type=str, default='R', help='Model chosen for testing: R (DnCNN) or F (BUIFD)')
parser.add_argument("--max_train_noise", type=int, default=55, help="Max noise level seen during the training")

parser.add_argument("--varying_noise", type=bool, default=False, help="Set to True if varying noise is used")

parser.add_argument("--epoch", type=int, default=49, help="Epoch used for testing")
parser.add_argument("--num_layers", type=int, default=20, help="Number of res blocks used in the model")

opt = parser.parse_args()

def normalize(data):
    return data/255.

def create_varying_noise(image_size, noise_std_min, noise_std_max):
    ''' outputs a noise image of size image_size, with varying noise levels, ranging from noise_std_min to noise_std_max
    the noise level increases linearly with the number of rows in the image '''
    noise = torch.FloatTensor(image_size).normal_(mean=0, std=0).cuda()

    row_size = torch.Size([image_size[0], image_size[1], image_size[2]])
    for row in range(image_size[3]):
        std_value = noise_std_min + (noise_std_max-noise_std_min) * (row/(image_size[3]*1.0-1))
        noise[:,:,:,row] = torch.FloatTensor(row_size).normal_(mean=0, std=std_value/255.).cuda()

    return noise


def inference(test_data, model, varying_noise=False, color_mode='gray'):
    files_source = glob.glob(os.path.join('testing_data', test_data, '*.png'))

    files_source.sort()
    # process data
    img_idx = 0
    std_values = list(range(5,76,5))
    psnr_results = np.zeros((len(files_source), len(std_values)))
    
    for f in files_source:
        # image
        if color_mode == 'color':
            Img = cv2.imread(f)
            Img = normalize(np.float32(Img[:,:,:]))
            Img = np.rollaxis(Img, axis=2, start=0)
            Img = np.expand_dims(Img, 0)
            ISource = torch.Tensor(Img)
            ISource = Variable(ISource.cuda())
        else:
            Img = cv2.imread(f)
            Img = normalize(np.float32(Img[:,:,0]))
            Img = np.expand_dims(Img, 0)
            Img = np.expand_dims(Img, 1)
            ISource = torch.Tensor(Img)
            ISource = Variable(ISource.cuda())

        for noise_idx, noise_std in enumerate(std_values):
            # create noise
            if varying_noise:
                if noise_std not in [15,25,40,55,65]:
                    continue
                noise = create_varying_noise(ISource.size(), noise_std - 10, noise_std + 10)
            else:
                noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=noise_std/255.).cuda()

            # create noisy images
            INoisy = ISource + noise
            INoisy = Variable(INoisy.cuda())

            
            INoisy = torch.clamp(INoisy, 0., 1.)


            # feed forward then clamp image
            with torch.no_grad():
                if  opt.model == 'R':
                    NoiseNetwork = model(INoisy)
                    INetwork_notclamped = INoisy - NoiseNetwork
                    INetwork = torch.clamp(INetwork_notclamped, 0., 1.)
                    psnr_results[img_idx, noise_idx] = batch_PSNR(INetwork, ISource, 1.)

                elif opt.model == 'F':
                    NoiseNetwork, NoiseLevelNetwork = model(INoisy)
                    
                    INetwork_notclamped = INoisy - NoiseNetwork
                    INetwork = torch.clamp(INetwork_notclamped, 0., 1.)
                    psnr_results[img_idx, noise_idx] = batch_PSNR(INetwork, ISource, 1.)

                    # NoiseLevelGT = INetwork * 0 + noise_std/75.0
                    # std_psnr = batch_PSNR(NoiseLevelNetwork, NoiseLevelGT, 1.)

        img_idx += 1
        
    return psnr_results



def main():
    
    model_name = opt.color_mode + ('_BUIFD_' if opt.model == 'F' else '_DnCNN_') + str(opt.max_train_noise)

    log_dir = os.path.join('Logs', model_name)
    model_dir = os.path.join('saved_models', model_name)

    print('Testing with model %s at epoch %d, with %s' %(model_name, opt.epoch, 'varying noise...' if opt.varying_noise else 'uniform noise...'))

    num_of_layers = opt.num_layers

    if opt.color_mode == 'color':
        model_channels = 3
    else:
        model_channels = 1

    if opt.model == 'R':
        net = DnCNN_RL(channels=model_channels, num_of_layers=num_of_layers)
    elif opt.model == 'F':
        net = BUIFD(channels=model_channels, num_of_layers=num_of_layers)

    model = nn.DataParallel(net).cuda()
    model.load_state_dict(torch.load(os.path.join(model_dir, 'net_%d.pth' % (opt.epoch))))
    model.eval()


    if opt.color_mode == 'color':
        test_data = 'CBSD68'
    else:
        test_data = 'BSD68'


    psnr_results = inference(test_data, model, opt.varying_noise, opt.color_mode)

    for std_idx, std in enumerate(range(5,76,5)):
        print( 'Average PSNR result for level %d: %.2fdB' %(std, np.mean(psnr_results[:,std_idx])) )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if opt.model == 'F':
        np.save( os.path.join( log_dir, ('varN_' if opt.varying_noise else '')  + 'PSNR_' + str(opt.epoch)), psnr_results )
    else:
        np.save( os.path.join( log_dir, ('varN_' if opt.varying_noise else '') + 'PSNR_' + str(opt.epoch)), psnr_results )
    
    print('Results saved inside Logs!')


if __name__ == "__main__":
    main()