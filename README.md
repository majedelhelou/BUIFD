# BUIFD

This repository contains the neeeded files to reproduce the denoising results with the pretrained networks, and to retrain and test any of the models, of the paper:

**[Blind Universal Bayesian Image Denoising with Gaussian Noise Level Learning](https://arxiv.org/abs/1907.03029)**

Contact author: [Majed El Helou](http://majedelhelou.github.io)

****
The PyTorch implementation is based on that of the paper *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*. The repository of that implementation can be found [*on this link*](https://github.com/SaoYan/DnCNN-PyTorch).
****

### 1. Dependencies
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision)
* OpenCV for Python
* [HDF5 for Python](http://www.h5py.org/)

### 2. Reproduce results with pretrained models
The folder 'Inference_Pretrained' is sufficient on its own to re-run the paper's experiments and to visualize results.
It contains all 8 models used in the paper, under 'Pretrained_Models', and the testing data under 'Data'.
To re-run all denoising experiments you can run:
```
bash inference_calls
```
or select a single experiment, for instance:
```
python test.py --color_mode gray --model R --max_train_noise 55 --epoch 49 --varying_noise True
```
which runs grayscale image denoising of the BSD68 test set with the DnCNN model (R for regular, F for BUIFD) that was trained up to noise level 55, and trained for 50 epochs. The test is carried out with varying noise in this case, with all the noise levels of the paper.
Results are saved in 'Logs' (already available in this repository too), and can be visualized by running:
```
jupyter notebook visualize_PSNR_results.ipynb
```
**Note:** the notebook assumes all experimental results are already generated, which is the case when you download the repository.

### 3. Re-train networks
The folder 'Training' is sufficient on its own to re-train any of the network models. It contains the 'training_data' and 'testing_data', the models are saved for each epoch in a subdirectory inside 'saved_models' and test results are saved in 'Logs'. Average PSNR test results per noise level are also printed at the end of training.
To re-train all 8 models with default settings, you can run:
```
bash example_train_test
```
Or you can train custom models individually:
```
python train.py --net_mode F --noise_max 55 --color 1 --preprocess True
```
which trains a BUIFD model with maximum training noise level 55. Then evaluate it on the CBSD68 test set:
```
python test.py --color_mode color --model F --max_train_noise 55 --epoch 49
```
**Note:** You can set 'preprocess' to False (default setting) if it is *not* the first time you train a certain model, and the hdf5 file is already generated and saved in the code directory. Otherwise it needs to be set to True.


### 4. BM3D comparison
For comparisons with BM3D and CBM3D denoising, the authors' code can be found [*on this link*](http://www.cs.tut.fi/~foi/GCF-BM3D/).



