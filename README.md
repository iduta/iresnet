## Improved Residual Networks

This is a PyTorch implementation of [iResNet paper](https://arxiv.org/abs/2004.04989):
```
@article{duta2020improved,
  author  = {Ionut Cosmin Duta and Li Liu and Fan Zhu and Ling Shao},
  title   = {Improved Residual Networks for Image and Video Recognition},
  journal = {arXiv preprint arXiv:2004.04989},
  year    = {2020},
}
```


The models trained on ImageNet can be found [here](https://drive.google.com/open?id=1t9IbIm5VV5NnXhKyw15FUt8SmmvhhLHU).


The iResNet (improved residual network) is able to improve the baseline (ResNet) 
in terms of recognition performance without increasing the number of parameters
and computational costs. The iResNet is very effective in training very deep models 
(see [the paper](https://arxiv.org/pdf/2004.04989.pdf) for details).

The accuracy on ImageNet (using the default training settings):


| Network | 50-layers |101-layers |152-layers |200-layers |
| :-----: | :-----: | :-----: |:-----: |:-----: |
| ResNet  | 76.12% ([model](https://drive.google.com/open?id=1yqp8Z6qp03ZKToACTJHLtynDBUToRLrU)) | 78.00% ([model](https://drive.google.com/open?id=13_OnBf7qJnFFBMrDZXdox7kmhMmxCXAG)) | 78.45% ([model](https://drive.google.com/open?id=1BsYmoAVJxumH4yWKH-DcJ_YDk__3ArQT))| 77.55% ([model](https://drive.google.com/open?id=1n4turCIswvNdWoRq2imZn1Ump-2giwKa))
| iResnet  | **77.31**% ([model](https://drive.google.com/open?id=1Waw3ob8KPXCY9iCLdAD6RUA0nvVguc6K))| **78.64**% ([model](https://drive.google.com/open?id=1cZ4XhwZfUOm_o0WZvenknHIqgeqkY34y))| **79.34**% ([model](https://drive.google.com/open?id=10heFLYX7VNlaSrDy4SZbdOOV9xwzwyli))| **79.48**% ([model](https://drive.google.com/open?id=1Ao-f--jNU7MYPqSW8UMonXVrq3mkLRpW))



### Requirements

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

A fast alternative (without the need to install PyTorch and other deep learning libraries) is to use [NVIDIA-Docker](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/pullcontainer.html#pullcontainer), 
we used [this container image](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_19-05.html#rel_19-05).


### Training
To train a model (for instance, iResNet with 50 layers) using DataParallel run `main.py`; 
you need also to provide `result_path` (the directory path where to save the results
 and logs) and the `--data` (the path to the ImageNet dataset): 
```bash
result_path=/your/path/to/save/results/and/logs/
mkdir -p ${result_path}
python main.py \
--data /your/path/to/ImageNet/dataset/ \
--result_path ${result_path} \
--arch iresnet \
--model_depth 50
```
To train using Multi-processing Distributed Data Parallel Training follow the instructions in the 
[official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

