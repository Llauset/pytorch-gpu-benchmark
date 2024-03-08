# Benchmarck IA CALMIP
This benchmarck allows to obtain the learning and inference speed for varius CNN models in __pytorch__  

## Benchmarck description

This code performs training and inference speed of ResNet50, ResNet101, ResNet152, wide_resnet101_2, wide_resnet50_2, DenseNet121 and DenseNet201 with batch size 12

Three types of the datatype are used. single-precision, double-precision, half-precision

###  requirement
* python>=3.6(for f-formatting)
* torchvision
* torch>=1.0.0
* pandas
* psutil
* matplotlib

## Usage

`python benchmark_models.py -g <Nb_GPUs>`

## Results

Average results will appear on screen and will also be stored on folder "results". 
".csv" files will be created for each datatype and each number of GPUs used.

## Plot results

Once the execution has finished a plot can be created:

`python plot.py`

Two plots will be generated one for training results and another for inference results.
The plot will present the average execution time (ms) for each model depending on the 
datatype and the number of GPUs.

## Reference examples

Olympe training:

![Olympe training](https://github.com/calmip/pytorch-gpu-benchmark/blob/main/fig/olympe_train.png)

Olympe inference:

![Olympe inference](https://github.com/calmip/pytorch-gpu-benchmark/blob/main/fig/olympe_inference.png)

# What is asked

Energy for ...
