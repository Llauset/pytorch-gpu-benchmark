# Benchmarck IA 
This benchmarck allows to obtain the learning and inference speed for varius CNN models in __pytorch__  

## Benchmarck description

This code "benchmark_models.py" performs training and inference speed of ResNet50, ResNet101, ResNet152, wide_resnet101_2, wide_resnet50_2, DenseNet121 and DenseNet201 with batch size 12

Three types of the datatype are used. single-precision, double-precision, half-precision
The code must not be changed.  

###  requirement
* python>=3.6(for f-formatting)
* torchvision
* torch>=1.0.0
* pandas
* psutil
* matplotlib

## Usage

`python benchmark_models.py -g <Nb_GPUs>`

sbatch script example for Olympe is provided "sbatch_example.sh"

## Results

Average results will appear on screen and will also be stored on folder "results". 
".csv" files will be created for each datatype and each number of GPUs used.

## Requested information

Once the execution has finished a plot showing the timing results can be created using the script plot.py:

`python plot.py`

Two plots will be generated, one for training results and another for inference results.
The plot will present the average execution time (ms) for each model depending on the datatype and the number of GPUs.

All the timing information has to be included. Other plot types can be provided.

Total energy consumption per gpu configuration is also required.
For example if two gpus are used we asked for the energy consumed during the entire execution of:
`python benchmark_models.py -g 2`

## Reference examples

In "olympe_plots_results" folder you can see as example the Olympe training results (olympe_train.png)
and the Olympe inference results (olympe_inference.png)

