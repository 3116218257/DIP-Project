# DIP-Project

## Directory Structure
- ```DRAC2022_TaskB``` code used for our backbone selection experiments
- ```new``` code used for our advanced experiments on ensemble learning, dataset amplification, pseudo-labeling, etc.
- ```new2``` code with no k-fold
- ```new2_kfold``` code with k-fold
- ```new3``` prototye code(a new structure with 2 levels of resnet)
- ```README.md```
- ```environment.yml```
- 
## Contents
- [Clone The Repo](#clone-the-repo)
- [Download Dataset](#download-dataset)
- [Environment](#environment)
- [Usage](#usage)
- 
## Clone The Repo
Using the following command line can clone this repo into your machine<br>
```bash 
git clone https://github.com/3116218257/DIP-Project.git
cd DIP-Project
```
Then create an empty directory for the dataset.
```bash
mkdir data
cd data
```

## Download Dataset
You need to download the Task B dataset, you can get it from https://data.mendeley.com/datasets/s8kbw25s3x/1. If you have downloaded the zipped file, use this command.
```bash
unzip 'B. Image Quality Assessment.zip'
rm -rf 'B. Image Quality Assessment.zip'
```

## Environment
Just create a virtual environment for our project using command line<br>
```bash
conda env create -f environment.yml
```
If there still are some missing package, download manually the packages in the ```environment.yml```.

## Usage
There are two sections, the ```DARC2022_TaskB``` directory contains code for our backbone selection experiments, the ```new``` directory contains code for our advanced experiments, you can play with both of them.

### Reminder
Please change the dataset directory in ```DRAC2022_TaskB/dataset.py``` and ```DRAC2022_TaskB/train.py``` before executing.

### Backbone Selection
You can choose different backbones in this section. Please refer to the configuration settings in ```train.py``` and change the ```args.sh``` file to use a new backbone.<br>
Then simply run the train.py using command<br>
```bash
sh args.sh.
```
Finally run ```test.py``` to get your needed ```.csv``` file.<br>
```bash
python test.py
```

### Advanced Experiments
The series of ```new``` directories contains different version of code that we used throughout our whole experiment process, we suggest you use the code in ```new```, which we used in our final experiments. The process of execution is similar with the operations above, use ```train.sh``` and ```test.sh``` to get the final ```.csv``` file. For configuration settings, please refer to ```config.py```.


