# DIP-Project

## Directory Structure
- ```DRAC2022_TaskB``` basic model selection part
- ```new``` added with some improving methods 
- ```new2``` code with no k-fold
- ```new2_kfold``` code with k-fold
- ```new3``` prototye code(do not use)
- ```README.md```
- ```environment.yml```
## Contents
- [Clone The Repo](#clone-the-repo)
- [Download Dataset](#download-dataset)
- [Environment](#environment)
- [Usage](#usage)
## Clone The Repo
Using the following command line can clone this repo in to your machine<br>
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
You need to download the Task B dataset, you get get it from https://data.mendeley.com/datasets/s8kbw25s3x/1 , if you downloaded the zipped file, use this command
```bash
unzip 'B. Image Quality Assessment.zip'
rm -rf 'B. Image Quality Assessment.zip'
```

## Environment
Just create a virtual environment for our project using command line<br>
```bash
conda env create -f environment.yml
```
If there still some missing package, just follow the version in the ```environment.yml```

## Usage
There are two sections, the ```DARC2022_TaskB``` directory contains our model selection works, the ```new``` directory contains our improving method with specific model, you can choose any of them.
### Reminder
Please change the dataset root in ```dataset.py``` and ```train.py```!

### Multi Model Choose
You can choose different model in this section. If you want to choose different model, just read the ```train.py``` and change the ```args.sh``` file to adjust a new model.<br>
Then simply run the train.py using command<br>
```bash
sh args.sh.
```
Finally run ```test.py``` to get your needed ```.csv``` file.<br>
```bash
python test.py
```

### Improving Method
The series of ```new``` directory contains our different version of code, we suggest to use the code in ```new```, the process of execution is similar with the operation above, use ```train.sh``` and ```test.sh``` to get the final ```.csv``` file.


