# DIP-Project
## Contents
- [Clone the repo](#heading-one)
- [Download dataset](#heading-two)
- [Usage](#heading-three)
## Clone the repo
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

## Download dataset
You need to download the Task B dataset, you get get it from https://data.mendeley.com/datasets/s8kbw25s3x/1 , if you downloaded the zipped file, use this command
```bash
unzip 'B. Image Quality Assessment.zip'
rm -rf 'B. Image Quality Assessment.zip'
```

## Usage
There are two sections, the ```DARC2022_TaskB``` directory contains our model selection works, the ```new``` directory contains our improving method with specific model, you can choose any of them.
### Reminder
Please change the dataset root in ```dataset.py``` and ```train.py```!

### Multi model choose
You can choose different model in this section. If you want to choose different model, just read the ```train.py``` and change the ```args.sh``` file to adjust a new model.<br>
Then simply run the train.py using command<br>
```bash
sh args.sh.
```
Finally run test.py.


