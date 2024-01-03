# DIP-Project

## 文件构成！！！

- new里是cx修改后的代码，在lys的代码的基础上增加了数据集扩增（main.py 里的repeat）和集成学习（test.py里的average和vote）。另外还修改了一些细节，比如把dataset_dir改到了args里设置。训练和测试时使用的命令可以参考train.sh和test.sh。

- new2里是cx自己写的代码，不带k-fold的版本，但似乎有什么大病，你要是觉得没用，删了也行

- new2_kfold是cx写的带k-fold的版本，但似乎和它的父亲一样，也有什么大病，你要是觉得没用，删了也行

- new3是cx写的那个两层的模型，不知道是因为有病还是这个模型本来就不行，效果也不好，但我还是把它放上来了，你要是觉得没用，删了也行

## Contents
- [Clone The Repo](#clone-the-repo)
- [Download Dataset](#download-dataset)
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


