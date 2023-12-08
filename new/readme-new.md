# ReadMe

- `environment.yml`：conda配置环境
  conda env create -f environment.yml

- 注意修改`train.py`中dataset路径
- python train.py --batch_size 32 --lr 0.0001 --weight_decay 0.0001 --device 1 --num_epochs 50 --model convnext --model_path ./saved_models

