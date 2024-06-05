# CPL-
This repository is the official Pytorch implementation for extension of the ICCV2023 paper **Confidence-aware Pseudo-label Learning for Weakly Supervised Visual Grounding**.

## Contents

1. [Usage](#usage)
2. [Results](#results)
3. [Contacts](#contacts)
4. [Acknowledgments](#acknowledgments)

## Usage

### Dependencies
- Python 3.9.10
- PyTorch 1.9.0 + cu111 + cp39
- [Pytorch-Bert 0.6.2](https://pypi.org/project/pytorch-pretrained-bert/)
- Check [requirements.txt](requirements.txt) for other dependencies. 

### Data Preparation
1.You can download the images from the original source and place them in `./data/image_data` folder:
- RefCOCO and ReferItGame
- [Flickr30K Entities](http://shannon.cs.illinois.edu/DenotationGraph/#:~:text=make%20face-,Downloads,-Please%20fill%20in)

Finally, the `./data/` and `./image_data/` folder will have the following structure:

```angular2html
|-- data
      |-- flickr
      |-- gref
      |-- gref_umd
      |-- referit
      |-- unc
      |-- unc+
|-- image_data
   |-- Flickr30k
      |-- flickr30k-images
   |-- other
      |-- images
   |-- referit
      |-- images
```
- ```./data/```: Take the Flickr30K dataset as an example, ./data/flickr/ shoud contain files about the dataset's train/validation/test annotations and our generated pseudo-samples and proposals pool for this dataset. You can download these file from [data](https://disk.pku.edu.cn/link/AA381C5DF7ACBE4A4DA1E203A70EBB6556) and put them on the corresponding folder.
- ```./image_data/Flickr30k/flickr30k-images/```: Image data for the Flickr30K dataset, please download from this [link](http://shannon.cs.illinois.edu/DenotationGraph/#:~:text=make%20face-,Downloads,-Please%20fill%20in). Fill the form and download the images.
- ```./image_data/other/images/```: Image data for RefCOCO/RefCOCO+/RefCOCOg. 
- ```./image_data/referit/images/```: Image data for ReferItGame.
### Pretrained Checkpoints
1.You can download the checkpoints （trained by CPL model) from [Google Drive](https://drive.google.com/file/d/19IhMNEgGIl4qGPq7v0SsD8VZSucfmlXj/view?usp=drive_link). These checkpoints should be downloaded and move to the checkpoints directory.

```
mv cpl_checkpoints.tar.gz ./checkpoints/
tar -zxvf cpl_checkpoints.tar.gz
```
### Training and Evaluation
You should refine the pseudo label by running the [generate_target.py](generate_target.py)

```
CUDA_VISIBLE_DEVICES=0 python generate_target.py --dataset unc --eval_model ./checkpoints/unc_best_checkpoint.pth --eval_set uni_modal
```

1.  Training on RefCOCO. 
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 28888 --use_env train.py --num_workers 8 --epochs 20 --batch_size 32 --lr 0.0001 --lr_bert 0.00001 --lr_visu_cnn 0.00001 --lr_visu_tra 0.00001 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --data_root ./data/image_data --split_root ./data/ --output_dir ./outputs/unc/
    ```
    *Notably, if you use a smaller batch size, you should also use a smaller learning rate. Original learning rate is set for batch size 128(4GPU x 32).* 
    Please refer to [scripts/train.sh](scripts/train.sh) for training commands on other datasets. 

2.  Evaluation on RefCOCO.
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --data_root ./data/image_data --split_root ./data/ --eval_model ./checkpoints/unc_best_checkpoint.pth --eval_set testA --output_dir ./outputs/unc/testA/;
    ```
    Please refer to [scripts/eval.sh](scripts/eval.sh) for evaluation commands on other splits or datasets.
