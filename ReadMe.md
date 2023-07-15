# Stroke Extraction Net

A method of stroke extraction for Chinese characters. [Paper(AAAI 2023 oral).](https://ojs.aaai.org/index.php/AAAI/article/view/25220)<br>
We propose a deep learning-based character stroke extraction method that takes semantic features and prior information of strokes into consideration. 
Our method mainly consists of three modules. Therefore, the code is built around three modules.

## Dataset 
**The Regular Handwriting Character Stroke Extraction Dataset (RHSEDB):** 
We construct RHSEDB referring to (Wang, Jiang, and Liu 2022) based on the online
handwriting dataset CASIA-OLHWDB (Liu et al. 2011), 
which contains 28,080 pieces of handwriting data written
by 17 writers in total. Each record of data in RHSEDB contains a target image and 
some single stroke label images of the target image arranged in reference stroke order. 
In these images, writing track of the stroke is normalized to a width of 6
pixels (the size of the stroke image is 256 pixels).  
The RHSEDB is available at [Google Drive](https://drive.google.com/file/d/1akn5zMDhwNkYl3iExiizXKlPweCc10-_/view?usp=drive_link). 
Download the data and unzip, the path looks like:``./dataset/RHSEDB``.

## Training
The **VGG** model parameters of char recognise can be downloaded from [VGG-model](https://drive.google.com/file/d/1UgE1iYv4r6sPsjMRb84ACCCLe5nYZtTb/view?usp=drive_link). The path looks like:``./char_recognise/out_vgg_bn/model/model.pth``.<br>
The **ContentNet** model parameters can be downloaded from [ContentNet-model](https://drive.google.com/file/d/1R2h-jDhv2pBHVEeBvFUfLH2jQ7qBCuXl/view?usp=drive_link). The path looks like:``./content_net_model/out/model_content.pth``.


Run 'main_train.py' to train the whole stroke extraction model. Or run 'train_ExtractNet.py', 
'train_SDNet.py' and 'train_SegNet.py' selectively to train a single module. 

## Requirements
    pytorch=1.9  
    python=3.8

## Citation
If you use this repo in your research, please consider citing this work using this BibTex entry:
```
@inproceedings{li2023stroke,
  title={Stroke Extraction of Chinese Character Based on Deep Structure Deformable Image Registration},
  author={Li, Meng and Yu, Yahan and Yang, Yi and Ren, Guanghao and Wang, Jian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={1},
  pages={1360--1367},
  year={2023}
}
```
    

