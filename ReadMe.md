# Stroke Extraction Net

A method of stroke extraction for Chinese characters.<br>
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
The RHSEDB is available at [Google Drive](https://drive.google.com/file/d/1Fdj0Yht_ywvnlZJhLYrQzbuTBVRQvDNO/view?usp=share_link). 
Download the data and unzip it to the *dataset* directory.

## Training
Run 'main_train.py' to train the whole stroke extraction model. Or run 'train_ExtractNet.py', 
'train_SDNet.py' and 'train_SegNet.py' selectively to train a single module. 

## Requirements
    pytorch=1.9  
    python=3.8

## Citation
If you use this repo in your research, please cite us as follows:

    

