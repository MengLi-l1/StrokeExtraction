1.Our method for stroke extraction of Chinese character main consist of three modules.
Therefore, the code is built around three modules.

2.You can execute 'main_train.py' and select dataset to train the whole stroke extraction model.

3.Limited by the size of uploaded files, we can only provide part of the datasets for training.

4.The details of these files are as follows:

FILE DIRECTORY:
  --char_recognise:  Chinese character recognition model and parameters that is used in SDNet.
  --content_net_model: ContentNet and parameters that is trained to auto-encode stroke images for content-loss.
  --dataset:
    --CCSEDB: dataset of calligraphy characters
    --RHSEDB: dataset of handwriting characters
  --model
    --model_of_ExtractNet.py
    --model_of_SDNet.py
    --model_of_SegNet.py
  --load_data_for_SDNet.py
  --load_data_for_SegNetExtractNet.py
  --main_train.py: Train the whole stroke extraction model
  --train_ExtractNet.py:
  --train_SDNet.py
  --train_SegNet.py
  --utils.py: Some common functions used in training
  --utils_loss_val.py: Loss functions and qualitative evaluation functions used in training and validating.

