# CCTR

## 1. Methods and Results
Table 1， Table 2 and Table 3.

*Note: Only the results from Table 2 are shown here for clarity. Other tables also can using this code to get results.*


## 2. Environment

- [PyTorch](http://pytorch.org/) version >= 1.13.0
- Python version >= 3.7

```shell
# Ubuntu 20.04 Cuda 11.8
conda create -n openocr python==3.8
conda activate cctr
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```



## 3. Model Training / Evaluation

### 3.1 Dataset Preparation

```
python ./dataset_Pre/convert_to_lmdb_format.py
python ./dataset_Pre/mdb_image_to_line.py
```

### 3.2 Training

```shell
# For language model
python ./LLM-2/train.py

# For fusion model
python ./1_prepare_ctc.py # Ocr100 generating original OCR
python ./2_prepare_fusion_data.py # Mask the sample according to the processing result of 1_prepare_ctc.py, and then generate the corresponding LM100
python ./3_normalize_data.py # Normalize ocr100 and LM100 sequences
python ./4_train_fusion_modern.py # Accept the data of 3_normalize_data.py and train the fusion layer
```

### 3.3 Evaluation

```shell
python ./5_test_line.py # 
python ./5_test_page.py # 
python ./tools/eval_rec_all_ch # Test the line level index of openocr
python ./eval _image_level_acc.py # Test the page level index of openocr
```
