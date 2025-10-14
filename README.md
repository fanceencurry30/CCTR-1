# SVTRv2

## 1. Methods and Results

### 1.1 Result 1

| Method       | RCTW | ReCTS | LSVT | ArT  | CTW  | Web  | HW   | Arg  |
| ------------ | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- |
| CRNN+VLFM    | 2.1  | 4.9   | 1.4  | 3.1  | 3.8  | 2.2  | 8.0  | 3.64 |
| LISTER+VLFM  | 38.0 | 39.5  | 46.8 | 42.4 | 48.3 | 45.0 | 48.6 | 44.1 |
| SMTR+VLFM    | 25.4 | 21.3  | 20.6 | 19.6 | 46.7 | 35.2 | 41.2 | 30.0 |
| SVTR v2+VLFM | 1.1  | 0.9   | 1.3  | 2.3  | 2.6  | 1.4  | 2.9  | 1.8  |

### 1.2 Result 2

| Method        | Scene | Web  | Doc  | HW   | Arg   | $L_{>25}$ | Params(M) |
| ------------- | ----- | ---- | ---- | ---- | ----- | --------- | --------- |
| SVTR v2(2025) | 80.0  | 82.3 | 99.5 | 81.6 | 83.31 | 52.8      | 22.5      |
| CCTR          | 80.0  | 82.3 | 99.5 | 81.6 | 83.31 | 52.8      | 59.7      |



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

```shell
# First stage
python CTC_LISTER.py   # for svtr,crnn,lister生成batch大小的CTCjson，包括top-100 概率及对应索引
python SMTR.py		   # for smtr

# Second stage
python project_lister.py
python project_svtr.py #读取前面生成的 JSON 文件（包含 CTC top-100 概率和解码文本），结合语言模型 (LM) 的概率，生成新的训练数据（OCR 概率 + LM 概率 + GT 概率）
python project_crnn.py
python project_smtr.py

# Third stage
python prepare_data.py #读取多个 OCR+LM+GT 概率分布 JSON 文件。按照一定规则筛选/平衡样本。把所有数据转换成 PyTorch 张量，并保存成 .pt 文件
                         #判断 OCR top1 是否等于 GT top1 → 分成匹配/不匹配。 选择样本（全样本 or 平衡采样）
```

### 3.2 Training

```shell
# For language model
python train.py

# For vision language attention module
python train_eval_fusion60p_1.py
```

### 3.3 Evaluation

```shell
python test_last_f.py
```