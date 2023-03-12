---
layout: post
title: tensorflow dataset에서 불러온 dataset을 tfrecord파일로 변환
subtitle: dataset tfrecord파일로 변환하기
categories: tfrecord
tags: [tfrecord, tensorflow-dataset]
---

```python
!pip install tensorflow-datasets
```

    Requirement already satisfied: tensorflow-datasets in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (4.7.0)
    Requirement already satisfied: absl-py in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from tensorflow-datasets) (1.3.0)
    Requirement already satisfied: six in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from tensorflow-datasets) (1.16.0)
    Requirement already satisfied: promise in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from tensorflow-datasets) (2.3)
    Requirement already satisfied: numpy in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from tensorflow-datasets) (1.23.5)
    Requirement already satisfied: etils[epath] in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from tensorflow-datasets) (0.9.0)
    Requirement already satisfied: requests>=2.19.0 in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from tensorflow-datasets) (2.28.1)
    Requirement already satisfied: tensorflow-metadata in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from tensorflow-datasets) (1.11.0)
    Requirement already satisfied: tqdm in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from tensorflow-datasets) (4.64.1)
    Requirement already satisfied: protobuf>=3.12.2 in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from tensorflow-datasets) (3.19.6)
    Requirement already satisfied: termcolor in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from tensorflow-datasets) (2.1.1)
    Requirement already satisfied: toml in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from tensorflow-datasets) (0.10.2)
    Requirement already satisfied: dill in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from tensorflow-datasets) (0.3.6)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from requests>=2.19.0->tensorflow-datasets) (3.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from requests>=2.19.0->tensorflow-datasets) (1.26.12)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from requests>=2.19.0->tensorflow-datasets) (2022.9.24)
    Requirement already satisfied: charset-normalizer<3,>=2 in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from requests>=2.19.0->tensorflow-datasets) (2.1.1)
    Requirement already satisfied: zipp in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from etils[epath]->tensorflow-datasets) (3.10.0)
    Requirement already satisfied: typing_extensions in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from etils[epath]->tensorflow-datasets) (4.4.0)
    Requirement already satisfied: importlib_resources in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from etils[epath]->tensorflow-datasets) (5.10.0)
    Requirement already satisfied: googleapis-common-protos<2,>=1.52.0 in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from tensorflow-metadata->tensorflow-datasets) (1.57.0)
    Requirement already satisfied: colorama in c:\users\cso13\documents\ai_visual_class\tensorflow_week\venv_tensorflowweek\lib\site-packages (from tqdm->tensorflow-datasets) (0.4.6)
    

    WARNING: You are using pip version 22.0.4; however, version 22.3.1 is available.
    You should consider upgrading via the 'C:\Users\cso13\Documents\AI_visual_class\tensorflow_week\venv_tensorflowweek\Scripts\python.exe -m pip install --upgrade pip' command.
    


```python
from tensorflow.io import serialize_tensor
from tensorflow.io import TFRecordWriter
from tensorflow.train import BytesList
from tensorflow.train import Feature
from tensorflow.train import Features
from tensorflow.train import Example
import tensorflow_datasets as tfds
import tensorflow as tf
import os
```


```python
AUTO = tf.data.AUTOTUNE
DATASET = "div2k/bicubic_x4"

SHARD_SIZE = 256
TRAIN_BATCH_SIZE = 64
INFER_BATCH_SIZE = 8
```


```python
HR_SHAPE = [96, 96, 3]
LR_SHAPE = [24, 24, 3]
SCALING_FACTOR = 4
```


```python
BASE_DATA_PATH = "dataset"
DIV2K_PATH = os.path.join(BASE_DATA_PATH, "div2k")
```


```python
def pre_process(element):
    lrImage = element["lr"]
    hrImage = element["hr"]

    lrByte = serialize_tensor(lrImage)
    hrByte = serialize_tensor(hrImage)

    return (lrByte, hrByte)
```


```python
def create_dataset(dataDir, split, shardsize):
    ds = tfds.load(DATASET, split = split, data_dir = dataDir)
    ds = (
        ds.map(pre_process, num_parallel_calls = AUTO).batch(shardsize)
    )

    return ds
```


```python
print("[INFO] div2k 학습과 테스트 데이터셋을 생성 중...")
trainDs = create_dataset(
    dataDir = DIV2K_PATH,
    split = "train",
    shardsize = SHARD_SIZE
)
```

    [INFO] div2k 학습과 테스트 데이터셋을 생성 중...
    Downloading and preparing dataset Unknown size (download: Unknown size, generated: Unknown size, total: Unknown size) to dataset\div2k\div2k\bicubic_x4\2.0.0...
    EXTRACTING {'train_lr_url': 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip', 'valid_lr_url': 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip', 'train_hr_url': 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip', 'valid_hr_url': 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'}
    


    Dl Completed...: 0 url [00:00, ? url/s]



    Dl Size...: 0 MiB [00:00, ? MiB/s]



    Extraction completed...: 0 file [00:00, ? file/s]



    Generating splits...:   0%|          | 0/2 [00:00<?, ? splits/s]



    Generating train examples...: 0 examples [00:00, ? examples/s]



    Shuffling dataset\div2k\div2k\bicubic_x4\2.0.0.incompleteGHS87H\div2k-train.tfrecord*...:   0%|          | 0/8…



    Generating validation examples...: 0 examples [00:00, ? examples/s]



    Shuffling dataset\div2k\div2k\bicubic_x4\2.0.0.incompleteGHS87H\div2k-validation.tfrecord*...:   0%|          …


    Dataset div2k downloaded and prepared to dataset\div2k\div2k\bicubic_x4\2.0.0. Subsequent calls will reuse this data.
    

