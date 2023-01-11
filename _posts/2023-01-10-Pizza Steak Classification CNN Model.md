---
layout: post
title: Pizza Steak Classification CNN Model
subtitle: Basic CNN Model
categories: AI
tags: [AI, CNN]
---

## Pizza Steak Classification CNN Model

## 이미지 파일을 열고 탐색하기


```python
import os

for dirpath, dirnames, filenames in os.walk("pizza_steak"):
    print(f"{dirpath}에는 {len(dirnames)}개 디렉토리와 {len(filenames)}개 파일이 존재합니다.")
```

    pizza_steak에는 2개 디렉토리와 0개 파일이 존재합니다.
    pizza_steak\test에는 2개 디렉토리와 0개 파일이 존재합니다.
    pizza_steak\test\pizza에는 0개 디렉토리와 250개 파일이 존재합니다.
    pizza_steak\test\steak에는 0개 디렉토리와 250개 파일이 존재합니다.
    pizza_steak\train에는 2개 디렉토리와 0개 파일이 존재합니다.
    pizza_steak\train\pizza에는 0개 디렉토리와 750개 파일이 존재합니다.
    pizza_steak\train\steak에는 0개 디렉토리와 750개 파일이 존재합니다.
    

train data는 750개 test data는 250개

* 이미지 분류의 문제는 라벨링이 필요하다. 이미지를 분류하는 3가지 방법
1. 라벨링에 사용할 라벨로 디렉토리를 생성하고 그 디렉토리에 해당하는 이미지를 넣기
2. 라벨링에 사용하는 라벨의 규칙을 정해서 각 파일이름에 추가한다. 예) pizza_001.jpg, pizza_002.jpg
3. 라벨링 정보를 가진 별도의 파일을 생성해서 목록을 정리한다. 이 목록을 분석해주는 로직이 필요하다.


```python
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
```


```python
# 파일 내부에 있는 이미지를 렌덤으로 확인
def view_random_image(target_dir, target_class):
    target_folder = target_dir + target_class
    
    random_image = random.sample(os.listdir(target_folder), 1)
    
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    
    print(f"Image shape: {img.shape}")
          
    return img
```


```python
img = view_random_image(
    target_dir = "pizza_steak/train/",
    target_class = "steak"
)
```

    Image shape: (512, 512, 3)
    


    
![output_7_1](https://user-images.githubusercontent.com/93850398/211528125-4cd55e49-108d-4fb2-b991-1aac043bc206.png)
    



```python
img.shape # (Width, Height, Color Channel)
```




    (512, 512, 3)



## 이미지 데이터 스케일링

CNN으로 학습을 진행하기 위해 학습 데이터를 0~1 사이의 값으로 스케일링을 한다. 전체 데이터를 255로 나누는 연산을 수행한다.


```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(42)

train_datagen = ImageDataGenerator(rescale = 1 / 255.)
test_datagen = ImageDataGenerator(rescale = 1 / 255.)

train_dir = "pizza_steak/train/"
test_dir = "pizza_steak/test/"

train_data = train_datagen.flow_from_directory(
    train_dir,
    batch_size = 32,
    target_size = (224, 224),
    class_mode = "binary" # 두가지
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    batch_size = 32,
    target_size = (224, 224),
    class_mode = "binary" # 두가지
)
```

    Found 1500 images belonging to 2 classes.
    Found 500 images belonging to 2 classes.
    

## CNN Model_1


```python
tf.random.set_seed(42)

model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters = 10, kernel_size = 3, activation = "relu"),
    tf.keras.layers.Conv2D(filters = 10, kernel_size = 3, activation = "relu"),
    tf.keras.layers.MaxPool2D(pool_size = 2, padding= "valid"),
    tf.keras.layers.Conv2D(filters = 10, kernel_size = 3, activation = "relu"),
    tf.keras.layers.Conv2D(filters = 10, kernel_size = 3, activation = "relu"),
    tf.keras.layers.MaxPool2D(pool_size = 2, padding= "valid"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

model_1.compile(
    loss = "binary_crossentropy",
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ["accuracy"]
)

history_1 = model_1.fit(
    train_data,
    epochs = 5,
    validation_data = test_data
)
```

    Epoch 1/5
    47/47 [==============================] - 48s 989ms/step - loss: 0.6777 - accuracy: 0.6147 - val_loss: 0.5074 - val_accuracy: 0.7480
    Epoch 2/5
    47/47 [==============================] - 43s 917ms/step - loss: 0.5058 - accuracy: 0.7700 - val_loss: 0.4624 - val_accuracy: 0.8120
    Epoch 3/5
    47/47 [==============================] - 42s 889ms/step - loss: 0.4487 - accuracy: 0.7920 - val_loss: 0.4192 - val_accuracy: 0.8360
    Epoch 4/5
    47/47 [==============================] - 62s 1s/step - loss: 0.4350 - accuracy: 0.8000 - val_loss: 0.3975 - val_accuracy: 0.8360
    Epoch 5/5
    47/47 [==============================] - 49s 1s/step - loss: 0.4032 - accuracy: 0.8213 - val_loss: 0.3492 - val_accuracy: 0.8480
    

epoch  : 전체 데이터셋을 1번 학습하는 과정
- 예제의 경우 총 1500장을 학습하는 과정

batch size
- 학습시 나누어서 학습할 이미지 개수
- 예제의 경우 학습할 때 32장씩 쪼개서 학습하겠다!
- 1500 / 32 = 46.875 (32장씩 쪼갠 47개 이미지 모임이 있다.)


```python
model_1.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, None, None, 10)    280       
                                                                     
     conv2d_1 (Conv2D)           (None, None, None, 10)    910       
                                                                     
     max_pooling2d (MaxPooling2D  (None, None, None, 10)   0         
     )                                                               
                                                                     
     conv2d_2 (Conv2D)           (None, None, None, 10)    910       
                                                                     
     conv2d_3 (Conv2D)           (None, None, None, 10)    910       
                                                                     
     max_pooling2d_1 (MaxPooling  (None, None, None, 10)   0         
     2D)                                                             
                                                                     
     flatten (Flatten)           (None, None)              0         
                                                                     
     dense (Dense)               (None, 1)                 28091     
                                                                     
    =================================================================
    Total params: 31,101
    Trainable params: 31,101
    Non-trainable params: 0
    _________________________________________________________________
    

## CNN Model_2


```python
tf.random.set_seed(42)

model_2 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(4, activation = "relu"),
    tf.keras.layers.Dense(4, activation = "relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

model_2.compile(
    loss = "binary_crossentropy",
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ["accuracy"]
)

history_2 = model_2.fit(
    train_data,
    epochs = 5,
    validation_data = test_data
)
```

    Epoch 1/5
    47/47 [==============================] - 9s 175ms/step - loss: 0.6938 - accuracy: 0.4800 - val_loss: 0.6931 - val_accuracy: 0.5000
    Epoch 2/5
    47/47 [==============================] - 8s 179ms/step - loss: 0.6932 - accuracy: 0.4867 - val_loss: 0.6931 - val_accuracy: 0.5000
    Epoch 3/5
    47/47 [==============================] - 8s 175ms/step - loss: 0.6932 - accuracy: 0.4747 - val_loss: 0.6931 - val_accuracy: 0.5000
    Epoch 4/5
    47/47 [==============================] - 8s 181ms/step - loss: 0.6932 - accuracy: 0.5000 - val_loss: 0.6931 - val_accuracy: 0.5000
    Epoch 5/5
    47/47 [==============================] - 8s 178ms/step - loss: 0.6932 - accuracy: 0.4907 - val_loss: 0.6931 - val_accuracy: 0.5000
    


```python
model_2.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_1 (Flatten)         (None, 150528)            0         
                                                                     
     dense_1 (Dense)             (None, 4)                 602116    
                                                                     
     dense_2 (Dense)             (None, 4)                 20        
                                                                     
     dense_3 (Dense)             (None, 1)                 5         
                                                                     
    =================================================================
    Total params: 602,141
    Trainable params: 602,141
    Non-trainable params: 0
    _________________________________________________________________
    

- Trainable params는 모델이 데이터에서 학습할 수 있는 패턴을 말한다.
- model_2 dense (밀집된) 계층은 모든 parameter들이 학습을 해야한다.
- model_1 CNN (Convolutional Neural Network)의 경우에는 입력 데이터에서 가장 중요한 패턴을 학습한다.
- 따라서 model_1은 model_2에 비해 학습을 해야하는 parameter의 개수가 적다.

## CNN Model_3

model_2를 개선 시키기 위해 모델의 complexity(복잡성)을 증가시켜야 한다.
1. 게층을 추가한다.
2. 계층을 이루는 neuron을 더 추가한다.


```python
# 계층을 추가하는 방법
tf.random.set_seed(42)

model_3 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

model_3.compile(
    loss = "binary_crossentropy",
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ["accuracy"]
)

history_3 = model_3.fit(
    train_data,
    epochs = 5,
    validation_data = test_data
)
```

    Epoch 1/5
    47/47 [==============================] - 12s 248ms/step - loss: 2.9693 - accuracy: 0.6153 - val_loss: 0.8831 - val_accuracy: 0.7660
    Epoch 2/5
    47/47 [==============================] - 13s 284ms/step - loss: 1.2462 - accuracy: 0.6893 - val_loss: 0.9601 - val_accuracy: 0.6480
    Epoch 3/5
    47/47 [==============================] - 12s 261ms/step - loss: 0.6415 - accuracy: 0.7573 - val_loss: 0.4422 - val_accuracy: 0.7840
    Epoch 4/5
    47/47 [==============================] - 14s 289ms/step - loss: 0.5245 - accuracy: 0.7653 - val_loss: 0.4559 - val_accuracy: 0.7900
    Epoch 5/5
    47/47 [==============================] - 13s 286ms/step - loss: 0.5399 - accuracy: 0.7573 - val_loss: 0.4137 - val_accuracy: 0.8000
    


```python
model_3.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_2 (Flatten)         (None, 150528)            0         
                                                                     
     dense_4 (Dense)             (None, 100)               15052900  
                                                                     
     dense_5 (Dense)             (None, 100)               10100     
                                                                     
     dense_6 (Dense)             (None, 100)               10100     
                                                                     
     dense_7 (Dense)             (None, 1)                 101       
                                                                     
    =================================================================
    Total params: 15,073,201
    Trainable params: 15,073,201
    Non-trainable params: 0
    _________________________________________________________________
    

model_1 : 30,000 < model_3 : 15,000,000

## 모델을 만드는 순서

1. 데이터 불러오기, 
2. 데이터 전처리, batch에 맞게 준비, 이미지 rescale, resize
3. 모델을 설계
4. 모델을 학습
5. 모델을 평가
6. 모델을 개선 (하이퍼파라미터를 수정)
7. 원하는 결과가 나올때 까지 계속 반복

## CNN Model_4


```python
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential
```


```python
tf.random.set_seed(42)

model_4 = Sequential([
    Conv2D(filters = 10, kernel_size = 3, strides = 1, padding = "valid", activation="relu"),
    Conv2D(filters = 10, kernel_size = 3, strides = 1, padding = "valid", activation="relu"),
    Conv2D(filters = 10, kernel_size = 3, strides = 1, padding = "valid", activation="relu"),
    Flatten(),
    Dense(1, activation = "sigmoid")
])

model_4.compile(
    loss = "binary_crossentropy",
    optimizer = Adam(),
    metrics = ["accuracy"]
)

history_4 = model_4.fit(
    train_data,
    epochs = 5,
    validation_data = test_data
)
```

    Epoch 1/5
    47/47 [==============================] - 47s 993ms/step - loss: 0.7639 - accuracy: 0.6600 - val_loss: 0.4472 - val_accuracy: 0.7860
    Epoch 2/5
    47/47 [==============================] - 71s 2s/step - loss: 0.4602 - accuracy: 0.7960 - val_loss: 0.3995 - val_accuracy: 0.8140
    Epoch 3/5
    47/47 [==============================] - 71s 1s/step - loss: 0.3622 - accuracy: 0.8420 - val_loss: 0.4012 - val_accuracy: 0.8120
    Epoch 4/5
    47/47 [==============================] - 50s 1s/step - loss: 0.2766 - accuracy: 0.8987 - val_loss: 0.4083 - val_accuracy: 0.8140
    Epoch 5/5
    47/47 [==============================] - 53s 1s/step - loss: 0.1569 - accuracy: 0.9520 - val_loss: 0.4748 - val_accuracy: 0.7900
    


```python
model_4.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d_4 (Conv2D)           (None, None, None, 10)    280       
                                                                     
     conv2d_5 (Conv2D)           (None, None, None, 10)    910       
                                                                     
     conv2d_6 (Conv2D)           (None, None, None, 10)    910       
                                                                     
     flatten_3 (Flatten)         (None, None)              0         
                                                                     
     dense_8 (Dense)             (None, 1)                 475241    
                                                                     
    =================================================================
    Total params: 477,341
    Trainable params: 477,341
    Non-trainable params: 0
    _________________________________________________________________
    

model_3 : 15,000,000 > model_4 : 477,000

## 결과값 시각화


```python
import pandas as pd
def plot_loss_curves(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    
    epochs = range(len(history.history["loss"]))
    
    plt.plot(epochs, loss, label = "Train Loss")
    plt.plot(epochs, val_loss, label = "Validataion Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.figure()
    plt.plot(epochs, accuracy, label = "Train Accuracy")
    plt.plot(epochs, val_accuracy, label = "Validataion Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()    
```


```python
plot_loss_curves(history_4)
```


    
![output_34_0](https://user-images.githubusercontent.com/93850398/211528602-e4736c12-aaed-45b1-9bdb-0e5bd134c3a8.png)
    



    
![output_34_1](https://user-images.githubusercontent.com/93850398/211528646-5bed12c4-2487-4610-a3ae-f708a9ffa876.png)
    


## 모델이 학습하는 과정

1. 시작을 할 수 있는 모델을 설계
2. 모델을 학습하고 평가
3. 오버피팅(과적합)일 경우 과적합을 줄이도록 한다(parameter를 조정하는 과정)
    - convolution 계층을 추가한다
    - convolution 계층의 필터의 갯수를 추가한다
    - dense을 추가해본다.

* 일반적인 CNN 구조
    - 입력 (input) -> Conv + ReLU (non-linearities) -> Pooling -> Fully connected (Dense) (출력 / output)
    - model_4에서는 pooling을 사용하지 않았다
    - 입력 (input) -> Conv + ReLU (non-linearities) -> Max Pooling -> Fully connected (Dense) (출력 / output)

## CNN Model_5


```python
tf.random.set_seed(42)

model_5 = Sequential([
    Conv2D(filters = 10, kernel_size = 3, strides = 1, padding = "valid", activation="relu"),
    MaxPool2D(pool_size = 2),
    Conv2D(filters = 10, kernel_size = 3, strides = 1, padding = "valid", activation="relu"),
    MaxPool2D(pool_size = 2),
    Conv2D(filters = 10, kernel_size = 3, strides = 1, padding = "valid", activation="relu"),
    MaxPool2D(pool_size = 2),
    Flatten(),
    Dense(1, activation = "sigmoid")
])

model_5.compile(
    loss = "binary_crossentropy",
    optimizer = Adam(),
    metrics = ["accuracy"]
)

history_5 = model_5.fit(
    train_data,
    epochs = 5,
    validation_data = test_data
)
```

    Epoch 1/5
    47/47 [==============================] - 24s 494ms/step - loss: 0.6128 - accuracy: 0.6573 - val_loss: 0.4882 - val_accuracy: 0.7600
    Epoch 2/5
    47/47 [==============================] - 23s 490ms/step - loss: 0.4679 - accuracy: 0.7867 - val_loss: 0.4231 - val_accuracy: 0.8020
    Epoch 3/5
    47/47 [==============================] - 25s 521ms/step - loss: 0.4209 - accuracy: 0.8080 - val_loss: 0.3312 - val_accuracy: 0.8840
    Epoch 4/5
    47/47 [==============================] - 27s 579ms/step - loss: 0.3951 - accuracy: 0.8247 - val_loss: 0.3255 - val_accuracy: 0.8620
    Epoch 5/5
    47/47 [==============================] - 38s 812ms/step - loss: 0.3872 - accuracy: 0.8340 - val_loss: 0.3179 - val_accuracy: 0.8860
    


```python
model_5.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d_7 (Conv2D)           (None, None, None, 10)    280       
                                                                     
     max_pooling2d_2 (MaxPooling  (None, None, None, 10)   0         
     2D)                                                             
                                                                     
     conv2d_8 (Conv2D)           (None, None, None, 10)    910       
                                                                     
     max_pooling2d_3 (MaxPooling  (None, None, None, 10)   0         
     2D)                                                             
                                                                     
     conv2d_9 (Conv2D)           (None, None, None, 10)    910       
                                                                     
     max_pooling2d_4 (MaxPooling  (None, None, None, 10)   0         
     2D)                                                             
                                                                     
     flatten_4 (Flatten)         (None, None)              0         
                                                                     
     dense_9 (Dense)             (None, 1)                 6761      
                                                                     
    =================================================================
    Total params: 8,861
    Trainable params: 8,861
    Non-trainable params: 0
    _________________________________________________________________
    


```python
plot_loss_curves(history_5)
```


    
![output_41_0](https://user-images.githubusercontent.com/93850398/211528740-fd5e2024-6ed6-445d-b299-1f28b889c98e.png)
    



    
![output_41_1](https://user-images.githubusercontent.com/93850398/211528810-a339f2f8-543a-492c-ae19-04d77d286c5f.png)
    


- Conv2D 계층에서 filter 10개가 찾은 feature들을 찾아낸다. 
- MaxPooling2D: 가장 중요한 feature들만 선택하고, 나머지는 버립니다.
- pool_size가 커질수록 버려지는(무시되는) feature들이 많아집니다. 기본 설정은(2, 2)이며 4개 값중에서 가장 높은 값을 선택합니다.

## 이미지 증강

- data augmentaion : 학습 데이터를 변형하는 과정
    - 1개 이미지에 대해서 좀더 다양한 이미지를 얻을 수 있다.
    - 모델은 좀더 많은 데이터 학습을 하면서 일반적인 특징을 찾아간다.
    - overfitting (과적합)을 방지할 수 있다.


```python
train_datagen_augmented = ImageDataGenerator(
    rescale = 1 / 255.,
    rotation_range= 20,
    shear_range = 0.2,
    zoom_range = 0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True
)

train_data_augmented = train_datagen_augmented.flow_from_directory(
    train_dir,
    batch_size = 32,
    target_size = (224, 224),
    class_mode = "binary",
    shuffle = False
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    batch_size = 32,
    target_size = (224, 224),
    class_mode = "binary",
    shuffle = False
)
```

    Found 1500 images belonging to 2 classes.
    Found 1500 images belonging to 2 classes.
    

## 증강된 이미지 확인


```python
images, labels = train_data.next() # 32장씩 준비된 47개 덩어리에서 1개만 추출
```


```python
augmented_images, augmented_labels = train_data_augmented.next() # 32장씩 준비된 47개 덩어리에서 1개만 추출
```


```python
random_number = random.randint(0, 32)
plt.imshow(images[random_number])
plt.title("Original Image")
plt.axis(False),
plt.figure()
plt.imshow(augmented_images[random_number])
plt.title("Augmented Image")
plt.axis(False)
```




    (-0.5, 223.5, 223.5, -0.5)




    
![output_49_1](https://user-images.githubusercontent.com/93850398/211528870-d5310e35-2f21-4d5e-9b95-616aeeb074fc.png)
    



    
![output_49_2](https://user-images.githubusercontent.com/93850398/211528928-87224880-1110-4ff0-8bc1-f4e2240ce09d.png)
    


## CNN Model_6


```python
tf.random.set_seed(42)

model_6 = Sequential([
    Conv2D(filters = 10, kernel_size = 3, strides = 1, padding = "valid", activation="relu"),
    MaxPool2D(pool_size = 2),
    Conv2D(filters = 10, kernel_size = 3, strides = 1, padding = "valid", activation="relu"),
    MaxPool2D(pool_size = 2),
    Conv2D(filters = 10, kernel_size = 3, strides = 1, padding = "valid", activation="relu"),
    MaxPool2D(pool_size = 2),
    Flatten(),
    Dense(1, activation = "sigmoid")
])

model_6.compile(
    loss = "binary_crossentropy",
    optimizer = Adam(),
    metrics = ["accuracy"]
)

history_6 = model_6.fit(
    train_data_augmented,
    epochs = 5,
    validation_data = test_data
)
```

    Epoch 1/5
    47/47 [==============================] - 49s 1s/step - loss: 0.7375 - accuracy: 0.5313 - val_loss: 0.6767 - val_accuracy: 0.5040
    Epoch 2/5
    47/47 [==============================] - 54s 1s/step - loss: 0.6799 - accuracy: 0.6080 - val_loss: 0.6486 - val_accuracy: 0.6360
    Epoch 3/5
    47/47 [==============================] - 76s 2s/step - loss: 0.6666 - accuracy: 0.6087 - val_loss: 0.6463 - val_accuracy: 0.5640
    Epoch 4/5
    47/47 [==============================] - 51s 1s/step - loss: 0.6395 - accuracy: 0.6527 - val_loss: 0.5699 - val_accuracy: 0.6820
    Epoch 5/5
    47/47 [==============================] - 47s 1s/step - loss: 0.6185 - accuracy: 0.6773 - val_loss: 0.5636 - val_accuracy: 0.7180
    


```python
plot_loss_curves(history_6)
```


    
![output_52_0](https://user-images.githubusercontent.com/93850398/211529017-06cdf593-61c9-4c9b-a7e6-2a5cd84c5e15.png)
    



    
![output_52_1](https://user-images.githubusercontent.com/93850398/211529058-da482169-89bd-4d62-9575-f7a8c290e542.png)
    


## CNN Model_7


```python
train_data_augmented = train_datagen_augmented.flow_from_directory(
    train_dir,
    batch_size = 32,
    target_size = (224, 224),
    class_mode = "binary",
    shuffle = True
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    batch_size = 32,
    target_size = (224, 224),
    class_mode = "binary",
    shuffle = True
)
```

    Found 1500 images belonging to 2 classes.
    Found 1500 images belonging to 2 classes.
    


```python
tf.random.set_seed(42)

model_7 = Sequential([
    Conv2D(filters = 10, kernel_size = 3, strides = 1, padding = "valid", activation="relu"),
    MaxPool2D(pool_size = 2),
    Conv2D(filters = 10, kernel_size = 3, strides = 1, padding = "valid", activation="relu"),
    MaxPool2D(pool_size = 2),
    Conv2D(filters = 10, kernel_size = 3, strides = 1, padding = "valid", activation="relu"),
    MaxPool2D(pool_size = 2),
    Flatten(),
    Dense(1, activation = "sigmoid")
])

model_7.compile(
    loss = "binary_crossentropy",
    optimizer = Adam(),
    metrics = ["accuracy"]
)

history_7 = model_7.fit(
    train_data_augmented,
    epochs = 5,
    validation_data = test_data
)
```

    Epoch 1/5
    47/47 [==============================] - 44s 917ms/step - loss: 0.6727 - accuracy: 0.5673 - val_loss: 0.5770 - val_accuracy: 0.6980
    Epoch 2/5
    47/47 [==============================] - 47s 1s/step - loss: 0.5828 - accuracy: 0.7167 - val_loss: 0.4234 - val_accuracy: 0.8160
    Epoch 3/5
    47/47 [==============================] - 65s 1s/step - loss: 0.4979 - accuracy: 0.7647 - val_loss: 0.3576 - val_accuracy: 0.8420
    Epoch 4/5
    47/47 [==============================] - 46s 966ms/step - loss: 0.4857 - accuracy: 0.7667 - val_loss: 0.3412 - val_accuracy: 0.8720
    Epoch 5/5
    47/47 [==============================] - 42s 886ms/step - loss: 0.4687 - accuracy: 0.7800 - val_loss: 0.3417 - val_accuracy: 0.8720
    


```python
plot_loss_curves(history_7)
```


    
![output_56_0](https://user-images.githubusercontent.com/93850398/211529149-920b3406-c363-4cc0-9dca-cdd4ab72e469.png)
    



    
![output_56_1](https://user-images.githubusercontent.com/93850398/211529183-ba91d588-9978-42a7-a428-116db5e3e2ff.png)
    


## 이미지 예측하기


```python
def load_and_prep_image(filename, image_shape = 224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels= 3)
    img = tf.image.resize(img, size = [image_shape, image_shape])
    img = img / 255.
    return img
```


```python
steak = load_and_prep_image("03-steak.jpeg")
```

이미지를 스케일링하고 모델에 적용시키는 이미지의 차원과 예측을 할 때 모델에 적용시키는 이미지는 기존의 차원에 하나를 더해야 한다
- 기존의 이미지 스케일링 (244, 224, 3)
- 예측시 이미지 스케일링(None, 224, 224, 3) => (batch_size , 224, 224, 3)


```python
print(f"shape before new dimension: {steak.shape}")

steak = tf.expand_dims(steak, axis = 0)
print(f"shape after new dimension: {steak.shape}")
```

    shape before new dimension: (224, 224, 3)
    shape after new dimension: (1, 224, 224, 3)
    


```python
model_7.predict(steak)
```

    1/1 [==============================] - 0s 146ms/step
    
    array([[0.92639554]], dtype=float32)




```python
pred = model_7.predict(steak)
pred
```

    1/1 [==============================] - 0s 44ms/step
    
    array([[0.92639554]], dtype=float32)




```python
# 결과값 출력
class_names = ["pizza", "steak"]
class_names
int(tf.round(pred[0][0]))
class_names[int(tf.round(pred[0][0]))]
```




    'steak'




```python
def pred_and_plot(model, filename, class_name):
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis = 0))
    pred_class = class_names[ int(tf.round(pred)[0][0]) ]
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False);
```


```python
pred_and_plot(model_7, "03-pizza-dad.jpeg", class_names)
```

    1/1 [==============================] - 0s 39ms/step
    


    
![output_66_1](https://user-images.githubusercontent.com/93850398/211529249-0e20fe7a-d56f-40f5-b62e-334b3d43509f.png)
    

