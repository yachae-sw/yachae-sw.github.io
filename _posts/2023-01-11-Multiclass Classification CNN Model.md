---
layout: post
title: Multiclass Classification CNN Model
subtitle: Multiclass CNN Model
categories: AI
tags: [AI, CNN]
---

## Multiclass Classification CNN Model

## 이미지 파일을 열고 파일 탐색하기


```python
import os

for dirpath, dirnames, filenames in os.walk("10_food_classes_all_data"):
    print(f"{dirpath}에는 {len(dirnames)}개 디렉토리와 {len(filenames)}개 파일이 존재합니다.")
```

    10_food_classes_all_data에는 2개 디렉토리와 0개 파일이 존재합니다.
    10_food_classes_all_data\test에는 10개 디렉토리와 0개 파일이 존재합니다.
    10_food_classes_all_data\test\chicken_curry에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_all_data\test\chicken_wings에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_all_data\test\fried_rice에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_all_data\test\grilled_salmon에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_all_data\test\hamburger에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_all_data\test\ice_cream에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_all_data\test\pizza에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_all_data\test\ramen에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_all_data\test\steak에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_all_data\test\sushi에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_all_data\train에는 10개 디렉토리와 0개 파일이 존재합니다.
    10_food_classes_all_data\train\chicken_curry에는 0개 디렉토리와 750개 파일이 존재합니다.
    10_food_classes_all_data\train\chicken_wings에는 0개 디렉토리와 750개 파일이 존재합니다.
    10_food_classes_all_data\train\fried_rice에는 0개 디렉토리와 750개 파일이 존재합니다.
    10_food_classes_all_data\train\grilled_salmon에는 0개 디렉토리와 750개 파일이 존재합니다.
    10_food_classes_all_data\train\hamburger에는 0개 디렉토리와 750개 파일이 존재합니다.
    10_food_classes_all_data\train\ice_cream에는 0개 디렉토리와 750개 파일이 존재합니다.
    10_food_classes_all_data\train\pizza에는 0개 디렉토리와 750개 파일이 존재합니다.
    10_food_classes_all_data\train\ramen에는 0개 디렉토리와 750개 파일이 존재합니다.
    10_food_classes_all_data\train\steak에는 0개 디렉토리와 750개 파일이 존재합니다.
    10_food_classes_all_data\train\sushi에는 0개 디렉토리와 750개 파일이 존재합니다.
    


```python
train_dir = "10_food_classes_all_data/train/"
test_dir = "10_food_classes_all_data/test/"
```

## 분류해야 할 데이터명 추출


```python
import pathlib

data_dir = pathlib.Path(train_dir)

for item in data_dir.glob("*"):
    print(item.name)
```

    chicken_curry
    chicken_wings
    fried_rice
    grilled_salmon
    hamburger
    ice_cream
    pizza
    ramen
    steak
    sushi
    


```python
import pathlib

data_dir = pathlib.Path(train_dir)

class_names = sorted([item.name for item in data_dir.glob("*")])
print(class_names)
```

    ['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 'hamburger', 'ice_cream', 'pizza', 'ramen', 'steak', 'sushi']
    


```python
len(class_names)
```




    10




```python
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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

img = view_random_image(
    target_dir = "10_food_classes_all_data/train/",
    target_class = class_names[3]
)
```

    Image shape: (512, 512, 3)
    


    
![output_9_1](https://user-images.githubusercontent.com/93850398/211586338-884d1702-5eb3-46ae-aeff-e4cdc37fabe5.png)
    


## 이미지 스케일링 및 증강


```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```


```python
train_datagen = ImageDataGenerator(rescale = 1/255.)
test_datagen = ImageDataGenerator(rescale = 1/255.)
```


```python
train_data = train_datagen.flow_from_directory(
    train_dir,
    batch_size = 32,
    target_size = (224, 224),
    class_mode = "categorical" # 여러가지
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    batch_size = 32,
    target_size = (224, 224),
    class_mode = "categorical" # 여러가지
)
```

    Found 7500 images belonging to 10 classes.
    Found 2500 images belonging to 10 classes.
    

## CNN Model_1(multiclass)


```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential
```


```python
tf.random.set_seed(42)

model_1 = Sequential([
    Conv2D(filters = 10, kernel_size = 3, activation = "relu"),
    Conv2D(filters = 10, kernel_size = 3, activation = "relu"),
    MaxPool2D(pool_size = 2, padding= "valid"),
    Conv2D(filters = 10, kernel_size = 3, activation = "relu"),
    Conv2D(filters = 10, kernel_size = 3, activation = "relu"),
    MaxPool2D(pool_size = 2, padding= "valid"),
    Flatten(),
    Dense(10, activation = "softmax")
])

model_1.compile(
    loss = "categorical_crossentropy",
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
    235/235 [==============================] - 199s 841ms/step - loss: 2.0863 - accuracy: 0.2472 - val_loss: 1.9573 - val_accuracy: 0.3116
    Epoch 2/5
    235/235 [==============================] - 194s 823ms/step - loss: 1.8214 - accuracy: 0.3797 - val_loss: 1.8581 - val_accuracy: 0.3492
    Epoch 3/5
    235/235 [==============================] - 200s 850ms/step - loss: 1.4284 - accuracy: 0.5264 - val_loss: 2.0026 - val_accuracy: 0.3340
    Epoch 4/5
    235/235 [==============================] - 202s 859ms/step - loss: 0.7769 - accuracy: 0.7447 - val_loss: 2.6571 - val_accuracy: 0.2680
    Epoch 5/5
    235/235 [==============================] - 194s 827ms/step - loss: 0.3035 - accuracy: 0.9049 - val_loss: 3.7968 - val_accuracy: 0.2572
    


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
                                                                     
     dense (Dense)               (None, 10)                280910    
                                                                     
    =================================================================
    Total params: 283,920
    Trainable params: 283,920
    Non-trainable params: 0
    _________________________________________________________________
    

현재 multiclass classification하는 모델은 과적합 (overfitting)상태이다.

## CNN Model_2(multiclass)

- 과적합을 극복하는 방법 = regularization (preventing overfitting)
    1. 더 많은 데이터를 학습시킨다.
    2. 모델을 간결하게 한다. 너무 집중해서 패턴을 확인하는 것이 아니라, 넓은 시각에서 패턴을 찾아본다.
    3. 데이터에 augmentation를 적용시킨다.
    4. 전이학습 (transfer learning)


```python
# 모델을 간결하게 하는 방법

tf.random.set_seed(42)

model_2 = Sequential([
    Conv2D(filters = 10, kernel_size = 3, activation = "relu"),
    MaxPool2D(pool_size = 2, padding= "valid"),
    Conv2D(filters = 10, kernel_size = 3, activation = "relu"),
    MaxPool2D(pool_size = 2, padding= "valid"),
    Flatten(),
    Dense(10, activation = "softmax")
])

model_2.compile(
    loss = "categorical_crossentropy",
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
    235/235 [==============================] - 78s 331ms/step - loss: 2.2196 - accuracy: 0.1829 - val_loss: 2.0497 - val_accuracy: 0.2660
    Epoch 2/5
    235/235 [==============================] - 81s 344ms/step - loss: 1.8771 - accuracy: 0.3553 - val_loss: 1.9564 - val_accuracy: 0.3144
    Epoch 3/5
    235/235 [==============================] - 88s 376ms/step - loss: 1.4557 - accuracy: 0.5201 - val_loss: 2.0933 - val_accuracy: 0.2960
    Epoch 4/5
    235/235 [==============================] - 80s 339ms/step - loss: 0.9041 - accuracy: 0.7172 - val_loss: 2.4456 - val_accuracy: 0.2896
    Epoch 5/5
    235/235 [==============================] - 105s 445ms/step - loss: 0.4292 - accuracy: 0.8853 - val_loss: 3.0912 - val_accuracy: 0.2800
    

## 결과값 시각화


```python
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
plot_loss_curves(history_2)
```


    
![output_24_0](https://user-images.githubusercontent.com/93850398/211586432-77f27cf4-45fe-4fc9-bfb3-01830a578014.png)
    


    
![output_24_1](https://user-images.githubusercontent.com/93850398/211586487-eacecf8e-15e7-4da3-b68a-7e8375972e46.png)
    


## CNN Model_3(multiclass)


```python
# 데이터에 augmentation을 적용

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
    class_mode = "categorical",
)
```

    Found 7500 images belonging to 10 classes.
    


```python
tf.random.set_seed(42)

model_3 = Sequential([
    Conv2D(filters = 10, kernel_size = 3, activation = "relu"),
    MaxPool2D(pool_size = 2, padding= "valid"),
    Conv2D(filters = 10, kernel_size = 3, activation = "relu"),
    MaxPool2D(pool_size = 2, padding= "valid"),
    Flatten(),
    Dense(10, activation = "softmax")
])

model_3.compile(
    loss = "categorical_crossentropy",
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ["accuracy"]
)

history_3 = model_3.fit(
    train_data_augmented,
    epochs = 5,
    validation_data = test_data
)
```

    Epoch 1/5
    235/235 [==============================] - 129s 547ms/step - loss: 2.1821 - accuracy: 0.2128 - val_loss: 1.9890 - val_accuracy: 0.3016
    Epoch 2/5
    235/235 [==============================] - 148s 630ms/step - loss: 2.0404 - accuracy: 0.2751 - val_loss: 1.8773 - val_accuracy: 0.3328
    Epoch 3/5
    235/235 [==============================] - 135s 576ms/step - loss: 1.9881 - accuracy: 0.3040 - val_loss: 1.8512 - val_accuracy: 0.3412
    Epoch 4/5
    235/235 [==============================] - 137s 583ms/step - loss: 1.9538 - accuracy: 0.3197 - val_loss: 1.8759 - val_accuracy: 0.3516
    Epoch 5/5
    235/235 [==============================] - 159s 677ms/step - loss: 1.9328 - accuracy: 0.3291 - val_loss: 1.8023 - val_accuracy: 0.3848
    


```python
plot_loss_curves(history_3)
```


    
![output_28_0](https://user-images.githubusercontent.com/93850398/211586553-94893825-1612-43bf-9ad3-034f26138123.png)
    



    
![output_28_1](https://user-images.githubusercontent.com/93850398/211586614-5bf793ad-fb2d-463a-b23d-87830aca70b9.png)
    


## CNN Model_4(multiclass)


```python
# LearningRate를 조절하는 방법

tf.random.set_seed(42)

model_4 = Sequential([
    Conv2D(filters = 10, kernel_size = 3, activation = "relu"),
    MaxPool2D(pool_size = 2, padding= "valid"),
    Conv2D(filters = 10, kernel_size = 3, activation = "relu"),
    MaxPool2D(pool_size = 2, padding= "valid"),
    Flatten(),
    Dense(10, activation = "softmax")
])

model_4.compile(
    loss = "categorical_crossentropy",
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ["accuracy"]
)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-4 * 10**(epoch/20) # 1e-4(0.0001)부터 시작해서 매 epoch당 10**(epoch/20)만큼 증가시킨다.
)

history_4 = model_4.fit(
    train_data_augmented,
    epochs = 20,
    validation_data = test_data,
    callbacks = [ lr_scheduler ]
)
```

    Epoch 1/20
    235/235 [==============================] - 139s 588ms/step - loss: 2.2190 - accuracy: 0.1764 - val_loss: 2.0985 - val_accuracy: 0.2524 - lr: 1.0000e-04
    Epoch 2/20
    235/235 [==============================] - 141s 600ms/step - loss: 2.1044 - accuracy: 0.2423 - val_loss: 2.0192 - val_accuracy: 0.2860 - lr: 1.1220e-04
    Epoch 3/20
    235/235 [==============================] - 149s 636ms/step - loss: 2.0450 - accuracy: 0.2752 - val_loss: 1.9611 - val_accuracy: 0.3088 - lr: 1.2589e-04
    Epoch 4/20
    235/235 [==============================] - 137s 581ms/step - loss: 2.0143 - accuracy: 0.2851 - val_loss: 1.9101 - val_accuracy: 0.3288 - lr: 1.4125e-04
    Epoch 5/20
    235/235 [==============================] - 139s 591ms/step - loss: 1.9875 - accuracy: 0.3041 - val_loss: 1.8815 - val_accuracy: 0.3448 - lr: 1.5849e-04
    Epoch 6/20
    235/235 [==============================] - 141s 598ms/step - loss: 1.9730 - accuracy: 0.3064 - val_loss: 1.9014 - val_accuracy: 0.3408 - lr: 1.7783e-04
    Epoch 7/20
    235/235 [==============================] - 135s 575ms/step - loss: 1.9497 - accuracy: 0.3272 - val_loss: 1.8479 - val_accuracy: 0.3464 - lr: 1.9953e-04
    Epoch 8/20
    235/235 [==============================] - 139s 591ms/step - loss: 1.9431 - accuracy: 0.3288 - val_loss: 1.8787 - val_accuracy: 0.3504 - lr: 2.2387e-04
    Epoch 9/20
    235/235 [==============================] - 154s 657ms/step - loss: 1.9243 - accuracy: 0.3345 - val_loss: 1.8583 - val_accuracy: 0.3580 - lr: 2.5119e-04
    Epoch 10/20
    235/235 [==============================] - 159s 676ms/step - loss: 1.9101 - accuracy: 0.3368 - val_loss: 1.7931 - val_accuracy: 0.3840 - lr: 2.8184e-04
    Epoch 11/20
    235/235 [==============================] - 158s 673ms/step - loss: 1.9040 - accuracy: 0.3381 - val_loss: 1.7667 - val_accuracy: 0.3944 - lr: 3.1623e-04
    Epoch 12/20
    235/235 [==============================] - 159s 678ms/step - loss: 1.8978 - accuracy: 0.3413 - val_loss: 1.7906 - val_accuracy: 0.3692 - lr: 3.5481e-04
    Epoch 13/20
    235/235 [==============================] - 152s 648ms/step - loss: 1.8818 - accuracy: 0.3509 - val_loss: 1.7495 - val_accuracy: 0.3956 - lr: 3.9811e-04
    Epoch 14/20
    235/235 [==============================] - 150s 639ms/step - loss: 1.8804 - accuracy: 0.3571 - val_loss: 1.8262 - val_accuracy: 0.3728 - lr: 4.4668e-04
    Epoch 15/20
    235/235 [==============================] - 154s 655ms/step - loss: 1.8654 - accuracy: 0.3668 - val_loss: 1.7696 - val_accuracy: 0.4056 - lr: 5.0119e-04
    Epoch 16/20
    235/235 [==============================] - 138s 588ms/step - loss: 1.8701 - accuracy: 0.3640 - val_loss: 1.7208 - val_accuracy: 0.4248 - lr: 5.6234e-04
    Epoch 17/20
    235/235 [==============================] - 142s 606ms/step - loss: 1.8685 - accuracy: 0.3641 - val_loss: 1.8101 - val_accuracy: 0.3888 - lr: 6.3096e-04
    Epoch 18/20
    235/235 [==============================] - 160s 680ms/step - loss: 1.8514 - accuracy: 0.3685 - val_loss: 1.6968 - val_accuracy: 0.4208 - lr: 7.0795e-04
    Epoch 19/20
    235/235 [==============================] - 155s 658ms/step - loss: 1.8613 - accuracy: 0.3644 - val_loss: 1.7285 - val_accuracy: 0.4032 - lr: 7.9433e-04
    Epoch 20/20
    235/235 [==============================] - 146s 622ms/step - loss: 1.8655 - accuracy: 0.3668 - val_loss: 1.7722 - val_accuracy: 0.4048 - lr: 8.9125e-04
    


```python
import pandas as pd

pd.DataFrame(history_4.history).plot(xlabel = "epochs")
plt.title("Model_4_training_curves");
```


    
![output_31_0](https://user-images.githubusercontent.com/93850398/211586688-459ee5cf-1bb0-4409-ac84-93a5b18121ca.png)
    



```python
import numpy as np

lrs = 1e-4 * (10**(np.arange(20)/20))
```


```python
plt.semilogx(lrs, history_4.history["loss"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs. Loss")
```




    Text(0.5, 1.0, 'Learning rate vs. Loss')




    
![output_33_1](https://user-images.githubusercontent.com/93850398/211586731-fd0e731c-feb4-4d1d-823a-cc9a4817aace.png)
    


Learning Rate의 값이 올라갈수록 Loss값이 작아진다
