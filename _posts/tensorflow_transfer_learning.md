```python
# Trasfer Learning (전이학습)
# 지금 우리는 직접 Convolutional Neural Network를 설계하면서 모델을 학습시켰습니다.
# 어느 정도는 학습을 한 것 같지만, 아직도 모델을 배울 것이 많다... 모델을 개선 시켜야 한다!!!
# 모델을 개선을 시키기 위해서
# 1) 하이퍼 파라미터 수정
# 2) 모델의 계층을 추가하거나, 계층의 neuron을 추가한다. => model complexity를 높인다.
# 3) learning rate 변경
# 4) 데이터를 더 많이 주던가...
# 5) 더 많이 학습을 시키던가...
# 1 ~ 5 의 작업들은 반복을 하면서 결과를 확인해서 조정해야 한다. => 시간이 많이 필요하다!

# 그래서! 우리는 전이학습을 사용한다.
# 전이학습이란! 특정한 다른 데이터셋에서 특징 (feature)을 잘 배운 즉, 패턴을 잘 인식하는 모델을
# 우리가 가져와서 사용한다! => 우리의 데이터셋으로 그 모델을 학습시킨다.
# 사전에 학습된 모델 (pretrained model)은 이미 학습이 잘 되어 있고, 특정한 문제에 패턴을 잘 찾는다!
# 패턴을 잘 찾는 모델에 내 문제를 내 데이터셋을 학습시켜본다!

# 전이학습의 장점
# 1) 우리가 해결하려는 문제에 대해서 이미 잘 해결한다고 입증된 신경망 가져와서 사용한다. 시간 절약!
# 2) 이미 잘 문제를 해결하는 모델이기도 하고, 사전에 학습이 되어 있기에, 적은 량의 데이터로도 좋은 성능을 얻을 수 있다!
```


```python
import zipfile

zip_ref = zipfile.ZipFile("10_food_classes_10_percent.zip", "r")
zip_ref.extractall()
zip_ref.close()
```


```python
import os

for dirpath, dirnames, filenames in os.walk("10_food_classes_10_percent"):
    print(f"{dirpath}에는 {len(dirnames)}개 디렉토리와 {len(filenames)}개 파일이 존재합니다.")
```

    10_food_classes_10_percent에는 2개 디렉토리와 0개 파일이 존재합니다.
    10_food_classes_10_percent\test에는 10개 디렉토리와 0개 파일이 존재합니다.
    10_food_classes_10_percent\test\chicken_curry에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\chicken_wings에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\fried_rice에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\grilled_salmon에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\hamburger에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\ice_cream에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\pizza에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\ramen에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\steak에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\sushi에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\train에는 10개 디렉토리와 0개 파일이 존재합니다.
    10_food_classes_10_percent\train\chicken_curry에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\chicken_wings에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\fried_rice에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\grilled_salmon에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\hamburger에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\ice_cream에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\pizza에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\ramen에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\steak에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\sushi에는 0개 디렉토리와 75개 파일이 존재합니다.
    


```python
train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"
```


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
    target_class = "chicken_curry"
)
```

    Image shape: (512, 512, 3)
    


    
![png](output_7_1.png)
    



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
    class_mode = "categorical"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    batch_size = 32,
    target_size = (224, 224),
    class_mode = "categorical"
)
```

    Found 750 images belonging to 10 classes.
    Found 2500 images belonging to 10 classes.
    


```python
# 1) ResNetV2 : 2016년부터 많이 사용되는 모델
# 2) EfficientNet : 2019년부터 많이 사용되는 모델
```


```python
RESNET_URL = "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5"
```


```python
# !pip install tensorflow-hub
```


```python
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
```


```python
feature_extractor_layer = hub.KerasLayer(
    RESNET_URL,
    trainable = False,
    name = "my_feature_extraction_layer",
    input_shape = (224, 224, 3)
)
```

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.data_structures has been moved to tensorflow.python.trackable.data_structures. The old module will be deleted in version 2.11.
    


```python
model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(10, activation = "softmax", name = "my_output_layer")
])
```

    WARNING:tensorflow:From C:\Users\fermat39\tensorflow_week\venv_tensorflowweek\lib\site-packages\tensorflow\python\autograph\pyct\static_analysis\liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.
    Instructions for updating:
    Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089
    

    WARNING:tensorflow:From C:\Users\fermat39\tensorflow_week\venv_tensorflowweek\lib\site-packages\tensorflow\python\autograph\pyct\static_analysis\liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.
    Instructions for updating:
    Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089
    


```python
train_data.num_classes
```




    10




```python
model.compile(
    loss = "categorical_crossentropy",
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ["accuracy"]
)
```


```python
resnet_history = model.fit(
    train_data,
    epochs = 5,
    validation_data = test_data
)
```

    Epoch 1/5
    24/24 [==============================] - 666s 28s/step - loss: 2.0515 - accuracy: 0.3213 - val_loss: 1.1629 - val_accuracy: 0.6276
    Epoch 2/5
    24/24 [==============================] - 666s 29s/step - loss: 0.8591 - accuracy: 0.7507 - val_loss: 0.7853 - val_accuracy: 0.7516
    Epoch 3/5
    24/24 [==============================] - 619s 27s/step - loss: 0.5776 - accuracy: 0.8227 - val_loss: 0.6891 - val_accuracy: 0.7780
    Epoch 4/5
    24/24 [==============================] - 597s 25s/step - loss: 0.4274 - accuracy: 0.8933 - val_loss: 0.6686 - val_accuracy: 0.7824
    Epoch 5/5
    24/24 [==============================] - 549s 24s/step - loss: 0.3414 - accuracy: 0.9267 - val_loss: 0.6338 - val_accuracy: 0.7988
    


```python
import matplotlib.pyplot as plt
```


```python
def plot_loss_curves(history):
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]

  epochs = range(len(history.history["loss"]))

  plt.plot(epochs, loss, label = "traning loss")
  plt.plot(epochs, val_loss, label = "val_loss")
  plt.title("loss")
  plt.xlabel("Epochs")
  plt.legend()

  plt.figure()
  plt.plot(epochs, accuracy, label = "traning accuracy")
  plt.plot(epochs, val_accuracy, label = "val_accuracy")
  plt.title("accuracy")
  plt.xlabel("Epochs")
  plt.legend()  
```


```python
plot_loss_curves(resnet_history)
```


    
![png](output_22_0.png)
    



    
![png](output_22_1.png)
    



```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     my_feature_extraction_layer  (None, 2048)             58331648  
      (KerasLayer)                                                   
                                                                     
     my_output_layer (Dense)     (None, 10)                20490     
                                                                     
    =================================================================
    Total params: 58,352,138
    Trainable params: 20,490
    Non-trainable params: 58,331,648
    _________________________________________________________________
    


```python
EFFICIENTNET_URL = "https://tfhub.dev/google/efficientnet/b7/feature-vector/1"
```


```python
feature_extractor_layer = hub.KerasLayer(
    EFFICIENTNET_URL,
    trainable = False,
    name = "my_feature_extraction_layer",
    input_shape = (224, 224, 3)
)

efficientnet_model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(10, activation = "softmax", name = "my_output_layer")
])

efficientnet_model.compile(
    loss = "categorical_crossentropy",
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ["accuracy"]
)

efficientnet_history = efficientnet_model.fit(
    train_data,
    epochs = 5,
    validation_data = test_data
)
```

    Epoch 1/5
    24/24 [==============================] - 942s 40s/step - loss: 1.5849 - accuracy: 0.5880 - val_loss: 0.9568 - val_accuracy: 0.7888
    Epoch 2/5
    24/24 [==============================] - 794s 34s/step - loss: 0.7921 - accuracy: 0.8147 - val_loss: 0.6738 - val_accuracy: 0.8264
    Epoch 3/5
    24/24 [==============================] - 798s 34s/step - loss: 0.5789 - accuracy: 0.8667 - val_loss: 0.5776 - val_accuracy: 0.8468
    Epoch 4/5
    24/24 [==============================] - 833s 36s/step - loss: 0.4767 - accuracy: 0.8907 - val_loss: 0.5310 - val_accuracy: 0.8504
    Epoch 5/5
    24/24 [==============================] - 916s 39s/step - loss: 0.4075 - accuracy: 0.8987 - val_loss: 0.5006 - val_accuracy: 0.8576
    


```python
plot_loss_curves(efficientnet_history)
```


    
![png](output_26_0.png)
    



    
![png](output_26_1.png)
    



```python
efficientnet_model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     my_feature_extraction_layer  (None, 2560)             64097680  
      (KerasLayer)                                                   
                                                                     
     my_output_layer (Dense)     (None, 10)                25610     
                                                                     
    =================================================================
    Total params: 64,123,290
    Trainable params: 25,610
    Non-trainable params: 64,097,680
    _________________________________________________________________
    


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     my_feature_extraction_layer  (None, 2048)             58331648  
      (KerasLayer)                                                   
                                                                     
     my_output_layer (Dense)     (None, 10)                20490     
                                                                     
    =================================================================
    Total params: 58,352,138
    Trainable params: 20,490
    Non-trainable params: 58,331,648
    _________________________________________________________________
    


```python
import zipfile

zip_ref = zipfile.ZipFile("10_food_classes_1_percent.zip", "r")
zip_ref.extractall()
zip_ref.close()
```


```python
import os

for dirpath, dirnames, filenames in os.walk("10_food_classes_1_percent"):
    print(f"{dirpath}에는 {len(dirnames)}개 디렉토리와 {len(filenames)}개 파일이 존재합니다.")
```

    10_food_classes_1_percent에는 2개 디렉토리와 0개 파일이 존재합니다.
    10_food_classes_1_percent\test에는 10개 디렉토리와 0개 파일이 존재합니다.
    10_food_classes_1_percent\test\chicken_curry에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_1_percent\test\chicken_wings에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_1_percent\test\fried_rice에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_1_percent\test\grilled_salmon에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_1_percent\test\hamburger에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_1_percent\test\ice_cream에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_1_percent\test\pizza에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_1_percent\test\ramen에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_1_percent\test\steak에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_1_percent\test\sushi에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_1_percent\train에는 10개 디렉토리와 0개 파일이 존재합니다.
    10_food_classes_1_percent\train\chicken_curry에는 0개 디렉토리와 7개 파일이 존재합니다.
    10_food_classes_1_percent\train\chicken_wings에는 0개 디렉토리와 7개 파일이 존재합니다.
    10_food_classes_1_percent\train\fried_rice에는 0개 디렉토리와 7개 파일이 존재합니다.
    10_food_classes_1_percent\train\grilled_salmon에는 0개 디렉토리와 7개 파일이 존재합니다.
    10_food_classes_1_percent\train\hamburger에는 0개 디렉토리와 7개 파일이 존재합니다.
    10_food_classes_1_percent\train\ice_cream에는 0개 디렉토리와 7개 파일이 존재합니다.
    10_food_classes_1_percent\train\pizza에는 0개 디렉토리와 7개 파일이 존재합니다.
    10_food_classes_1_percent\train\ramen에는 0개 디렉토리와 7개 파일이 존재합니다.
    10_food_classes_1_percent\train\steak에는 0개 디렉토리와 7개 파일이 존재합니다.
    10_food_classes_1_percent\train\sushi에는 0개 디렉토리와 7개 파일이 존재합니다.
    


```python
train_dir = "10_food_classes_1_percent/train/"
test_dir = "10_food_classes_1_percent/test/"
```


```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1/255.)
test_datagen = ImageDataGenerator(rescale = 1/255.)
```


```python
import pathlib

data_dir = pathlib.Path(train_dir)

class_names = sorted([item.name for item in data_dir.glob("*")])
print(class_names)
```

    ['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 'hamburger', 'ice_cream', 'pizza', 'ramen', 'steak', 'sushi']
    


```python
# callback 함수
# GUI 프로그램의 경우 이벤트 처리가 매우 중요, 사용자 키보드 입력, 마우스 입력 등등
# 특정한 이벤트가 발생했을 때 해당 이벤트를 처리해 주는 것 -> callback

# 1) Experiment tracking with TensorBoard
# 2) Model checkpointing
# 3) Early stopping
```


```python
train_data = train_datagen.flow_from_directory(
    train_dir,
    batch_size = 32,
    target_size = (224, 224),
    class_mode = "categorical"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    batch_size = 32,
    target_size = (224, 224),
    class_mode = "categorical"
)
```

    Found 70 images belonging to 10 classes.
    Found 2500 images belonging to 10 classes.
    


```python
import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir = log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback
```


```python
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

EFFICIENTNET_URL = "https://tfhub.dev/google/efficientnet/b7/feature-vector/1"

feature_extractor_layer = hub.KerasLayer(
    EFFICIENTNET_URL,
    trainable = False,
    name = "my_feature_extraction_layer",
    input_shape = (224, 224, 3)
)

efficientnet_model_1 = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(10, activation = "softmax", name = "my_output_layer")
])

efficientnet_model_1.compile(
    loss = "categorical_crossentropy",
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ["accuracy"]
)

efficientnet_history_1 = efficientnet_model_1.fit(
    train_data,
    epochs = 3,
    validation_data = test_data,
    callbacks = [
        create_tensorboard_callback(
            dir_name = "tensorflow_hub",
            experiment_name = "efficientnet_test"
        )
    ]
)
```

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.data_structures has been moved to tensorflow.python.trackable.data_structures. The old module will be deleted in version 2.11.
    WARNING:tensorflow:From C:\Users\fermat39\tensorflow_week\venv_tensorflowweek\lib\site-packages\tensorflow\python\autograph\pyct\static_analysis\liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.
    Instructions for updating:
    Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089
    

    WARNING:tensorflow:From C:\Users\fermat39\tensorflow_week\venv_tensorflowweek\lib\site-packages\tensorflow\python\autograph\pyct\static_analysis\liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.
    Instructions for updating:
    Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089
    

    Saving TensorBoard log files to: tensorflow_hub/efficientnet_test/20221130-185119
    Epoch 1/3
    3/3 [==============================] - 630s 302s/step - loss: 2.3667 - accuracy: 0.1143 - val_loss: 2.1483 - val_accuracy: 0.3124
    Epoch 2/3
    3/3 [==============================] - 607s 299s/step - loss: 1.9849 - accuracy: 0.4000 - val_loss: 1.9431 - val_accuracy: 0.4580
    Epoch 3/3
    3/3 [==============================] - 611s 302s/step - loss: 1.7268 - accuracy: 0.7000 - val_loss: 1.7599 - val_accuracy: 0.5392
    


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
```


```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.5),
    layers.RandomZoom(0.5),
    layers.RandomHeight(0.5),
    layers.RandomWidth(0.5),
], name = "my_data_augmentation"
)
```


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random

target_class = random.choice(class_names)
target_dir = "10_food_classes_1_percent/train/" + target_class
random_image = random.choice(os.listdir(target_dir))
random_image_path = target_dir + "/" + random_image
img = mpimg.imread(random_image_path)
plt.imshow(img)
plt.title(f"Origianl random image from class : {target_class}")
plt.axis(False)

plt.figure()
augmented_img = data_augmentation(tf.expand_dims(img, axis = 0))
plt.imshow(tf.squeeze(augmented_img)/255.)
plt.title(f"Augmented random image from class : {target_class}")
plt.axis(False)
```

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    




    (-0.5, 401.5, 568.5, -0.5)




    
![png](output_40_21.png)
    



    
![png](output_40_22.png)
    



```python
base_model = tf.keras.applications.EfficientNetB7(include_top = False)
base_model.trainable = False

inputs = tf.keras.layers.Input(shape=(224, 224, 3), name = "1p_input_layer")

x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name = "1p_global_average_pooling_layer")(x)
outputs = tf.keras.layers.Dense(10, activation="softmax", name="1p_output_layer")(x)
efficientnet_model_1 = tf.keras.Model(inputs, outputs)

efficientnet_model_1.compile(
    loss = "categorical_crossentropy",
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ["accuracy"]
)

efficientnet_history_1 = efficientnet_model_1.fit(
    train_data,
    epochs = 5,
    validation_data = test_data,
    callbacks = [
        create_tensorboard_callback(
            dir_name = "tensorflow_hub",
            experiment_name = "efficientnet_test"
        )
    ]
)
```

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    


```python
efficientnet_model_1.summary()
```

    Model: "model_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     1p_input_layer (InputLayer)  [(None, 224, 224, 3)]    0         
                                                                     
     my_data_augmentation (Seque  (None, None, None, 3)    0         
     ntial)                                                          
                                                                     
     efficientnetb7 (Functional)  (None, None, None, 2560)  64097687 
                                                                     
     1p_global_average_pooling_l  (None, 2560)             0         
     ayer (GlobalAveragePooling2                                     
     D)                                                              
                                                                     
     1p_output_layer (Dense)     (None, 10)                25610     
                                                                     
    =================================================================
    Total params: 64,123,297
    Trainable params: 25,610
    Non-trainable params: 64,097,687
    _________________________________________________________________
    


```python
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib

for dirpath, dirnames, filenames in os.walk("10_food_classes_10_percent"):
    print(f"{dirpath}에는 {len(dirnames)}개 디렉토리와 {len(filenames)}개 파일이 존재합니다.")

train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"


train_datagen = ImageDataGenerator(rescale = 1/255.)
test_datagen = ImageDataGenerator(rescale = 1/255.)


data_dir = pathlib.Path(train_dir)

class_names = sorted([item.name for item in data_dir.glob("*")])
print(class_names)

train_data = train_datagen.flow_from_directory(
    train_dir,
    batch_size = 32,
    target_size = (224, 224),
    class_mode = "categorical"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    batch_size = 32,
    target_size = (224, 224),
    class_mode = "categorical"
)
```

    10_food_classes_10_percent에는 2개 디렉토리와 0개 파일이 존재합니다.
    10_food_classes_10_percent\test에는 10개 디렉토리와 0개 파일이 존재합니다.
    10_food_classes_10_percent\test\chicken_curry에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\chicken_wings에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\fried_rice에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\grilled_salmon에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\hamburger에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\ice_cream에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\pizza에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\ramen에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\steak에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\test\sushi에는 0개 디렉토리와 250개 파일이 존재합니다.
    10_food_classes_10_percent\train에는 10개 디렉토리와 0개 파일이 존재합니다.
    10_food_classes_10_percent\train\chicken_curry에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\chicken_wings에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\fried_rice에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\grilled_salmon에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\hamburger에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\ice_cream에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\pizza에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\ramen에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\steak에는 0개 디렉토리와 75개 파일이 존재합니다.
    10_food_classes_10_percent\train\sushi에는 0개 디렉토리와 75개 파일이 존재합니다.
    ['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 'hamburger', 'ice_cream', 'pizza', 'ramen', 'steak', 'sushi']
    Found 750 images belonging to 10 classes.
    Found 2500 images belonging to 10 classes.
    


```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.5),
    layers.RandomZoom(0.5),
    layers.RandomHeight(0.5),
    layers.RandomWidth(0.5),
], name = "my_data_augmentation"
)
```


```python
import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir = log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback
```


```python
checkpoint_path = "ten_percent_model_checkpoints_weights/checkpoint.ckpt"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only = True,
    save_best_only=False,
    save_freq="epoch",
    verbose=1
)
```


```python
base_model = tf.keras.applications.EfficientNetB7(include_top = False)
base_model.trainable = False

inputs = tf.keras.layers.Input(shape=(224, 224, 3), name = "10p_input_layer")
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name = "10p_global_average_pooling_layer")(x)
outputs = tf.keras.layers.Dense(10, activation="softmax", name="10p_output_layer")(x)
efficientnet_model_2 = tf.keras.Model(inputs, outputs)

efficientnet_model_2.compile(
    loss = "categorical_crossentropy",
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ["accuracy"]
)

efficientnet_history_2 = efficientnet_model_2.fit(
    train_data,
    epochs = 5,
    validation_data = test_data,
    callbacks = [
        create_tensorboard_callback(
            dir_name = "tensorflow_hub",
            experiment_name = "efficientnet_test"
        ),
        checkpoint_callback
    ]
)
```

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    Saving TensorBoard log files to: tensorflow_hub/efficientnet_test/20221201-192003
    WARNING:tensorflow:Model failed to serialize as JSON. Ignoring... Unable to serialize [2.0896919 2.1128857 2.1081853] to JSON. Unrecognized type <class 'tensorflow.python.framework.ops.EagerTensor'>.
    

    WARNING:tensorflow:Model failed to serialize as JSON. Ignoring... Unable to serialize [2.0896919 2.1128857 2.1081853] to JSON. Unrecognized type <class 'tensorflow.python.framework.ops.EagerTensor'>.
    

    Epoch 1/5
    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    24/24 [==============================] - ETA: 0s - loss: 2.4332 - accuracy: 0.0880
    Epoch 1: saving model to ten_percent_model_checkpoints_weights\checkpoint.ckpt
    24/24 [==============================] - 688s 29s/step - loss: 2.4332 - accuracy: 0.0880 - val_loss: 2.3791 - val_accuracy: 0.1000
    Epoch 2/5
    24/24 [==============================] - ETA: 0s - loss: 2.3801 - accuracy: 0.0973
    Epoch 2: saving model to ten_percent_model_checkpoints_weights\checkpoint.ckpt
    24/24 [==============================] - 622s 27s/step - loss: 2.3801 - accuracy: 0.0973 - val_loss: 2.3841 - val_accuracy: 0.1000
    Epoch 3/5
    24/24 [==============================] - ETA: 0s - loss: 2.4161 - accuracy: 0.0933
    Epoch 3: saving model to ten_percent_model_checkpoints_weights\checkpoint.ckpt
    24/24 [==============================] - 632s 27s/step - loss: 2.4161 - accuracy: 0.0933 - val_loss: 2.3287 - val_accuracy: 0.1000
    Epoch 4/5
    24/24 [==============================] - ETA: 0s - loss: 2.3607 - accuracy: 0.0960
    Epoch 4: saving model to ten_percent_model_checkpoints_weights\checkpoint.ckpt
    24/24 [==============================] - 659s 29s/step - loss: 2.3607 - accuracy: 0.0960 - val_loss: 2.3444 - val_accuracy: 0.1020
    Epoch 5/5
    24/24 [==============================] - ETA: 0s - loss: 2.3577 - accuracy: 0.1053
    Epoch 5: saving model to ten_percent_model_checkpoints_weights\checkpoint.ckpt
    24/24 [==============================] - 820s 35s/step - loss: 2.3577 - accuracy: 0.1053 - val_loss: 2.3411 - val_accuracy: 0.1004
    


```python
plot_loss_curves(efficientnet_history_2)
```


    
![png](output_48_0.png)
    



    
![png](output_48_1.png)
    



```python
checkpoint_path = "ten_percent_model_checkpoints_weights/checkpoint.ckpt"

efficientnet_model_2.load_weights(checkpoint_path)
```




    <tensorflow.python.checkpoint.checkpoint.CheckpointLoadStatus at 0x17c14a2a580>




```python
loaded_weightS_model_results = efficientnet_model_2.evaluate(test_data)
```

    79/79 [==============================] - 617s 8s/step - loss: 2.3411 - accuracy: 0.1004
    


```python
# 지금까지 우리가 한 전이학습 : feature extraction transfer learning + 5 epochs + 10% dataset
```


```python
# fine tuning transfer learning의 목적 : 우리의 데이터로 (custom data)로 기존에 학습된 모델을 좀더 학습을 시키자!
```


```python
efficientnet_model_2.summary()
```

    Model: "model_3"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     10p_input_layer (InputLayer  [(None, 224, 224, 3)]    0         
     )                                                               
                                                                     
     my_data_augmentation (Seque  (None, None, None, 3)    0         
     ntial)                                                          
                                                                     
     efficientnetb7 (Functional)  (None, None, None, 2560)  64097687 
                                                                     
     10p_global_average_pooling_  (None, 2560)             0         
     layer (GlobalAveragePooling                                     
     2D)                                                             
                                                                     
     10p_output_layer (Dense)    (None, 10)                25610     
                                                                     
    =================================================================
    Total params: 64,123,297
    Trainable params: 25,610
    Non-trainable params: 64,097,687
    _________________________________________________________________
    


```python
efficientnet_model_2.layers
```




    [<keras.engine.input_layer.InputLayer at 0x18ed41b60d0>,
     <keras.engine.sequential.Sequential at 0x18ed9abf6a0>,
     <keras.engine.functional.Functional at 0x18f1de16910>,
     <keras.layers.pooling.global_average_pooling2d.GlobalAveragePooling2D at 0x18ee2e270d0>,
     <keras.layers.core.dense.Dense at 0x18f1bf8ec70>]




```python
for layer in efficientnet_model_2.layers:
    print(layer.trainable)
```

    True
    True
    False
    True
    True
    


```python
efficientnet_model_2.layers[2]
```




    <keras.engine.functional.Functional at 0x18f1de16910>




```python
efficientnet_model_2.layers[2].trainable_variables
```




    []




```python
len(efficientnet_model_2.layers[2].trainable_variables)
```




    0




```python
len(base_model.trainable_variables)
```




    0




```python
for layer_number, layer in enumerate(base_model.layers):
    print(f"{layer_number}\t{layer.name}\t\t\t\t{layer.trainable}")
```

    0	input_4				False
    1	rescaling_6				False
    2	normalization_3				False
    3	rescaling_7				False
    4	stem_conv_pad				False
    5	stem_conv				False
    6	stem_bn				False
    7	stem_activation				False
    8	block1a_dwconv				False
    9	block1a_bn				False
    10	block1a_activation				False
    11	block1a_se_squeeze				False
    12	block1a_se_reshape				False
    13	block1a_se_reduce				False
    14	block1a_se_expand				False
    15	block1a_se_excite				False
    16	block1a_project_conv				False
    17	block1a_project_bn				False
    18	block1b_dwconv				False
    19	block1b_bn				False
    20	block1b_activation				False
    21	block1b_se_squeeze				False
    22	block1b_se_reshape				False
    23	block1b_se_reduce				False
    24	block1b_se_expand				False
    25	block1b_se_excite				False
    26	block1b_project_conv				False
    27	block1b_project_bn				False
    28	block1b_drop				False
    29	block1b_add				False
    30	block1c_dwconv				False
    31	block1c_bn				False
    32	block1c_activation				False
    33	block1c_se_squeeze				False
    34	block1c_se_reshape				False
    35	block1c_se_reduce				False
    36	block1c_se_expand				False
    37	block1c_se_excite				False
    38	block1c_project_conv				False
    39	block1c_project_bn				False
    40	block1c_drop				False
    41	block1c_add				False
    42	block1d_dwconv				False
    43	block1d_bn				False
    44	block1d_activation				False
    45	block1d_se_squeeze				False
    46	block1d_se_reshape				False
    47	block1d_se_reduce				False
    48	block1d_se_expand				False
    49	block1d_se_excite				False
    50	block1d_project_conv				False
    51	block1d_project_bn				False
    52	block1d_drop				False
    53	block1d_add				False
    54	block2a_expand_conv				False
    55	block2a_expand_bn				False
    56	block2a_expand_activation				False
    57	block2a_dwconv_pad				False
    58	block2a_dwconv				False
    59	block2a_bn				False
    60	block2a_activation				False
    61	block2a_se_squeeze				False
    62	block2a_se_reshape				False
    63	block2a_se_reduce				False
    64	block2a_se_expand				False
    65	block2a_se_excite				False
    66	block2a_project_conv				False
    67	block2a_project_bn				False
    68	block2b_expand_conv				False
    69	block2b_expand_bn				False
    70	block2b_expand_activation				False
    71	block2b_dwconv				False
    72	block2b_bn				False
    73	block2b_activation				False
    74	block2b_se_squeeze				False
    75	block2b_se_reshape				False
    76	block2b_se_reduce				False
    77	block2b_se_expand				False
    78	block2b_se_excite				False
    79	block2b_project_conv				False
    80	block2b_project_bn				False
    81	block2b_drop				False
    82	block2b_add				False
    83	block2c_expand_conv				False
    84	block2c_expand_bn				False
    85	block2c_expand_activation				False
    86	block2c_dwconv				False
    87	block2c_bn				False
    88	block2c_activation				False
    89	block2c_se_squeeze				False
    90	block2c_se_reshape				False
    91	block2c_se_reduce				False
    92	block2c_se_expand				False
    93	block2c_se_excite				False
    94	block2c_project_conv				False
    95	block2c_project_bn				False
    96	block2c_drop				False
    97	block2c_add				False
    98	block2d_expand_conv				False
    99	block2d_expand_bn				False
    100	block2d_expand_activation				False
    101	block2d_dwconv				False
    102	block2d_bn				False
    103	block2d_activation				False
    104	block2d_se_squeeze				False
    105	block2d_se_reshape				False
    106	block2d_se_reduce				False
    107	block2d_se_expand				False
    108	block2d_se_excite				False
    109	block2d_project_conv				False
    110	block2d_project_bn				False
    111	block2d_drop				False
    112	block2d_add				False
    113	block2e_expand_conv				False
    114	block2e_expand_bn				False
    115	block2e_expand_activation				False
    116	block2e_dwconv				False
    117	block2e_bn				False
    118	block2e_activation				False
    119	block2e_se_squeeze				False
    120	block2e_se_reshape				False
    121	block2e_se_reduce				False
    122	block2e_se_expand				False
    123	block2e_se_excite				False
    124	block2e_project_conv				False
    125	block2e_project_bn				False
    126	block2e_drop				False
    127	block2e_add				False
    128	block2f_expand_conv				False
    129	block2f_expand_bn				False
    130	block2f_expand_activation				False
    131	block2f_dwconv				False
    132	block2f_bn				False
    133	block2f_activation				False
    134	block2f_se_squeeze				False
    135	block2f_se_reshape				False
    136	block2f_se_reduce				False
    137	block2f_se_expand				False
    138	block2f_se_excite				False
    139	block2f_project_conv				False
    140	block2f_project_bn				False
    141	block2f_drop				False
    142	block2f_add				False
    143	block2g_expand_conv				False
    144	block2g_expand_bn				False
    145	block2g_expand_activation				False
    146	block2g_dwconv				False
    147	block2g_bn				False
    148	block2g_activation				False
    149	block2g_se_squeeze				False
    150	block2g_se_reshape				False
    151	block2g_se_reduce				False
    152	block2g_se_expand				False
    153	block2g_se_excite				False
    154	block2g_project_conv				False
    155	block2g_project_bn				False
    156	block2g_drop				False
    157	block2g_add				False
    158	block3a_expand_conv				False
    159	block3a_expand_bn				False
    160	block3a_expand_activation				False
    161	block3a_dwconv_pad				False
    162	block3a_dwconv				False
    163	block3a_bn				False
    164	block3a_activation				False
    165	block3a_se_squeeze				False
    166	block3a_se_reshape				False
    167	block3a_se_reduce				False
    168	block3a_se_expand				False
    169	block3a_se_excite				False
    170	block3a_project_conv				False
    171	block3a_project_bn				False
    172	block3b_expand_conv				False
    173	block3b_expand_bn				False
    174	block3b_expand_activation				False
    175	block3b_dwconv				False
    176	block3b_bn				False
    177	block3b_activation				False
    178	block3b_se_squeeze				False
    179	block3b_se_reshape				False
    180	block3b_se_reduce				False
    181	block3b_se_expand				False
    182	block3b_se_excite				False
    183	block3b_project_conv				False
    184	block3b_project_bn				False
    185	block3b_drop				False
    186	block3b_add				False
    187	block3c_expand_conv				False
    188	block3c_expand_bn				False
    189	block3c_expand_activation				False
    190	block3c_dwconv				False
    191	block3c_bn				False
    192	block3c_activation				False
    193	block3c_se_squeeze				False
    194	block3c_se_reshape				False
    195	block3c_se_reduce				False
    196	block3c_se_expand				False
    197	block3c_se_excite				False
    198	block3c_project_conv				False
    199	block3c_project_bn				False
    200	block3c_drop				False
    201	block3c_add				False
    202	block3d_expand_conv				False
    203	block3d_expand_bn				False
    204	block3d_expand_activation				False
    205	block3d_dwconv				False
    206	block3d_bn				False
    207	block3d_activation				False
    208	block3d_se_squeeze				False
    209	block3d_se_reshape				False
    210	block3d_se_reduce				False
    211	block3d_se_expand				False
    212	block3d_se_excite				False
    213	block3d_project_conv				False
    214	block3d_project_bn				False
    215	block3d_drop				False
    216	block3d_add				False
    217	block3e_expand_conv				False
    218	block3e_expand_bn				False
    219	block3e_expand_activation				False
    220	block3e_dwconv				False
    221	block3e_bn				False
    222	block3e_activation				False
    223	block3e_se_squeeze				False
    224	block3e_se_reshape				False
    225	block3e_se_reduce				False
    226	block3e_se_expand				False
    227	block3e_se_excite				False
    228	block3e_project_conv				False
    229	block3e_project_bn				False
    230	block3e_drop				False
    231	block3e_add				False
    232	block3f_expand_conv				False
    233	block3f_expand_bn				False
    234	block3f_expand_activation				False
    235	block3f_dwconv				False
    236	block3f_bn				False
    237	block3f_activation				False
    238	block3f_se_squeeze				False
    239	block3f_se_reshape				False
    240	block3f_se_reduce				False
    241	block3f_se_expand				False
    242	block3f_se_excite				False
    243	block3f_project_conv				False
    244	block3f_project_bn				False
    245	block3f_drop				False
    246	block3f_add				False
    247	block3g_expand_conv				False
    248	block3g_expand_bn				False
    249	block3g_expand_activation				False
    250	block3g_dwconv				False
    251	block3g_bn				False
    252	block3g_activation				False
    253	block3g_se_squeeze				False
    254	block3g_se_reshape				False
    255	block3g_se_reduce				False
    256	block3g_se_expand				False
    257	block3g_se_excite				False
    258	block3g_project_conv				False
    259	block3g_project_bn				False
    260	block3g_drop				False
    261	block3g_add				False
    262	block4a_expand_conv				False
    263	block4a_expand_bn				False
    264	block4a_expand_activation				False
    265	block4a_dwconv_pad				False
    266	block4a_dwconv				False
    267	block4a_bn				False
    268	block4a_activation				False
    269	block4a_se_squeeze				False
    270	block4a_se_reshape				False
    271	block4a_se_reduce				False
    272	block4a_se_expand				False
    273	block4a_se_excite				False
    274	block4a_project_conv				False
    275	block4a_project_bn				False
    276	block4b_expand_conv				False
    277	block4b_expand_bn				False
    278	block4b_expand_activation				False
    279	block4b_dwconv				False
    280	block4b_bn				False
    281	block4b_activation				False
    282	block4b_se_squeeze				False
    283	block4b_se_reshape				False
    284	block4b_se_reduce				False
    285	block4b_se_expand				False
    286	block4b_se_excite				False
    287	block4b_project_conv				False
    288	block4b_project_bn				False
    289	block4b_drop				False
    290	block4b_add				False
    291	block4c_expand_conv				False
    292	block4c_expand_bn				False
    293	block4c_expand_activation				False
    294	block4c_dwconv				False
    295	block4c_bn				False
    296	block4c_activation				False
    297	block4c_se_squeeze				False
    298	block4c_se_reshape				False
    299	block4c_se_reduce				False
    300	block4c_se_expand				False
    301	block4c_se_excite				False
    302	block4c_project_conv				False
    303	block4c_project_bn				False
    304	block4c_drop				False
    305	block4c_add				False
    306	block4d_expand_conv				False
    307	block4d_expand_bn				False
    308	block4d_expand_activation				False
    309	block4d_dwconv				False
    310	block4d_bn				False
    311	block4d_activation				False
    312	block4d_se_squeeze				False
    313	block4d_se_reshape				False
    314	block4d_se_reduce				False
    315	block4d_se_expand				False
    316	block4d_se_excite				False
    317	block4d_project_conv				False
    318	block4d_project_bn				False
    319	block4d_drop				False
    320	block4d_add				False
    321	block4e_expand_conv				False
    322	block4e_expand_bn				False
    323	block4e_expand_activation				False
    324	block4e_dwconv				False
    325	block4e_bn				False
    326	block4e_activation				False
    327	block4e_se_squeeze				False
    328	block4e_se_reshape				False
    329	block4e_se_reduce				False
    330	block4e_se_expand				False
    331	block4e_se_excite				False
    332	block4e_project_conv				False
    333	block4e_project_bn				False
    334	block4e_drop				False
    335	block4e_add				False
    336	block4f_expand_conv				False
    337	block4f_expand_bn				False
    338	block4f_expand_activation				False
    339	block4f_dwconv				False
    340	block4f_bn				False
    341	block4f_activation				False
    342	block4f_se_squeeze				False
    343	block4f_se_reshape				False
    344	block4f_se_reduce				False
    345	block4f_se_expand				False
    346	block4f_se_excite				False
    347	block4f_project_conv				False
    348	block4f_project_bn				False
    349	block4f_drop				False
    350	block4f_add				False
    351	block4g_expand_conv				False
    352	block4g_expand_bn				False
    353	block4g_expand_activation				False
    354	block4g_dwconv				False
    355	block4g_bn				False
    356	block4g_activation				False
    357	block4g_se_squeeze				False
    358	block4g_se_reshape				False
    359	block4g_se_reduce				False
    360	block4g_se_expand				False
    361	block4g_se_excite				False
    362	block4g_project_conv				False
    363	block4g_project_bn				False
    364	block4g_drop				False
    365	block4g_add				False
    366	block4h_expand_conv				False
    367	block4h_expand_bn				False
    368	block4h_expand_activation				False
    369	block4h_dwconv				False
    370	block4h_bn				False
    371	block4h_activation				False
    372	block4h_se_squeeze				False
    373	block4h_se_reshape				False
    374	block4h_se_reduce				False
    375	block4h_se_expand				False
    376	block4h_se_excite				False
    377	block4h_project_conv				False
    378	block4h_project_bn				False
    379	block4h_drop				False
    380	block4h_add				False
    381	block4i_expand_conv				False
    382	block4i_expand_bn				False
    383	block4i_expand_activation				False
    384	block4i_dwconv				False
    385	block4i_bn				False
    386	block4i_activation				False
    387	block4i_se_squeeze				False
    388	block4i_se_reshape				False
    389	block4i_se_reduce				False
    390	block4i_se_expand				False
    391	block4i_se_excite				False
    392	block4i_project_conv				False
    393	block4i_project_bn				False
    394	block4i_drop				False
    395	block4i_add				False
    396	block4j_expand_conv				False
    397	block4j_expand_bn				False
    398	block4j_expand_activation				False
    399	block4j_dwconv				False
    400	block4j_bn				False
    401	block4j_activation				False
    402	block4j_se_squeeze				False
    403	block4j_se_reshape				False
    404	block4j_se_reduce				False
    405	block4j_se_expand				False
    406	block4j_se_excite				False
    407	block4j_project_conv				False
    408	block4j_project_bn				False
    409	block4j_drop				False
    410	block4j_add				False
    411	block5a_expand_conv				False
    412	block5a_expand_bn				False
    413	block5a_expand_activation				False
    414	block5a_dwconv				False
    415	block5a_bn				False
    416	block5a_activation				False
    417	block5a_se_squeeze				False
    418	block5a_se_reshape				False
    419	block5a_se_reduce				False
    420	block5a_se_expand				False
    421	block5a_se_excite				False
    422	block5a_project_conv				False
    423	block5a_project_bn				False
    424	block5b_expand_conv				False
    425	block5b_expand_bn				False
    426	block5b_expand_activation				False
    427	block5b_dwconv				False
    428	block5b_bn				False
    429	block5b_activation				False
    430	block5b_se_squeeze				False
    431	block5b_se_reshape				False
    432	block5b_se_reduce				False
    433	block5b_se_expand				False
    434	block5b_se_excite				False
    435	block5b_project_conv				False
    436	block5b_project_bn				False
    437	block5b_drop				False
    438	block5b_add				False
    439	block5c_expand_conv				False
    440	block5c_expand_bn				False
    441	block5c_expand_activation				False
    442	block5c_dwconv				False
    443	block5c_bn				False
    444	block5c_activation				False
    445	block5c_se_squeeze				False
    446	block5c_se_reshape				False
    447	block5c_se_reduce				False
    448	block5c_se_expand				False
    449	block5c_se_excite				False
    450	block5c_project_conv				False
    451	block5c_project_bn				False
    452	block5c_drop				False
    453	block5c_add				False
    454	block5d_expand_conv				False
    455	block5d_expand_bn				False
    456	block5d_expand_activation				False
    457	block5d_dwconv				False
    458	block5d_bn				False
    459	block5d_activation				False
    460	block5d_se_squeeze				False
    461	block5d_se_reshape				False
    462	block5d_se_reduce				False
    463	block5d_se_expand				False
    464	block5d_se_excite				False
    465	block5d_project_conv				False
    466	block5d_project_bn				False
    467	block5d_drop				False
    468	block5d_add				False
    469	block5e_expand_conv				False
    470	block5e_expand_bn				False
    471	block5e_expand_activation				False
    472	block5e_dwconv				False
    473	block5e_bn				False
    474	block5e_activation				False
    475	block5e_se_squeeze				False
    476	block5e_se_reshape				False
    477	block5e_se_reduce				False
    478	block5e_se_expand				False
    479	block5e_se_excite				False
    480	block5e_project_conv				False
    481	block5e_project_bn				False
    482	block5e_drop				False
    483	block5e_add				False
    484	block5f_expand_conv				False
    485	block5f_expand_bn				False
    486	block5f_expand_activation				False
    487	block5f_dwconv				False
    488	block5f_bn				False
    489	block5f_activation				False
    490	block5f_se_squeeze				False
    491	block5f_se_reshape				False
    492	block5f_se_reduce				False
    493	block5f_se_expand				False
    494	block5f_se_excite				False
    495	block5f_project_conv				False
    496	block5f_project_bn				False
    497	block5f_drop				False
    498	block5f_add				False
    499	block5g_expand_conv				False
    500	block5g_expand_bn				False
    501	block5g_expand_activation				False
    502	block5g_dwconv				False
    503	block5g_bn				False
    504	block5g_activation				False
    505	block5g_se_squeeze				False
    506	block5g_se_reshape				False
    507	block5g_se_reduce				False
    508	block5g_se_expand				False
    509	block5g_se_excite				False
    510	block5g_project_conv				False
    511	block5g_project_bn				False
    512	block5g_drop				False
    513	block5g_add				False
    514	block5h_expand_conv				False
    515	block5h_expand_bn				False
    516	block5h_expand_activation				False
    517	block5h_dwconv				False
    518	block5h_bn				False
    519	block5h_activation				False
    520	block5h_se_squeeze				False
    521	block5h_se_reshape				False
    522	block5h_se_reduce				False
    523	block5h_se_expand				False
    524	block5h_se_excite				False
    525	block5h_project_conv				False
    526	block5h_project_bn				False
    527	block5h_drop				False
    528	block5h_add				False
    529	block5i_expand_conv				False
    530	block5i_expand_bn				False
    531	block5i_expand_activation				False
    532	block5i_dwconv				False
    533	block5i_bn				False
    534	block5i_activation				False
    535	block5i_se_squeeze				False
    536	block5i_se_reshape				False
    537	block5i_se_reduce				False
    538	block5i_se_expand				False
    539	block5i_se_excite				False
    540	block5i_project_conv				False
    541	block5i_project_bn				False
    542	block5i_drop				False
    543	block5i_add				False
    544	block5j_expand_conv				False
    545	block5j_expand_bn				False
    546	block5j_expand_activation				False
    547	block5j_dwconv				False
    548	block5j_bn				False
    549	block5j_activation				False
    550	block5j_se_squeeze				False
    551	block5j_se_reshape				False
    552	block5j_se_reduce				False
    553	block5j_se_expand				False
    554	block5j_se_excite				False
    555	block5j_project_conv				False
    556	block5j_project_bn				False
    557	block5j_drop				False
    558	block5j_add				False
    559	block6a_expand_conv				False
    560	block6a_expand_bn				False
    561	block6a_expand_activation				False
    562	block6a_dwconv_pad				False
    563	block6a_dwconv				False
    564	block6a_bn				False
    565	block6a_activation				False
    566	block6a_se_squeeze				False
    567	block6a_se_reshape				False
    568	block6a_se_reduce				False
    569	block6a_se_expand				False
    570	block6a_se_excite				False
    571	block6a_project_conv				False
    572	block6a_project_bn				False
    573	block6b_expand_conv				False
    574	block6b_expand_bn				False
    575	block6b_expand_activation				False
    576	block6b_dwconv				False
    577	block6b_bn				False
    578	block6b_activation				False
    579	block6b_se_squeeze				False
    580	block6b_se_reshape				False
    581	block6b_se_reduce				False
    582	block6b_se_expand				False
    583	block6b_se_excite				False
    584	block6b_project_conv				False
    585	block6b_project_bn				False
    586	block6b_drop				False
    587	block6b_add				False
    588	block6c_expand_conv				False
    589	block6c_expand_bn				False
    590	block6c_expand_activation				False
    591	block6c_dwconv				False
    592	block6c_bn				False
    593	block6c_activation				False
    594	block6c_se_squeeze				False
    595	block6c_se_reshape				False
    596	block6c_se_reduce				False
    597	block6c_se_expand				False
    598	block6c_se_excite				False
    599	block6c_project_conv				False
    600	block6c_project_bn				False
    601	block6c_drop				False
    602	block6c_add				False
    603	block6d_expand_conv				False
    604	block6d_expand_bn				False
    605	block6d_expand_activation				False
    606	block6d_dwconv				False
    607	block6d_bn				False
    608	block6d_activation				False
    609	block6d_se_squeeze				False
    610	block6d_se_reshape				False
    611	block6d_se_reduce				False
    612	block6d_se_expand				False
    613	block6d_se_excite				False
    614	block6d_project_conv				False
    615	block6d_project_bn				False
    616	block6d_drop				False
    617	block6d_add				False
    618	block6e_expand_conv				False
    619	block6e_expand_bn				False
    620	block6e_expand_activation				False
    621	block6e_dwconv				False
    622	block6e_bn				False
    623	block6e_activation				False
    624	block6e_se_squeeze				False
    625	block6e_se_reshape				False
    626	block6e_se_reduce				False
    627	block6e_se_expand				False
    628	block6e_se_excite				False
    629	block6e_project_conv				False
    630	block6e_project_bn				False
    631	block6e_drop				False
    632	block6e_add				False
    633	block6f_expand_conv				False
    634	block6f_expand_bn				False
    635	block6f_expand_activation				False
    636	block6f_dwconv				False
    637	block6f_bn				False
    638	block6f_activation				False
    639	block6f_se_squeeze				False
    640	block6f_se_reshape				False
    641	block6f_se_reduce				False
    642	block6f_se_expand				False
    643	block6f_se_excite				False
    644	block6f_project_conv				False
    645	block6f_project_bn				False
    646	block6f_drop				False
    647	block6f_add				False
    648	block6g_expand_conv				False
    649	block6g_expand_bn				False
    650	block6g_expand_activation				False
    651	block6g_dwconv				False
    652	block6g_bn				False
    653	block6g_activation				False
    654	block6g_se_squeeze				False
    655	block6g_se_reshape				False
    656	block6g_se_reduce				False
    657	block6g_se_expand				False
    658	block6g_se_excite				False
    659	block6g_project_conv				False
    660	block6g_project_bn				False
    661	block6g_drop				False
    662	block6g_add				False
    663	block6h_expand_conv				False
    664	block6h_expand_bn				False
    665	block6h_expand_activation				False
    666	block6h_dwconv				False
    667	block6h_bn				False
    668	block6h_activation				False
    669	block6h_se_squeeze				False
    670	block6h_se_reshape				False
    671	block6h_se_reduce				False
    672	block6h_se_expand				False
    673	block6h_se_excite				False
    674	block6h_project_conv				False
    675	block6h_project_bn				False
    676	block6h_drop				False
    677	block6h_add				False
    678	block6i_expand_conv				False
    679	block6i_expand_bn				False
    680	block6i_expand_activation				False
    681	block6i_dwconv				False
    682	block6i_bn				False
    683	block6i_activation				False
    684	block6i_se_squeeze				False
    685	block6i_se_reshape				False
    686	block6i_se_reduce				False
    687	block6i_se_expand				False
    688	block6i_se_excite				False
    689	block6i_project_conv				False
    690	block6i_project_bn				False
    691	block6i_drop				False
    692	block6i_add				False
    693	block6j_expand_conv				False
    694	block6j_expand_bn				False
    695	block6j_expand_activation				False
    696	block6j_dwconv				False
    697	block6j_bn				False
    698	block6j_activation				False
    699	block6j_se_squeeze				False
    700	block6j_se_reshape				False
    701	block6j_se_reduce				False
    702	block6j_se_expand				False
    703	block6j_se_excite				False
    704	block6j_project_conv				False
    705	block6j_project_bn				False
    706	block6j_drop				False
    707	block6j_add				False
    708	block6k_expand_conv				False
    709	block6k_expand_bn				False
    710	block6k_expand_activation				False
    711	block6k_dwconv				False
    712	block6k_bn				False
    713	block6k_activation				False
    714	block6k_se_squeeze				False
    715	block6k_se_reshape				False
    716	block6k_se_reduce				False
    717	block6k_se_expand				False
    718	block6k_se_excite				False
    719	block6k_project_conv				False
    720	block6k_project_bn				False
    721	block6k_drop				False
    722	block6k_add				False
    723	block6l_expand_conv				False
    724	block6l_expand_bn				False
    725	block6l_expand_activation				False
    726	block6l_dwconv				False
    727	block6l_bn				False
    728	block6l_activation				False
    729	block6l_se_squeeze				False
    730	block6l_se_reshape				False
    731	block6l_se_reduce				False
    732	block6l_se_expand				False
    733	block6l_se_excite				False
    734	block6l_project_conv				False
    735	block6l_project_bn				False
    736	block6l_drop				False
    737	block6l_add				False
    738	block6m_expand_conv				False
    739	block6m_expand_bn				False
    740	block6m_expand_activation				False
    741	block6m_dwconv				False
    742	block6m_bn				False
    743	block6m_activation				False
    744	block6m_se_squeeze				False
    745	block6m_se_reshape				False
    746	block6m_se_reduce				False
    747	block6m_se_expand				False
    748	block6m_se_excite				False
    749	block6m_project_conv				False
    750	block6m_project_bn				False
    751	block6m_drop				False
    752	block6m_add				False
    753	block7a_expand_conv				False
    754	block7a_expand_bn				False
    755	block7a_expand_activation				False
    756	block7a_dwconv				False
    757	block7a_bn				False
    758	block7a_activation				False
    759	block7a_se_squeeze				False
    760	block7a_se_reshape				False
    761	block7a_se_reduce				False
    762	block7a_se_expand				False
    763	block7a_se_excite				False
    764	block7a_project_conv				False
    765	block7a_project_bn				False
    766	block7b_expand_conv				False
    767	block7b_expand_bn				False
    768	block7b_expand_activation				False
    769	block7b_dwconv				False
    770	block7b_bn				False
    771	block7b_activation				False
    772	block7b_se_squeeze				False
    773	block7b_se_reshape				False
    774	block7b_se_reduce				False
    775	block7b_se_expand				False
    776	block7b_se_excite				False
    777	block7b_project_conv				False
    778	block7b_project_bn				False
    779	block7b_drop				False
    780	block7b_add				False
    781	block7c_expand_conv				False
    782	block7c_expand_bn				False
    783	block7c_expand_activation				False
    784	block7c_dwconv				False
    785	block7c_bn				False
    786	block7c_activation				False
    787	block7c_se_squeeze				False
    788	block7c_se_reshape				False
    789	block7c_se_reduce				False
    790	block7c_se_expand				False
    791	block7c_se_excite				False
    792	block7c_project_conv				False
    793	block7c_project_bn				False
    794	block7c_drop				False
    795	block7c_add				False
    796	block7d_expand_conv				False
    797	block7d_expand_bn				False
    798	block7d_expand_activation				False
    799	block7d_dwconv				False
    800	block7d_bn				False
    801	block7d_activation				False
    802	block7d_se_squeeze				False
    803	block7d_se_reshape				False
    804	block7d_se_reduce				False
    805	block7d_se_expand				False
    806	block7d_se_excite				False
    807	block7d_project_conv				False
    808	block7d_project_bn				False
    809	block7d_drop				False
    810	block7d_add				False
    811	top_conv				False
    812	top_bn				False
    813	top_activation				False
    


```python
base_model.trainable
```




    False




```python
base_model.trainable = True
```


```python
base_model.trainable
```




    True




```python
len(base_model.layers[:-10])
```




    804




```python
for layer in base_model.layers[:-10]:
    layer.trainable = False
```


```python
for layer_number, layer in enumerate(base_model.layers):
    print(f"{layer_number}\t{layer.name}\t\t\t\t{layer.trainable}")
```

    0	input_4				False
    1	rescaling_6				False
    2	normalization_3				False
    3	rescaling_7				False
    4	stem_conv_pad				False
    5	stem_conv				False
    6	stem_bn				False
    7	stem_activation				False
    8	block1a_dwconv				False
    9	block1a_bn				False
    10	block1a_activation				False
    11	block1a_se_squeeze				False
    12	block1a_se_reshape				False
    13	block1a_se_reduce				False
    14	block1a_se_expand				False
    15	block1a_se_excite				False
    16	block1a_project_conv				False
    17	block1a_project_bn				False
    18	block1b_dwconv				False
    19	block1b_bn				False
    20	block1b_activation				False
    21	block1b_se_squeeze				False
    22	block1b_se_reshape				False
    23	block1b_se_reduce				False
    24	block1b_se_expand				False
    25	block1b_se_excite				False
    26	block1b_project_conv				False
    27	block1b_project_bn				False
    28	block1b_drop				False
    29	block1b_add				False
    30	block1c_dwconv				False
    31	block1c_bn				False
    32	block1c_activation				False
    33	block1c_se_squeeze				False
    34	block1c_se_reshape				False
    35	block1c_se_reduce				False
    36	block1c_se_expand				False
    37	block1c_se_excite				False
    38	block1c_project_conv				False
    39	block1c_project_bn				False
    40	block1c_drop				False
    41	block1c_add				False
    42	block1d_dwconv				False
    43	block1d_bn				False
    44	block1d_activation				False
    45	block1d_se_squeeze				False
    46	block1d_se_reshape				False
    47	block1d_se_reduce				False
    48	block1d_se_expand				False
    49	block1d_se_excite				False
    50	block1d_project_conv				False
    51	block1d_project_bn				False
    52	block1d_drop				False
    53	block1d_add				False
    54	block2a_expand_conv				False
    55	block2a_expand_bn				False
    56	block2a_expand_activation				False
    57	block2a_dwconv_pad				False
    58	block2a_dwconv				False
    59	block2a_bn				False
    60	block2a_activation				False
    61	block2a_se_squeeze				False
    62	block2a_se_reshape				False
    63	block2a_se_reduce				False
    64	block2a_se_expand				False
    65	block2a_se_excite				False
    66	block2a_project_conv				False
    67	block2a_project_bn				False
    68	block2b_expand_conv				False
    69	block2b_expand_bn				False
    70	block2b_expand_activation				False
    71	block2b_dwconv				False
    72	block2b_bn				False
    73	block2b_activation				False
    74	block2b_se_squeeze				False
    75	block2b_se_reshape				False
    76	block2b_se_reduce				False
    77	block2b_se_expand				False
    78	block2b_se_excite				False
    79	block2b_project_conv				False
    80	block2b_project_bn				False
    81	block2b_drop				False
    82	block2b_add				False
    83	block2c_expand_conv				False
    84	block2c_expand_bn				False
    85	block2c_expand_activation				False
    86	block2c_dwconv				False
    87	block2c_bn				False
    88	block2c_activation				False
    89	block2c_se_squeeze				False
    90	block2c_se_reshape				False
    91	block2c_se_reduce				False
    92	block2c_se_expand				False
    93	block2c_se_excite				False
    94	block2c_project_conv				False
    95	block2c_project_bn				False
    96	block2c_drop				False
    97	block2c_add				False
    98	block2d_expand_conv				False
    99	block2d_expand_bn				False
    100	block2d_expand_activation				False
    101	block2d_dwconv				False
    102	block2d_bn				False
    103	block2d_activation				False
    104	block2d_se_squeeze				False
    105	block2d_se_reshape				False
    106	block2d_se_reduce				False
    107	block2d_se_expand				False
    108	block2d_se_excite				False
    109	block2d_project_conv				False
    110	block2d_project_bn				False
    111	block2d_drop				False
    112	block2d_add				False
    113	block2e_expand_conv				False
    114	block2e_expand_bn				False
    115	block2e_expand_activation				False
    116	block2e_dwconv				False
    117	block2e_bn				False
    118	block2e_activation				False
    119	block2e_se_squeeze				False
    120	block2e_se_reshape				False
    121	block2e_se_reduce				False
    122	block2e_se_expand				False
    123	block2e_se_excite				False
    124	block2e_project_conv				False
    125	block2e_project_bn				False
    126	block2e_drop				False
    127	block2e_add				False
    128	block2f_expand_conv				False
    129	block2f_expand_bn				False
    130	block2f_expand_activation				False
    131	block2f_dwconv				False
    132	block2f_bn				False
    133	block2f_activation				False
    134	block2f_se_squeeze				False
    135	block2f_se_reshape				False
    136	block2f_se_reduce				False
    137	block2f_se_expand				False
    138	block2f_se_excite				False
    139	block2f_project_conv				False
    140	block2f_project_bn				False
    141	block2f_drop				False
    142	block2f_add				False
    143	block2g_expand_conv				False
    144	block2g_expand_bn				False
    145	block2g_expand_activation				False
    146	block2g_dwconv				False
    147	block2g_bn				False
    148	block2g_activation				False
    149	block2g_se_squeeze				False
    150	block2g_se_reshape				False
    151	block2g_se_reduce				False
    152	block2g_se_expand				False
    153	block2g_se_excite				False
    154	block2g_project_conv				False
    155	block2g_project_bn				False
    156	block2g_drop				False
    157	block2g_add				False
    158	block3a_expand_conv				False
    159	block3a_expand_bn				False
    160	block3a_expand_activation				False
    161	block3a_dwconv_pad				False
    162	block3a_dwconv				False
    163	block3a_bn				False
    164	block3a_activation				False
    165	block3a_se_squeeze				False
    166	block3a_se_reshape				False
    167	block3a_se_reduce				False
    168	block3a_se_expand				False
    169	block3a_se_excite				False
    170	block3a_project_conv				False
    171	block3a_project_bn				False
    172	block3b_expand_conv				False
    173	block3b_expand_bn				False
    174	block3b_expand_activation				False
    175	block3b_dwconv				False
    176	block3b_bn				False
    177	block3b_activation				False
    178	block3b_se_squeeze				False
    179	block3b_se_reshape				False
    180	block3b_se_reduce				False
    181	block3b_se_expand				False
    182	block3b_se_excite				False
    183	block3b_project_conv				False
    184	block3b_project_bn				False
    185	block3b_drop				False
    186	block3b_add				False
    187	block3c_expand_conv				False
    188	block3c_expand_bn				False
    189	block3c_expand_activation				False
    190	block3c_dwconv				False
    191	block3c_bn				False
    192	block3c_activation				False
    193	block3c_se_squeeze				False
    194	block3c_se_reshape				False
    195	block3c_se_reduce				False
    196	block3c_se_expand				False
    197	block3c_se_excite				False
    198	block3c_project_conv				False
    199	block3c_project_bn				False
    200	block3c_drop				False
    201	block3c_add				False
    202	block3d_expand_conv				False
    203	block3d_expand_bn				False
    204	block3d_expand_activation				False
    205	block3d_dwconv				False
    206	block3d_bn				False
    207	block3d_activation				False
    208	block3d_se_squeeze				False
    209	block3d_se_reshape				False
    210	block3d_se_reduce				False
    211	block3d_se_expand				False
    212	block3d_se_excite				False
    213	block3d_project_conv				False
    214	block3d_project_bn				False
    215	block3d_drop				False
    216	block3d_add				False
    217	block3e_expand_conv				False
    218	block3e_expand_bn				False
    219	block3e_expand_activation				False
    220	block3e_dwconv				False
    221	block3e_bn				False
    222	block3e_activation				False
    223	block3e_se_squeeze				False
    224	block3e_se_reshape				False
    225	block3e_se_reduce				False
    226	block3e_se_expand				False
    227	block3e_se_excite				False
    228	block3e_project_conv				False
    229	block3e_project_bn				False
    230	block3e_drop				False
    231	block3e_add				False
    232	block3f_expand_conv				False
    233	block3f_expand_bn				False
    234	block3f_expand_activation				False
    235	block3f_dwconv				False
    236	block3f_bn				False
    237	block3f_activation				False
    238	block3f_se_squeeze				False
    239	block3f_se_reshape				False
    240	block3f_se_reduce				False
    241	block3f_se_expand				False
    242	block3f_se_excite				False
    243	block3f_project_conv				False
    244	block3f_project_bn				False
    245	block3f_drop				False
    246	block3f_add				False
    247	block3g_expand_conv				False
    248	block3g_expand_bn				False
    249	block3g_expand_activation				False
    250	block3g_dwconv				False
    251	block3g_bn				False
    252	block3g_activation				False
    253	block3g_se_squeeze				False
    254	block3g_se_reshape				False
    255	block3g_se_reduce				False
    256	block3g_se_expand				False
    257	block3g_se_excite				False
    258	block3g_project_conv				False
    259	block3g_project_bn				False
    260	block3g_drop				False
    261	block3g_add				False
    262	block4a_expand_conv				False
    263	block4a_expand_bn				False
    264	block4a_expand_activation				False
    265	block4a_dwconv_pad				False
    266	block4a_dwconv				False
    267	block4a_bn				False
    268	block4a_activation				False
    269	block4a_se_squeeze				False
    270	block4a_se_reshape				False
    271	block4a_se_reduce				False
    272	block4a_se_expand				False
    273	block4a_se_excite				False
    274	block4a_project_conv				False
    275	block4a_project_bn				False
    276	block4b_expand_conv				False
    277	block4b_expand_bn				False
    278	block4b_expand_activation				False
    279	block4b_dwconv				False
    280	block4b_bn				False
    281	block4b_activation				False
    282	block4b_se_squeeze				False
    283	block4b_se_reshape				False
    284	block4b_se_reduce				False
    285	block4b_se_expand				False
    286	block4b_se_excite				False
    287	block4b_project_conv				False
    288	block4b_project_bn				False
    289	block4b_drop				False
    290	block4b_add				False
    291	block4c_expand_conv				False
    292	block4c_expand_bn				False
    293	block4c_expand_activation				False
    294	block4c_dwconv				False
    295	block4c_bn				False
    296	block4c_activation				False
    297	block4c_se_squeeze				False
    298	block4c_se_reshape				False
    299	block4c_se_reduce				False
    300	block4c_se_expand				False
    301	block4c_se_excite				False
    302	block4c_project_conv				False
    303	block4c_project_bn				False
    304	block4c_drop				False
    305	block4c_add				False
    306	block4d_expand_conv				False
    307	block4d_expand_bn				False
    308	block4d_expand_activation				False
    309	block4d_dwconv				False
    310	block4d_bn				False
    311	block4d_activation				False
    312	block4d_se_squeeze				False
    313	block4d_se_reshape				False
    314	block4d_se_reduce				False
    315	block4d_se_expand				False
    316	block4d_se_excite				False
    317	block4d_project_conv				False
    318	block4d_project_bn				False
    319	block4d_drop				False
    320	block4d_add				False
    321	block4e_expand_conv				False
    322	block4e_expand_bn				False
    323	block4e_expand_activation				False
    324	block4e_dwconv				False
    325	block4e_bn				False
    326	block4e_activation				False
    327	block4e_se_squeeze				False
    328	block4e_se_reshape				False
    329	block4e_se_reduce				False
    330	block4e_se_expand				False
    331	block4e_se_excite				False
    332	block4e_project_conv				False
    333	block4e_project_bn				False
    334	block4e_drop				False
    335	block4e_add				False
    336	block4f_expand_conv				False
    337	block4f_expand_bn				False
    338	block4f_expand_activation				False
    339	block4f_dwconv				False
    340	block4f_bn				False
    341	block4f_activation				False
    342	block4f_se_squeeze				False
    343	block4f_se_reshape				False
    344	block4f_se_reduce				False
    345	block4f_se_expand				False
    346	block4f_se_excite				False
    347	block4f_project_conv				False
    348	block4f_project_bn				False
    349	block4f_drop				False
    350	block4f_add				False
    351	block4g_expand_conv				False
    352	block4g_expand_bn				False
    353	block4g_expand_activation				False
    354	block4g_dwconv				False
    355	block4g_bn				False
    356	block4g_activation				False
    357	block4g_se_squeeze				False
    358	block4g_se_reshape				False
    359	block4g_se_reduce				False
    360	block4g_se_expand				False
    361	block4g_se_excite				False
    362	block4g_project_conv				False
    363	block4g_project_bn				False
    364	block4g_drop				False
    365	block4g_add				False
    366	block4h_expand_conv				False
    367	block4h_expand_bn				False
    368	block4h_expand_activation				False
    369	block4h_dwconv				False
    370	block4h_bn				False
    371	block4h_activation				False
    372	block4h_se_squeeze				False
    373	block4h_se_reshape				False
    374	block4h_se_reduce				False
    375	block4h_se_expand				False
    376	block4h_se_excite				False
    377	block4h_project_conv				False
    378	block4h_project_bn				False
    379	block4h_drop				False
    380	block4h_add				False
    381	block4i_expand_conv				False
    382	block4i_expand_bn				False
    383	block4i_expand_activation				False
    384	block4i_dwconv				False
    385	block4i_bn				False
    386	block4i_activation				False
    387	block4i_se_squeeze				False
    388	block4i_se_reshape				False
    389	block4i_se_reduce				False
    390	block4i_se_expand				False
    391	block4i_se_excite				False
    392	block4i_project_conv				False
    393	block4i_project_bn				False
    394	block4i_drop				False
    395	block4i_add				False
    396	block4j_expand_conv				False
    397	block4j_expand_bn				False
    398	block4j_expand_activation				False
    399	block4j_dwconv				False
    400	block4j_bn				False
    401	block4j_activation				False
    402	block4j_se_squeeze				False
    403	block4j_se_reshape				False
    404	block4j_se_reduce				False
    405	block4j_se_expand				False
    406	block4j_se_excite				False
    407	block4j_project_conv				False
    408	block4j_project_bn				False
    409	block4j_drop				False
    410	block4j_add				False
    411	block5a_expand_conv				False
    412	block5a_expand_bn				False
    413	block5a_expand_activation				False
    414	block5a_dwconv				False
    415	block5a_bn				False
    416	block5a_activation				False
    417	block5a_se_squeeze				False
    418	block5a_se_reshape				False
    419	block5a_se_reduce				False
    420	block5a_se_expand				False
    421	block5a_se_excite				False
    422	block5a_project_conv				False
    423	block5a_project_bn				False
    424	block5b_expand_conv				False
    425	block5b_expand_bn				False
    426	block5b_expand_activation				False
    427	block5b_dwconv				False
    428	block5b_bn				False
    429	block5b_activation				False
    430	block5b_se_squeeze				False
    431	block5b_se_reshape				False
    432	block5b_se_reduce				False
    433	block5b_se_expand				False
    434	block5b_se_excite				False
    435	block5b_project_conv				False
    436	block5b_project_bn				False
    437	block5b_drop				False
    438	block5b_add				False
    439	block5c_expand_conv				False
    440	block5c_expand_bn				False
    441	block5c_expand_activation				False
    442	block5c_dwconv				False
    443	block5c_bn				False
    444	block5c_activation				False
    445	block5c_se_squeeze				False
    446	block5c_se_reshape				False
    447	block5c_se_reduce				False
    448	block5c_se_expand				False
    449	block5c_se_excite				False
    450	block5c_project_conv				False
    451	block5c_project_bn				False
    452	block5c_drop				False
    453	block5c_add				False
    454	block5d_expand_conv				False
    455	block5d_expand_bn				False
    456	block5d_expand_activation				False
    457	block5d_dwconv				False
    458	block5d_bn				False
    459	block5d_activation				False
    460	block5d_se_squeeze				False
    461	block5d_se_reshape				False
    462	block5d_se_reduce				False
    463	block5d_se_expand				False
    464	block5d_se_excite				False
    465	block5d_project_conv				False
    466	block5d_project_bn				False
    467	block5d_drop				False
    468	block5d_add				False
    469	block5e_expand_conv				False
    470	block5e_expand_bn				False
    471	block5e_expand_activation				False
    472	block5e_dwconv				False
    473	block5e_bn				False
    474	block5e_activation				False
    475	block5e_se_squeeze				False
    476	block5e_se_reshape				False
    477	block5e_se_reduce				False
    478	block5e_se_expand				False
    479	block5e_se_excite				False
    480	block5e_project_conv				False
    481	block5e_project_bn				False
    482	block5e_drop				False
    483	block5e_add				False
    484	block5f_expand_conv				False
    485	block5f_expand_bn				False
    486	block5f_expand_activation				False
    487	block5f_dwconv				False
    488	block5f_bn				False
    489	block5f_activation				False
    490	block5f_se_squeeze				False
    491	block5f_se_reshape				False
    492	block5f_se_reduce				False
    493	block5f_se_expand				False
    494	block5f_se_excite				False
    495	block5f_project_conv				False
    496	block5f_project_bn				False
    497	block5f_drop				False
    498	block5f_add				False
    499	block5g_expand_conv				False
    500	block5g_expand_bn				False
    501	block5g_expand_activation				False
    502	block5g_dwconv				False
    503	block5g_bn				False
    504	block5g_activation				False
    505	block5g_se_squeeze				False
    506	block5g_se_reshape				False
    507	block5g_se_reduce				False
    508	block5g_se_expand				False
    509	block5g_se_excite				False
    510	block5g_project_conv				False
    511	block5g_project_bn				False
    512	block5g_drop				False
    513	block5g_add				False
    514	block5h_expand_conv				False
    515	block5h_expand_bn				False
    516	block5h_expand_activation				False
    517	block5h_dwconv				False
    518	block5h_bn				False
    519	block5h_activation				False
    520	block5h_se_squeeze				False
    521	block5h_se_reshape				False
    522	block5h_se_reduce				False
    523	block5h_se_expand				False
    524	block5h_se_excite				False
    525	block5h_project_conv				False
    526	block5h_project_bn				False
    527	block5h_drop				False
    528	block5h_add				False
    529	block5i_expand_conv				False
    530	block5i_expand_bn				False
    531	block5i_expand_activation				False
    532	block5i_dwconv				False
    533	block5i_bn				False
    534	block5i_activation				False
    535	block5i_se_squeeze				False
    536	block5i_se_reshape				False
    537	block5i_se_reduce				False
    538	block5i_se_expand				False
    539	block5i_se_excite				False
    540	block5i_project_conv				False
    541	block5i_project_bn				False
    542	block5i_drop				False
    543	block5i_add				False
    544	block5j_expand_conv				False
    545	block5j_expand_bn				False
    546	block5j_expand_activation				False
    547	block5j_dwconv				False
    548	block5j_bn				False
    549	block5j_activation				False
    550	block5j_se_squeeze				False
    551	block5j_se_reshape				False
    552	block5j_se_reduce				False
    553	block5j_se_expand				False
    554	block5j_se_excite				False
    555	block5j_project_conv				False
    556	block5j_project_bn				False
    557	block5j_drop				False
    558	block5j_add				False
    559	block6a_expand_conv				False
    560	block6a_expand_bn				False
    561	block6a_expand_activation				False
    562	block6a_dwconv_pad				False
    563	block6a_dwconv				False
    564	block6a_bn				False
    565	block6a_activation				False
    566	block6a_se_squeeze				False
    567	block6a_se_reshape				False
    568	block6a_se_reduce				False
    569	block6a_se_expand				False
    570	block6a_se_excite				False
    571	block6a_project_conv				False
    572	block6a_project_bn				False
    573	block6b_expand_conv				False
    574	block6b_expand_bn				False
    575	block6b_expand_activation				False
    576	block6b_dwconv				False
    577	block6b_bn				False
    578	block6b_activation				False
    579	block6b_se_squeeze				False
    580	block6b_se_reshape				False
    581	block6b_se_reduce				False
    582	block6b_se_expand				False
    583	block6b_se_excite				False
    584	block6b_project_conv				False
    585	block6b_project_bn				False
    586	block6b_drop				False
    587	block6b_add				False
    588	block6c_expand_conv				False
    589	block6c_expand_bn				False
    590	block6c_expand_activation				False
    591	block6c_dwconv				False
    592	block6c_bn				False
    593	block6c_activation				False
    594	block6c_se_squeeze				False
    595	block6c_se_reshape				False
    596	block6c_se_reduce				False
    597	block6c_se_expand				False
    598	block6c_se_excite				False
    599	block6c_project_conv				False
    600	block6c_project_bn				False
    601	block6c_drop				False
    602	block6c_add				False
    603	block6d_expand_conv				False
    604	block6d_expand_bn				False
    605	block6d_expand_activation				False
    606	block6d_dwconv				False
    607	block6d_bn				False
    608	block6d_activation				False
    609	block6d_se_squeeze				False
    610	block6d_se_reshape				False
    611	block6d_se_reduce				False
    612	block6d_se_expand				False
    613	block6d_se_excite				False
    614	block6d_project_conv				False
    615	block6d_project_bn				False
    616	block6d_drop				False
    617	block6d_add				False
    618	block6e_expand_conv				False
    619	block6e_expand_bn				False
    620	block6e_expand_activation				False
    621	block6e_dwconv				False
    622	block6e_bn				False
    623	block6e_activation				False
    624	block6e_se_squeeze				False
    625	block6e_se_reshape				False
    626	block6e_se_reduce				False
    627	block6e_se_expand				False
    628	block6e_se_excite				False
    629	block6e_project_conv				False
    630	block6e_project_bn				False
    631	block6e_drop				False
    632	block6e_add				False
    633	block6f_expand_conv				False
    634	block6f_expand_bn				False
    635	block6f_expand_activation				False
    636	block6f_dwconv				False
    637	block6f_bn				False
    638	block6f_activation				False
    639	block6f_se_squeeze				False
    640	block6f_se_reshape				False
    641	block6f_se_reduce				False
    642	block6f_se_expand				False
    643	block6f_se_excite				False
    644	block6f_project_conv				False
    645	block6f_project_bn				False
    646	block6f_drop				False
    647	block6f_add				False
    648	block6g_expand_conv				False
    649	block6g_expand_bn				False
    650	block6g_expand_activation				False
    651	block6g_dwconv				False
    652	block6g_bn				False
    653	block6g_activation				False
    654	block6g_se_squeeze				False
    655	block6g_se_reshape				False
    656	block6g_se_reduce				False
    657	block6g_se_expand				False
    658	block6g_se_excite				False
    659	block6g_project_conv				False
    660	block6g_project_bn				False
    661	block6g_drop				False
    662	block6g_add				False
    663	block6h_expand_conv				False
    664	block6h_expand_bn				False
    665	block6h_expand_activation				False
    666	block6h_dwconv				False
    667	block6h_bn				False
    668	block6h_activation				False
    669	block6h_se_squeeze				False
    670	block6h_se_reshape				False
    671	block6h_se_reduce				False
    672	block6h_se_expand				False
    673	block6h_se_excite				False
    674	block6h_project_conv				False
    675	block6h_project_bn				False
    676	block6h_drop				False
    677	block6h_add				False
    678	block6i_expand_conv				False
    679	block6i_expand_bn				False
    680	block6i_expand_activation				False
    681	block6i_dwconv				False
    682	block6i_bn				False
    683	block6i_activation				False
    684	block6i_se_squeeze				False
    685	block6i_se_reshape				False
    686	block6i_se_reduce				False
    687	block6i_se_expand				False
    688	block6i_se_excite				False
    689	block6i_project_conv				False
    690	block6i_project_bn				False
    691	block6i_drop				False
    692	block6i_add				False
    693	block6j_expand_conv				False
    694	block6j_expand_bn				False
    695	block6j_expand_activation				False
    696	block6j_dwconv				False
    697	block6j_bn				False
    698	block6j_activation				False
    699	block6j_se_squeeze				False
    700	block6j_se_reshape				False
    701	block6j_se_reduce				False
    702	block6j_se_expand				False
    703	block6j_se_excite				False
    704	block6j_project_conv				False
    705	block6j_project_bn				False
    706	block6j_drop				False
    707	block6j_add				False
    708	block6k_expand_conv				False
    709	block6k_expand_bn				False
    710	block6k_expand_activation				False
    711	block6k_dwconv				False
    712	block6k_bn				False
    713	block6k_activation				False
    714	block6k_se_squeeze				False
    715	block6k_se_reshape				False
    716	block6k_se_reduce				False
    717	block6k_se_expand				False
    718	block6k_se_excite				False
    719	block6k_project_conv				False
    720	block6k_project_bn				False
    721	block6k_drop				False
    722	block6k_add				False
    723	block6l_expand_conv				False
    724	block6l_expand_bn				False
    725	block6l_expand_activation				False
    726	block6l_dwconv				False
    727	block6l_bn				False
    728	block6l_activation				False
    729	block6l_se_squeeze				False
    730	block6l_se_reshape				False
    731	block6l_se_reduce				False
    732	block6l_se_expand				False
    733	block6l_se_excite				False
    734	block6l_project_conv				False
    735	block6l_project_bn				False
    736	block6l_drop				False
    737	block6l_add				False
    738	block6m_expand_conv				False
    739	block6m_expand_bn				False
    740	block6m_expand_activation				False
    741	block6m_dwconv				False
    742	block6m_bn				False
    743	block6m_activation				False
    744	block6m_se_squeeze				False
    745	block6m_se_reshape				False
    746	block6m_se_reduce				False
    747	block6m_se_expand				False
    748	block6m_se_excite				False
    749	block6m_project_conv				False
    750	block6m_project_bn				False
    751	block6m_drop				False
    752	block6m_add				False
    753	block7a_expand_conv				False
    754	block7a_expand_bn				False
    755	block7a_expand_activation				False
    756	block7a_dwconv				False
    757	block7a_bn				False
    758	block7a_activation				False
    759	block7a_se_squeeze				False
    760	block7a_se_reshape				False
    761	block7a_se_reduce				False
    762	block7a_se_expand				False
    763	block7a_se_excite				False
    764	block7a_project_conv				False
    765	block7a_project_bn				False
    766	block7b_expand_conv				False
    767	block7b_expand_bn				False
    768	block7b_expand_activation				False
    769	block7b_dwconv				False
    770	block7b_bn				False
    771	block7b_activation				False
    772	block7b_se_squeeze				False
    773	block7b_se_reshape				False
    774	block7b_se_reduce				False
    775	block7b_se_expand				False
    776	block7b_se_excite				False
    777	block7b_project_conv				False
    778	block7b_project_bn				False
    779	block7b_drop				False
    780	block7b_add				False
    781	block7c_expand_conv				False
    782	block7c_expand_bn				False
    783	block7c_expand_activation				False
    784	block7c_dwconv				False
    785	block7c_bn				False
    786	block7c_activation				False
    787	block7c_se_squeeze				False
    788	block7c_se_reshape				False
    789	block7c_se_reduce				False
    790	block7c_se_expand				False
    791	block7c_se_excite				False
    792	block7c_project_conv				False
    793	block7c_project_bn				False
    794	block7c_drop				False
    795	block7c_add				False
    796	block7d_expand_conv				False
    797	block7d_expand_bn				False
    798	block7d_expand_activation				False
    799	block7d_dwconv				False
    800	block7d_bn				False
    801	block7d_activation				False
    802	block7d_se_squeeze				False
    803	block7d_se_reshape				False
    804	block7d_se_reduce				True
    805	block7d_se_expand				True
    806	block7d_se_excite				True
    807	block7d_project_conv				True
    808	block7d_project_bn				True
    809	block7d_drop				True
    810	block7d_add				True
    811	top_conv				True
    812	top_bn				True
    813	top_activation				True
    


```python
efficientnet_model_2.compile(
    loss = "categorical_crossentropy",
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ["accuracy"]
)
```


```python
efficientnet_model_2.summary()
```

    Model: "model_3"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     10p_input_layer (InputLayer  [(None, 224, 224, 3)]    0         
     )                                                               
                                                                     
     my_data_augmentation (Seque  (None, None, None, 3)    0         
     ntial)                                                          
                                                                     
     efficientnetb7 (Functional)  (None, None, None, 2560)  64097687 
                                                                     
     10p_global_average_pooling_  (None, 2560)             0         
     layer (GlobalAveragePooling                                     
     2D)                                                             
                                                                     
     10p_output_layer (Dense)    (None, 10)                25610     
                                                                     
    =================================================================
    Total params: 64,123,297
    Trainable params: 5,360,810
    Non-trainable params: 58,762,487
    _________________________________________________________________
    


```python
len(efficientnet_model_2.layers[2].trainable_variables)
```




    10




```python
efficientnet_history_2 = efficientnet_model_2.fit(
    train_data,
    epochs = 5,
    validation_data = test_data,
    callbacks = [
        create_tensorboard_callback(
            dir_name = "tensorflow_hub",
            experiment_name = "efficientnet_test"
        ),
        checkpoint_callback
    ]
)
```

    Saving TensorBoard log files to: tensorflow_hub/efficientnet_test/20221201-190812
    WARNING:tensorflow:Model failed to serialize as JSON. Ignoring... Unable to serialize [2.0896919 2.1128857 2.1081853] to JSON. Unrecognized type <class 'tensorflow.python.framework.ops.EagerTensor'>.
    

    WARNING:tensorflow:Model failed to serialize as JSON. Ignoring... Unable to serialize [2.0896919 2.1128857 2.1081853] to JSON. Unrecognized type <class 'tensorflow.python.framework.ops.EagerTensor'>.
    

    Epoch 1/5
    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
    

    24/24 [==============================] - ETA: 0s - loss: 2.5274 - accuracy: 0.1027 


```python

```
