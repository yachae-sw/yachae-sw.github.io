---
layout: post
title: serialization 하기
subtitle: tfrecord파일
categories: tfrecord
tags: [serialization, tfrecord]
---

## tfrecord_serialization(list)


```python
import tensorflow as tf
from tensorflow.io import serialize_tensor, parse_tensor
```

### 변환


```python
originalData = tf.constant(
    value = [1, 2, 3, 4],
    dtype = tf.dtypes.uint8
)
```


```python
originalData
```




    <tf.Tensor: shape=(4,), dtype=uint8, numpy=array([1, 2, 3, 4], dtype=uint8)>




```python
serializedData = serialize_tensor(originalData)
```


```python
serializedData
```




    <tf.Tensor: shape=(), dtype=string, numpy=b'\x08\x04\x12\x04\x12\x02\x08\x04"\x04\x01\x02\x03\x04'>



### 복원


```python
parsedData = parse_tensor(
    serializedData,
    out_type = tf.dtypes.uint8
)
```


```python
parsedData
```




    <tf.Tensor: shape=(4,), dtype=uint8, numpy=array([1, 2, 3, 4], dtype=uint8)>


## Visual Python Upgrade
NOTE: 
- Refresh your web browser to start a new version.
- Save VP Note before refreshing the page.


```python
# Visual Python
!pip install visualpython --upgrade
```

    Defaulting to user installation because normal site-packages is not writeable
    Collecting visualpython
      Downloading visualpython-2.3.3-py3-none-any.whl (15.0 MB)
         ---------------------------------------- 15.0/15.0 MB 3.9 MB/s eta 0:00:00
    Installing collected packages: visualpython
    Successfully installed visualpython-2.3.3
    


```python
# Visual Python
!visualpy install
```

    'visualpy'은(는) 내부 또는 외부 명령, 실행할 수 있는 프로그램, 또는
    배치 파일이 아닙니다.
    


```python
from tensorflow.io import TFRecordWriter
from tensorflow.data import TFRecordDataset
```


```python
record = "대한민국"
binaryRecord = record.encode()
binaryRecord
```




    b'\xeb\x8c\x80\xed\x95\x9c\xeb\xaf\xbc\xea\xb5\xad'




```python
record2 = "카타르 월드컵"
binaryRecord2 = record2.encode()
binaryRecord2
```




    b'\xec\xb9\xb4\xed\x83\x80\xeb\xa5\xb4 \xec\x9b\x94\xeb\x93\x9c\xec\xbb\xb5'




```python
with TFRecordWriter("single_data.tfrecord") as recordWriter:
    recordWriter.write(binaryRecord)
    recordWriter.write(binaryRecord2)
```


```python
with open("single_data.tfrecord", "rb") as filePointer:
    print(filePointer.read())
```

    b'\x0c\x00\x00\x00\x00\x00\x00\x00\xf4\xc8\xfe\n\xeb\x8c\x80\xed\x95\x9c\xeb\xaf\xbc\xea\xb5\xad\xc9l\x85\xe1\x13\x00\x00\x00\x00\x00\x00\x00\x0c\x912\x1b\xec\xb9\xb4\xed\x83\x80\xeb\xa5\xb4 \xec\x9b\x94\xeb\x93\x9c\xec\xbb\xb5e\xc1\xa13'
    


```python
dataset = TFRecordDataset("single_data.tfrecord")
```


```python
dataset
```




    <TFRecordDatasetV2 element_spec=TensorSpec(shape=(), dtype=tf.string, name=None)>




```python
type(dataset)
```




    tensorflow.python.data.ops.readers.TFRecordDatasetV2




```python
for i, element in enumerate(dataset):
    print(i, element)
```

    0 tf.Tensor(b'\xeb\x8c\x80\xed\x95\x9c\xeb\xaf\xbc\xea\xb5\xad', shape=(), dtype=string)
    1 tf.Tensor(b'\xec\xb9\xb4\xed\x83\x80\xeb\xa5\xb4 \xec\x9b\x94\xeb\x93\x9c\xec\xbb\xb5', shape=(), dtype=string)
    


```python
# b'\xeb\x8c\x80\xed\x95\x9c\xeb\xaf\xbc\xea\xb5\xad'
for i, element in enumerate(dataset):
    element = element.numpy().decode()
    print(element)
```

    대한민국
    카타르 월드컵