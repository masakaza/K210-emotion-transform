# K210-emotion-transform
基于K210的KPU对 __CNN__ 等神经网络具有 __硬件加速__ 功能，使用 __基于MobilenetV2迁移学习网络__ 识别面部表情和使用 __EdgeSpeechNet轻量化卷积网络__ 识别语音情绪

## 目录
- [1. 安装](#安装)
    - [1.1 安装环境](#安装环境)
    - [1.2 安装数据集](#安装数据集)
- [2.使用方法](#使用方法)
- [3.微调模型](#微调模型)
- [4.使用自己的数据集训练](#使用自己的数据集训练)
    - [4.1图片数据集](#图片数据集)
    - [4.2语音数据集](#语音数据集)
- [5.模型转换](#模型转换)

## 安装
这里推荐使用虚拟环境安装库文件

### 安装环境
先切换到程序所在目录
```bash
cd <你的程序所在的目录> 
```

然后安装程序所需要的包

```bash
pip install -r requirements.txt
```
或者也可以手动安装

- __tensorflow == 2.10.0__
- __librosa == 0.10.2.post1__
- __numpy == 1.26.4__
- __pandas == 2.2.3__
- __scikit-learn == 1.6.1__

### 安装数据集

- [CASIA](https://github.com/masakaza/K210-emotion-transform/tree/main/casia)
- [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)


## 使用方法
下载好之后打开 __train.py__ 修改这些内容

```python
if __name__ == '__main__':
    # 定义回调函数
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        patience=5,
        factor=0.5,
        min_lr=1e-6,
    )

    checkpoint = callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_categorical_accuracy',
        verbose=0,
    )

    n_mfcc=13                       #输入音频特征数
    max_pad_len=500                 #音频长度
    sr=16000                        #每秒采样率
    epoch=100                       #训练轮数
    batch_size=32                   #每次训练的数据大小
    callbacks_func=[reduce_lr, checkpoint]              #回调函数
    loss_func='categorical_crossentropy'                #损失函数
    metrics_func='categorical_accuracy'                 #准确率函数
    model=str(input('训练哪个模型'))

    train_model(
    model,
    n_mfcc,
    max_pad_len,
    sr,
    loss_func,
    metrics_func,
    epoch,batch_size,
    callbacks_func
    )
```

当训练 __EdgeSpeechNet__ 时 __n_mfcc__ 和 __sr__ 调的越高模型计算量越大

 __max_pan_len__ 是设置语音的长度每100个为1秒
 
可以根据需求添加回调函数和更改损失函数和准确率函数，具体的内容可以查看 __[Tensorflow API文档](https://www.tensorflow.org/versions/r2.10/api_docs/python/tf)__

__运行__
```bash
python3 train.py
```

## 微调模型
如果想微调模型，可以调整下面这些内容
```python
if model.lower == 'mobilient' :
        models = MobilienetV2_TL.build_mobilenetv2(
            num_classes=7,
            input_shape=(112,112,3),
            drop=0.2,
            loss_func=loss_func,
            metrics_func=metrics_func,
            layer_num=100,
            trainable_layer=70,
            learning_rate=1e-4
            )
```
```python
elif model.lower == 'edgespeechnet':
        models = EdgeSpeechNet.build_edgespeedchnet(
            num_classes=6,
            input_shape=(max_pad_len,n_mfcc,1),
            drop=0.5,
            loss_func=loss_func,
            metrics_func=metrics_func,
            learning_rate=1e-3
        )
```
__EdgeSpeechNet__ 不要调整input_shape参数

- __layer_num__ ：截断Mobilenetv2模型的层数 对EdgeSpeechNet模型不起作用
- __trainable_layer__ ：冻结Mobilenetv2预训练模型参数的层数 对EdgeSpeechNet模型不起作用
- __Drop__ ：调整模型中drop层的参数即每次训练随机失活 i% 的神经节点
- __learning_rate__ ：学习率大小
- __num_classes__ : 分类的类别总数

对于 __EdgeSpeechNet.py__ 你还可以修改l2正则化参数
```python
layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape),
layers.BatchNormalization(),
layers.MaxPooling2D((2, 2)),  # 使用 MaxPooling2D

layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.002)),
layers.BatchNormalization(),
layers.MaxPooling2D((2, 2)),

layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.002)),
```
- 修改 __regularizers.l2(x)__ 里面的参数来调整正则化力度

## 使用自己的数据集训练
这个项目使用的是 __FER-2013__ 和 __CASIA__ 数据集，可以根据自己的需求下载其他数据集，下载完后按照以下方法来构建 __train__ 和 __labels__

### 图片数据集
在 __train.py__ 文件中修改这些内容
```python
train_img_pth='./FER-2013/train/'
test_img_pth='./FER-2013/test/'
```
把上面的路径改成你的数据集的路径


__同时请确保你的文件结构是这样的__
```bash
├─ <你的数据集文件>
    ├─ train/
        ├─ <类别>
            ├─ <图片...>
    ├─ teat/
        ├─ <类别>
            ├─ <图片...>
```
如果是 __灰度图像__ 按照下面代码转换图片为3通道
```python
import cv2

target_size = (112,112)  # 模型输入尺寸

def process_image(img_path):
    """处理单张图片：读取、转换通道、调整尺寸"""
    # 读取灰度图像（单通道）
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 转换为三通道（重复灰度通道）
    rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    
    # 调整尺寸
    resized_img = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_img
```

如果你想更改 __图像数据增强__ 可以修改以下内容
```python
train_data=ImageDataGenerator(
    rescale=1./255, #归一化
    rotation_range=30, #随机旋转
    width_shift_range=0.2, #随机水平平移
    height_shift_range=0.2, #随机垂直平移
    shear_range=0.2, #随机剪切变换
    zoom_range=0.2, #随机缩放
    horizontal_flip=True, #随机水平翻转
    fill_mode='nearest', #填充模式
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,  #使用模型预处理
)
```

### 语音数据集
使用语音数据集时，需要先生成一遍文件路径和labels对应的csv文件

修改 __labels_gen.py__ 内容
```python
wav_pth='./casia/'          #数据集目录
csv_file_pth='labels.csv'   #csv文件保存目录，一般不需要修改
```
同时确保文件结构
```bash
├─ <你的数据集文件>
    ├─ <类别>
        ├─ <语音...>
    ├─ <类别>
        ├─ <语音...>
```
生成的csv文件内容如下
```bash
filename,label
./casia/angry/201.wav,"[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
./casia/angry/202.wav,"[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
./casia/angry/203.wav,"[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
./casia/angry/204.wav,"[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
./casia/angry/205.wav,"[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
./casia/angry/206.wav,"[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]"

.......
```
其中labels是one-hot编码的，如想改变编码方式，自行修改 __labels_gen.py__ 文件

最后运行文件
```bash
python3 labels_gen.py
```
在目录下会生成 __labels.csv__ 文件

## 模型转换
可以参考[K210-TFlite-keras-conver-kmodel](https://github.com/masakaza/K210-TFlite-keras-conver-kmodel)进行模型的转换与部署