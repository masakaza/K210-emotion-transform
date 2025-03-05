import tensorflow as tf
import MobilienetV2_TL,EdgeSpeechNet
import librosa
import numpy as np
import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator

def extract_mfcc(audio_file, sr=16000, n_mfcc=13, max_pad_len=500):
    try:
        y, sr = librosa.load(audio_file, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # 统一长度
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc.T  # 转置，变成 (timesteps, features)
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

def train_model(model,n_mfcc,max_pad_len,sr,loss_func,metrics_func,epoch,batch_size,callbacks_func:list):
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
        
        train_img_pth='./FER-2013/train/'
        test_img_pth='./FER-2013/test/'
        
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

        train_generator=train_data.flow_from_directory(
            train_img_pth,
            target_size=(112,112), #调整尺寸 
            batch_size=32,
            color_mode='rgb',
            class_mode='categorical',
            shuffle=True,
        )

        test_data=ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        )

        test_generator=test_data.flow_from_directory(
            test_img_pth,
            target_size=(112,112),
            batch_size=32,
            color_mode='rgb',
            class_mode='categorical',
            shuffle=False,
        )
        
        models.fit(
            train_generator,
            epochs=120,
            validation_data=test_generator,
            callbacks=[reduce_lr,checkpoint]
        )

    elif model.lower == 'edgespeechnet':
        models = EdgeSpeechNet.build_edgespeedchnet(
            num_classes=6,
            input_shape=(max_pad_len,n_mfcc,1),
            drop=0.5,
            loss_func=loss_func,
            metrics_func=metrics_func,
            learning_rate=1e-3
        )

        # 读取数据并提取MFCC特征
        df = pd.read_csv('labels.csv')
        file_paths = df['filename'].tolist()
        labels = df['label'].tolist()

        X = []
        for path in file_paths:
            features = extract_mfcc(path,sr=sr,n_mfcc=n_mfcc,max_pad_len=max_pad_len)
            if features is not None:
                X.append(features)

        # 转换为 numpy 数组，X 的形状大约为 (样本数, timesteps, features)
        X = np.array(X)

        # 处理标签
        y = []
        for i in labels:
            arr = ast.literal_eval(i)
            y.append(np.array(arr))

        y = np.array(y, dtype=np.float32)

        # 将数据转换为适应 2D 卷积的形状 (samples, timesteps, features, 1)
        X_4d = np.expand_dims(X, axis=-1)  # 形状变为 (samples, timesteps, features, 1)

        # 5. 划分训练集和验证集
        x_train, x_val, y_train, y_val = train_test_split(X_4d, y, test_size=0.2, random_state=42, stratify=y)

        models.fit(
            x_train, y_train,
            epochs=epoch,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=callbacks_func,
        )

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

    n_mfcc=13
    max_pad_len=500
    sr=16000
    epoch=100
    batch_size=32
    callbacks_func=[reduce_lr, checkpoint]
    loss_func='categorical_crossentropy'
    metrics_func='categorical_accuracy'
    model=str(input('训练哪个模型'))
    train_model(model,n_mfcc,max_pad_len,sr,loss_func,metrics_func,epoch,batch_size,callbacks_func)