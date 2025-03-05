from tensorflow import keras
from keras import layers, optimizers, regularizers

def build_edgespeedchnet(input_shape:tuple, num_classes:int,drop:float,loss_func:str,metrics_func:str,learning_rate:float):
    model = keras.Sequential([
        # Conv2D 替代 Conv1D
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # 使用 MaxPooling2D

        layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.002)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.002)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Reshape 层，调整数据形状以适应 BiLSTM 输入
        layers.Reshape((-1, 256)),  # 调整为 (batch_size, timesteps, features)

        layers.Bidirectional(layers.LSTM(128, return_sequences=True, unroll=True)),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False, unroll=True)),

        layers.Dropout(drop),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss_func,
        metrics=metrics_func,
    )

    return model
