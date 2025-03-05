from tensorflow.keras import layers,models,optimizers
from tensorflow.keras.applications import MobileNetV2

def build_mobilenetv2(num_classes, input_shape:tuple,drop:float,loss_func:str,metrics_func:str,layer_num:int,trainable_layer:int,learning_rate:float):
    '''
    num_classes:int  分类的类别数目,
    input_shape:Tuple 模型输入的尺寸,
    drop:float Dropout参数,
    loss_func:str 损失函数,
    metrics_func:str 准确率函数,
    layer_num:int 截断层数,
    trainable_layer:int 冻结层数,
    '''

    # 加载预训练的 MobileNetV2 模型（不包含顶层）
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # 设置 trainable 状态：
    # 冻结前 trainable_layer 层，解冻第 trainable_layer ~ layer_num 层
    for i, layer in enumerate(base_model.layers[:layer_num]):
        if i < trainable_layer:
            layer.trainable = False
        else:
            layer.trainable = True

    # 构建特征提取器，输出取自 base_model 的第 100 层
    feature_extractor = models.Model(inputs=base_model.input, outputs=base_model.layers[layer_num].output)
    
    # 构建分类器部分
    x = feature_extractor.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(drop)(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)
    
    # 构建整个模型
    model = models.Model(inputs=base_model.input, outputs=predictions)
    
    # 编译模型
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss_func,
        metrics=[metrics_func],
        experimental_run_tf_function=False  # 兼容量化训练
    )
    
    return model
