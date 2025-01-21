# K210-emotion-transform

以K210板为主要嵌入式设备的情感分析项目

## 项目简介

基于K210开发板对于CNN（卷积模型）具有硬件的优化效果、因此作为主要开发板来开发此项目

## 项目模型结构：

### Tiny-YOLO:
    适用于嵌入式设备的微型人脸识别模型
    框选出人脸后把人脸截出并送入微表情识别模型进行推算
- 优点：
    能降低模型的计算复杂度
### MobileNet_V2:
    把人脸的微表情进行推演，推算出当前情绪
- 优点：
    能利用到K210板为CNN模型优化的硬件条件，加快计算速度
### EdgeSpeechNet:
- 捕捉人物声音的特征，并通过特征推算出情绪的可能
- 适用RNN模型，相比于CNN的硬件优化，该模型的模型大小不宜过大
### Micro_Transformer:
    通过Transformer模型加权结合上诉模型的推演结果，并整合结果输出

## 注意事项：

K210板不适宜适用数据量过大的模型，应该适当缩减计算过程及模型大小，同时达到良好的计算结果和预期 