# ResNet과 PlainNet 비교하기

# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이효준
- 리뷰어 : 김설아


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [x] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?  
  
- [x] 2.주석을 보고 작성자의 코드가 이해되었나요?  
```python
def build_plainnet(input_shape=(32, 32, 3), is_50=False):

    inputs = keras.layers.Input(shape=input_shape)

    # 공통 앞 부분 (conv1)
    x = keras.layers.Conv2D(64, 
                            kernel_size=7, 
                            padding='SAME', 
                            strides=2,
                            name='conv1_conv')(inputs)
    x = keras.layers.BatchNormalization(name='conv1_bn')(x)
    x = keras.layers.Activation('relu', name='conv1_relu')(x)

    # 공통 앞 부분 (conv2)
    x = keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2,
        name=f'conv2_pooling')(x)

    if is_50 is False :
        # resnet_34 구현
        strides = [1, 2, 2, 2]
        channel_cnt = [64, 128, 256, 512]
        layer_cnt = [3, 4, 6, 3]
        for i, (l_cnt, channel, stride) in enumerate(zip(layer_cnt, channel_cnt, strides)):
            x = build_plain_block(
                x,
                stride = stride,
                layer_cnt=l_cnt, 
                channel=channel,
                block_num=i,
                is_50=False
            )
    else:
        # resnet_50 구현
        strides = [2, 2, 2, 2]
        channel_cnt = [64, 128, 256, 512]
        layer_cnt = [3, 4, 6, 3]
        for i, (l_cnt, channel, stride) in enumerate(zip(layer_cnt, channel_cnt, strides)):
            x = build_plain_block(
                x,
                stride = stride,
                layer_cnt=l_cnt, 
                channel=channel,
                block_num=i,
                is_50=True
            )
    
    # 공통 출력 부분
    output = keras.layers.GlobalAveragePooling2D()(x)
    output = keras.layers.Flatten(name='flatten')(output)
    output = keras.layers.Dense(5, activation='softmax')(output)

    model = keras.Model(
        inputs=inputs,
        outputs=output
    )

    return model
```  
다음과 같이 주석을 통해 어떤 부분에 대한 내용인지 이해하기가 수월했습니다.  

- [ ] 3.코드가 에러를 유발할 가능성이 있나요?
```python
resnet_34 = build_resnet(input_shape=(224, 224, 3), is_50=False, plain=False)
```
함수의 순서가 올바르게 작성되어 에러를 유발할 가능성이 없습니다.
- [ ] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
```python
keras.utils.plot_model(resnet_34, show_shapes=True)
```
이해하고 작성 후 시각화를 통해 재확인을 하셨습니다.
- [ ] 5.코드가 간결한가요?
```python

def build_plainnet(input_shape=(32, 32, 3), is_50=False):
.
.
.
        for i, (l_cnt, channel, stride) in enumerate(zip(layer_cnt, channel_cnt, strides)):
            x = build_plain_block(
                x,
                stride = stride,
                layer_cnt=l_cnt, 
                channel=channel,
                block_num=i,
                is_50=False
            )
.
.
.
```
다음과 같이 함수를 정의해서 복잡한 단계를 간결한 코드로 정리하셨습니다.
