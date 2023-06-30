# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이효준
- 리뷰어 : 이하영


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [O] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?  
- [O] 2.주석을 보고 작성자의 코드가 이해되었나요?  
```python
def generate_cam(model, item):
    item = copy.deepcopy(item)
    width = item['image'].shape[1]
    height = item['image'].shape[0]
    
    img_tensor, class_idx = normalize_and_resize_img(item)
    
    
    # 학습한 모델에서 원하는 Layer의 output을 얻기 위해서 모델의 input과 output을 새롭게 정의해줍니다.
    cam_model = tf.keras.models.Model([model.inputs], [model.layers[-3].output, model.output])
    # cam_model의 inputs shape이 4D 랭크 텐서로 구성되어 있어, 차원을 맞추기 위해 tf.expand_dims() 함수를 사용합니다.
    # img_tesnsor.shape : (224, 224, 3) -> (1, 224, 224, 3)
    # cam_moodel에 img_tensor를 넣어 conv_outputs와 predictions을 구합니다.
    # predictions는 최종 softmax함수까지 통과한 output이 저장됩니다.
    # predictions.shape : (1, 120)
    conv_outputs, predictions = cam_model(tf.expand_dims(img_tensor, 0)) 
    # conv_outputs.shape : (1, 7, 7, 2048)인데, (H, W ,C)로 변환합니다. (7, 7, 2048)
    conv_outputs = conv_outputs[0, :, :, :]
    
    # 모델의 weight activation은 마지막 layer에 있습니다.
    # GAP을 통해 생성된 (None, 2048)가 -- Dense Layer--> (120, )벡터로 변환되고
    # (120, )이 --softmax--> (120, )이며 값들은 각 Class의 확률 정보가 저장 됩니다.
    # 이에 get_weights는 (Dense Layer 변환 weights(2048, 120), bias(120,))가 저장됩니다.)
    class_weights = model.layers[-1].get_weights()[0] 
    
    cam_image = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
    # class_weights[:, class_idx].shape : (2048, )와 conv_outputs.shape : [7, 7, 2048]
    # A사진으로부터 Class를 추출하는 (2048, 120) 가중치로부터 class_idx에 해당하는 (2048, ) 벡터의 각 elements를 
    # 마지막 convoulution의 결과(7, 7, 2048)의 0번째 채널에 곱한다.
    # cam_image는 
    for i, w in enumerate(class_weights[:, class_idx]):
        # conv_outputs의 i번째 채널과 i번째 weight를 곱해서 누적하면 활성화된 정도가 나타날 겁니다.
        # print(f"w.shape: {w}, con_outputs[:, :, i].shape: {conv_outputs[:, :, i].shape}")
        cam_image += w * conv_outputs[:, :, i]

    cam_image /= np.max(cam_image) # activation score를 normalize합니다.
    cam_image = cam_image.numpy()
    cam_image = cv2.resize(cam_image, (width, height)) # 원래 이미지의 크기로 resize합니다.
    return cam_image
```  
>  CAM 생성 함수의 코드에 대한 자세한 설명이 되어 있습니다.

- [O] 3.코드가 에러를 유발할 가능성이 있나요?<br/>
```
Epoch 19/20<br/>
750/750 [==============================] - 156s 208ms/step - loss: 0.0037 - accuracy: 0.9998 - val_loss: 1.0278 - val_accuracy: 0.7325<br/>
Epoch 20/20<br/>
750/750 [==============================] - 156s 207ms/step - loss: 0.0040 - accuracy: 0.9996 - val_loss: 0.9970 - val_accuracy: 0.7355<br/>
/opt/conda/lib/python3.9/site-packages/keras/utils/generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
  warnings.warn('Custom mask layers require a config and must override '
```
>  모델을 저장하고 나중에 다시 로드할 때 Keras는 모든 계층의 설정을 필요로 합니다. 이는 각 계층이 어떻게 구성되어 있는지를 파악하고, 동일한 구조와 파라미터로 모델을 재구성하기 위함입니다.
- [O] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
```
Grad-CAM은 CAM과 달리 Weights를 가져오지 않고, Gradient를 계산해야합니다.
따라서 with tf.GradientTape() as tape:을 통해 Gradient 계산을 실시합니다.

weights = np.mean(grad_val, axis=(0, 1)) # gradient의 GAP으로 weight를 구합니다.
grad_cam_image = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
for k, w in enumerate(weights):
    # output의 k번째 채널과 k번째 weight를 곱하고 누적해서 class activation map을 얻습니다.
    grad_cam_image += w * output[:, :, k]
A는 Activation Laye(conv{}_block{}_out)를 통과한 feature입니다.
Activation을 통과한 N*N의 각 평면과 GAP으로 구한 alpha(weight)를 곱하여 모두 더한 후 Relu함수를 통과시킵니다.

grad_cam_image = tf.math.maximum(0, grad_cam_image)
grad_cam_image /= np.max(grad_cam_image)
grad_cam_image = grad_cam_image.numpy()
grad_cam_image = cv2.resize(grad_cam_image, (width, height))
tf.math.maximum(0, grad_cam_image)는 ReLU와 같은 효과를 나타냅니다.
이후 Grad-map의 원소값 중 최대값이 1이 되도록 스케일을 조정하고, 원본 이미지 크게로 키워줍니다.
```
> Grad-CAM과 CAM의 차이를 이해하여 코드가 작성되었습니다.

- [O] 5.코드가 간결한가요?
```python
grad_cam_conv2_images = []
for i in items:
    grad_cam_conv2_images.append(i['image'])
    for v in range(3):
        grad_cam_conv2_images.append(generate_grad_cam(cam_model, f'conv2_block{v+1}_out', i))

plt.figure(figsize=(12,12))
for k, v in enumerate(grad_cam_conv2_images):
    plt.subplot(4, 4, k+1)
    plt.imshow(v)
```
![download](https://github.com/7eta/Aiffel_CV/assets/50302638/35d94645-20b6-484f-a00a-761346ab0853)
> 반복문을 사용하여 여러 장의 Grad-CAM 이미지를 데이터마다 출력하였습니다.
