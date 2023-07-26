# 제목 : PoseEstimation - 행동 스티커 만들기

# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이효준
- 리뷰어 : 장승우


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [❌] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?  
아쉽게도 baseline 모델로 키포인트나 스켈레톤 이미지 출력 시 에러가 발생한 것 제외하고는 잘 되있습니다.
  
- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?  

```python
def automatic_gpu_usage() :
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

automatic_gpu_usage()
```  
>  네~ 주석과 모델 이미지도 첨부해주셔서 이해가 쉬웠어요~

- [❌] 3.코드가 에러를 유발할 가능성이 있나요?

```python
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_3478/552700347.py in <module>
      2 IMG_PATH = os.path.join(PROJECT_PATH, 'test_image.jpg')
      3 
----> 4 image, keypoints = predict(IMG_PATH)
      5 draw_keypoints_on_image(image, keypoints)
      6 draw_skeleton_on_image(image, keypoints)

/tmp/ipykernel_3478/428616262.py in predict(image_path)
      6     inputs = tf.expand_dims(inputs, 0)
      7     outputs = model(inputs, training=False)
----> 8     heatmap = tf.squeeze(outputs[-1], axis=0).numpy()
      9     kp = extract_keypoints_from_heatmap(heatmap)
     10     return image, kp

InvalidArgumentError: Can not squeeze dim[0], expected a dimension of 1, got 64 [Op:Squeeze]
```
>  에러가 발생한 부분을 제외하고는 특이점은 없었어요~

- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
```python
model_name = 'baseline'

if model_name == 'hourglass':
    PROJECT_PATH = os.getenv('HOME') + '/aiffel/mpii'
    MODEL_PATH = os.path.join(PROJECT_PATH, 'models/model-v0.0.1-epoch-2-loss-1.3072.h5')
    model = StackedHourglassNetwork(
        input_shape=(256, 256, 3), num_stack=4, num_residual=1,
        num_heatmap=16)
    model.load_weights(MODEL_PATH)  # 본인이 학습한 weight path로 바꿔주세요. 
elif model_name == 'baseline':
    PROJECT_PATH = os.getenv('HOME') + '/aiffel/CV-PoseEstimation'
    MODEL_PATH = os.path.join(PROJECT_PATH, 'models/model-epoch-5-loss-0.2912.h5')
    model = Simplebaseline(input_shape=(256, 256, 3))
    model.load_weights(MODEL_PATH)  # 본인이 학습한 weight path로 바꿔주세요.     
else:
    pass

```
> 네~ 질문한 내용에 대해 잘 설명해주셨어요~

- [⭕] 5.코드가 간결한가요?
```python
def _make_deconv_layer(num_deconv_layers):
    seq_model = tf.keras.models.Sequential()
    for i in range(num_deconv_layers):
        seq_model.add(tf.keras.layers.Conv2DTranspose(256, kernel_size=(4,4), strides=(2,2), padding='same'))
        seq_model.add(tf.keras.layers.BatchNormalization())
        seq_model.add(tf.keras.layers.ReLU())
    return seq_model
def Simplebaseline(input_shape=(256, 256, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet')(inputs)
    x = _make_deconv_layer(3)(x)
    out = tf.keras.layers.Conv2D(16, kernel_size=(1,1), padding='same')(x)

    model = tf.keras.Model(inputs, out, name='simple_baseline')
    return model
```
> 네~ 기능에 따라 잘 분류되어있어요~
