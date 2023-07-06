# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이효준
- 리뷰어 : 이동익


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [x] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?   
  1. U-Net을 통한 세그멘테이션 작업이 정상적으로 진행되었는가?   
![image](https://github.com/domik017123/Aiffel_CV/assets/126870709/f721f67b-6c4d-4386-9bdc-83af5ac0883a)   
  도로 검출이 잘 진행되었습니다.
  2. U-Net++ 모델이 성공적으로 구현되었는가?
![image](https://github.com/domik017123/Aiffel_CV/assets/126870709/5d3560c0-33fa-4e24-951a-cf6339515b27)   
모델을 잘 설계하셨고 IoU 0.87의 결과가 나왔습니다.   
  3. U-Net과 U-Net++ 두 모델의 성능이 정량적/정성적으로 잘 비교되었는가?
![image](https://github.com/domik017123/Aiffel_CV/assets/126870709/25ad38b8-7eef-4a30-9c8f-9af0019a0d84)   
loss와 accuracy를 통한 비교가 이루어졌습니다.   

  
- [x] 2.주석을 보고 작성자의 코드가 이해되었나요?
![image](https://github.com/domik017123/Aiffel_CV/assets/126870709/d51aaddb-280e-40f5-8735-310ff3f307ad)   

```python
# U-Net++ 구현
def Nest_Net(input_shape=(224, 224, 3), num_class=1, deep_supervision=False):
    nb_filter = [64,128,256,512,1024]
    
    img_input = Input(input_shape, name='main_input')
    
    # X_0,0
    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)
    
    # X_1,0
    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)
    
    # X_0,1
    # Upsampling
    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    # Skip Connection
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=3)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    
    # output
    nesnet_output_1 = Conv2D(num_class, (1,1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nesnet_output_2 = Conv2D(num_class, (1,1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nesnet_output_3 = Conv2D(num_class, (1,1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nesnet_output_4 = Conv2D(num_class, (1,1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)
    
    # Supervision 여부에 따른 출력 결정
    if deep_supervision:
        model = Model(inputs=img_input, outputs=[nesnet_output_1,
                                              nesnet_output_2,
                                              nesnet_output_3,
                                              nesnet_output_4])
    else:
        model = Model(inputs=img_input, outputs = [nesnet_output_4])
        
    return model
```  
시각화된 UNet의 구조와 일치하는 레이어 번호로 주석을 달아 주셔서 모델 구조 파악이 용이했습니다.

- [x] 3.코드가 에러를 유발할 가능성이 있나요?
없는 것 같습니다.
- [x] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
```python
class KittiGenerator(tf.keras.utils.Sequence):
    '''
    KittiGenerator는 tf.keras.utils.Sequence를 상속받습니다.
    우리가 KittiDataset을 원하는 방식으로 preprocess하기 위해서 Sequnce를 커스텀해 사용합니다.
    '''
    def __init__(self, 
                 dir_path, 
                 batch_size=4, 
                 img_size=(224, 224, 3), 
                 output_size=(224, 224), 
                 is_train=True, 
                 augmentation=None):
        '''
        dir_path: dataset의 diLrectory path입니다.
        batch_size: batch_size입니다.
        img_size: preprocess에 사용할 입력이미지의 크기입니다.
        output_size: ground_truth를 만들어주기 위한 크기입니다.
        is_train: 이 Generator가 학습용인지 테스트용인지 구분합니다.
        augmentation: 적용하길 원하는 augmentation 함수를 인자로 받습니다.
        '''

        # load_dataset()을 통해서 kitti dataset의 directory path에서 라벨과 이미지를 확인합니다.
        self.data = self.load_dataset()
```
generator를 이용해 전처리과정을 진행하는 과정을 이해하시고 상세히 기술해주셔서 많은 도움이 되었습니다!

- [x] 5.코드가 간결한가요?
```python
def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

    act = 'elu'
    dropout_rate = 0.5

    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
#     x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
#     x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x
```
standard_unit을 선언하여 불필요한 반복이 줄었습니다.
