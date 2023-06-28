# Augmentation

# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이효준
- 리뷰어 : 김설아


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [x] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?  
  
- [x] 2.주석을 보고 작성자의 코드가 이해되었나요?  
```python
 데이터셋에 augmentation 적용하는 함수
def apply_normalize_on_dataset(ds, is_test=False, batch_size=16, with_aug=False, with_cutmix=False, with_mixup=False):
    # 위에서 만든 normalize 함수를 병렬로 map 하기
    ds = ds.map(
        normalize_and_resize_img,   # 기본적인 전처리 함수 적용
        num_parallel_calls=2        # 병렬처리할 때 가용할 CPU 코어 개수
    )
    # apply base augmentation
    if not is_test and with_aug:
        ds = ds.map(
            augment
        )
        
    # split dataset into batches of batch_size    
    ds = ds.batch(batch_size)
    
    
    if not is_test and with_cutmix:      # apply CutMix augmentation
        ds = ds.map(
            cutmix,
            num_parallel_calls=2
        )
    elif not is_test and with_mixup:     # apply MixUP augmentation
        ds = ds.map(
            mixup,
            num_parallel_calls=2
        )
    else:                                # apply one-hot encoding
        ds = ds.map(
            onehot,
            num_parallel_calls=2
        )

    if not is_test:
        ds = ds.repeat()
        ds = ds.shuffle(200)
        
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds
```  
>  함수 내부에 작동하는 순서에 따라 자세한 주석으로 이해하기 수월했습니다

- [x] 3.코드가 에러를 유발할 가능성이 있나요?
```python
if os.path.isfile('./resnet50_no_aug.h5'):
    history_resnet50_no_aug = pickle.load(open('./resnet50_no_aug.pkl', 'rb'))
    # keras.models.load_model('./resnet50_no_aug.h5')
else:
    history_resnet50_no_aug = resnet50.fit(
        ds_train_no_aug, # augmentation 적용하지 않은 데이터셋 사용
        steps_per_epoch=int(ds_info.splits['train'].num_examples/16),
        validation_steps=int(ds_info.splits['test'].num_examples/16),
        epochs=EPOCH,
        validation_data=ds_test_no_aug, #ds_test_no_aug
        verbose=1,
        use_multiprocessing=True,
    )
    resnet50.save('./resnet50_no_aug.h5')
    pickle.dump(history_resnet50_no_aug.history, open('./resnet50_no_aug.pkl', 'wb')
```
>  history를 저장하며 진행하여 에러를 방지하셨습니다.
- [x] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
```python
def get_clip_box(image_a, image_b):
    # image.shape = (height, width, channel)
    image_size_x = image_a.shape[1]
    image_size_y = image_a.shape[0]
    
    # get center of box
    x = tf.cast(tf.random.uniform([], 0, image_size_x), tf.int32)
    y = tf.cast(tf.random.uniform([], 0, image_size_y), tf.int32)
    
    # 새로 수정한 내용
    alpha = 1
    lam = np.random.beta(alpha, alpha)

    # get width, height of box
    width = tf.cast(image_size_x*tf.math.sqrt(1-lam), tf.int32)
    height = tf.cast(image_size_y*tf.math.sqrt(1-lam), tf.int32)
    
    # clip box in image and get minmax bbox
    x_min = tf.math.maximum(0, x-width//2)
    y_min = tf.math.maximum(0, y-height//2)
    x_max = tf.math.minimum(image_size_x, x+width//2)
    y_max = tf.math.minimum(image_size_y, y+height//2)
    
    return x_min, y_min, x_max, y_max
```
> 논문에서 combination ratio lamda의 공식을 이해하고 alpha를 수정하며 모델 학습을 비교하셨습니다.

- [x] 5.코드가 간결한가요?
```python
cutmix_alpha_1 = pickle.load(open('./resnet50_cutmix_aug.pkl', 'rb'))
cutmix_alpha_1_dot_5 = pickle.load(open('./resnet50_cutmix_aug_a1.5.pkl', 'rb'))
cutmix_alpha_1_with_aug = pickle.load(open('./resnet50_cutmix_aug_more.pkl', 'rb'))
```
> history를 저장하여 시도한 모델들에 대한 정리만 보일 수 있게 해주셨습니다.
