# 제목 : Face Detection

# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이효준
- 리뷰어 : 이성주


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?  
  |평가문항|상세기준|평가|
  |--------|--------|--------|
  |1. multiface detection을 위한 widerface 데이터셋의 전처리가 적절히 진행되었다.|tfrecord 생성, augmentation, prior box 생성 등의 과정이 정상적으로 진행되었다.|![image](https://github.com/7eta/Aiffel_CV/assets/29011595/9230b6f8-a4ef-4af6-8911-763643db5b54) tfrecord 생성, augmentation, prior box 생성 등의 과정이 정상적으로 진행되었습니다.
  |2. SSD 모델이 안정적으로 학습되어 multiface detection이 가능해졌다.|inference를 통해 정확한 위치의 face bounding box를 detect한 결과이미지가 제출되었다.| ![image](https://github.com/7eta/Aiffel_CV/assets/29011595/0204c5b3-baab-4e5e-afd0-58cbcbee282b) 학습이 잘 진행되고 있습니다...|
  |3. 이미지 속 다수의 얼굴에 스티커가 적용되었다.|이미지 속 다수의 얼굴의 적절한 위치에 스티커가 적용된 결과이미지가 제출되었다.|![image](https://github.com/7eta/Aiffel_CV/assets/29011595/17a89b47-1ff8-4ddb-9d04-3ea1e2c2028d) 학습이 다 진행되면 이쁜 왕관이 될것 같습니다.|


- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?  
```python
def put_stickers(img, boxes, prior_index, img_sticker):
    img_height = img.shape[0]
    img_width = img.shape[1]

    x_min = int(boxes[prior_index][0] * img_width)
    y_min = int(boxes[prior_index][1] * img_height)
    x_max = int(boxes[prior_index][2] * img_width)
    y_max = int(boxes[prior_index][3] * img_height)
    
    w = x_max - x_min
    h = w//4
    
    img_sticker = cv2.resize(img_sticker, (w, h*2))
    
    # 스티커 사이즈만큼 공간 잡고
    sticker_area = img_raw[y_min-h:y_min+h, x_min:x_min+w] 
    
    # 그 부위에서 스티커를 제외한 부분은 기존 사진이 노출되도록 설정
    img_raw[y_min-h:y_min+h, x_min:x_min+w] = np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
```  
>  주석을 보고 코드 이해가 잘 되었습니다.

- [❌] 3.코드가 에러를 유발할 가능성이 있나요?
> 없습니다.

- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
> 넵 코드를 제대로 이해하고 작성하였습니다.

- [⭕] 5.코드가 간결한가요?
```python
for prior_index in range(len(boxes)):
    put_stickers(img_raw, boxes, prior_index, img_sticker)
```
> 함수화 처리로 코드를 간결하게 작성하였습니다.

