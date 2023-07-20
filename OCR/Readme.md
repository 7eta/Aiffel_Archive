# 제목 : End-to-End OCR

# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이효준
- 리뷰어 : 이동익


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?  
> 라벨 인코딩 후 recognition 모델학습이 정상적으로 진행되었고 이를 이용해 ocr이 잘 수행되었습니다.
- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?  
```python
'''
뒤에 계속 999999가 나와요..
'''
def decode_predict_ctc(out, chars = TARGET_CHARACTERS):

    results = []
    # keras.backend.ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1)
    # y_pred에 output이 들어가고, shape은 (1, 24, 38)
    # input_length에는 shape이 (1, )인 24로 초기화된(1 * 24(out.shape[1])) ndarray가 입력됨
    # top_paths=1인 경우 가능성 높은 경로 1개를 반환함
    # 가장 중요한건데, 빈 레이블은 -1이 return됨
    indexes = K.get_value(
        K.ctc_decode(
            out, input_length=np.ones(out.shape[0]) * out.shape[1],
            greedy=False , beam_width=5, top_paths=1
        )[0][0]
    )[0]
    # 코드 생략
'''
맨 위에서 변환할 문자를 정의할때 TARGET_CHARACTERS = ENG_CHAR_UPPER + NUMBERS을 통해 ABCD~789 순서 있게 된다.
이때 -1은 TARGET_CHARACTERS의 맨 뒤에서 첫번째 숫자 9를 가르키게 되어 빈 레이블(-1)이 9로 디코딩 된다.
'''
```  
>  네, 특히 출력 뒷부분에 999가 나오는 부분을 해결하시고 설명해주셔서 저도 도움을 받았습니다.

- [❌] 3.코드가 에러를 유발할 가능성이 있나요?
>  없는 것 같습니다.
- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
```python
def _get_img_label(self, index):
    # 내용
    # 일부
    # 생략
    target_width = min(int(width*self.img_size[1]/height), self.img_size[0])
    target_img_size = (target_width, self.img_size[1])
    img = np.array(img.resize(target_img_size)).transpose(1,0,2)
##img_size=(100,32) 으로 width와 height을 일정한 크기로 맞춰줍니다.
```
> 네 , 위와같이 구현과정별로 상세히 설명을 작성해주셨습니다. 

- [⭕] 5.코드가 간결한가요?
```python
img_pil, cropped_img = detect_text(SAMPLE_IMG_PATH)
display(img_pil)

for _img in cropped_img:
    recognize_img(_img)
```
> 간결합니다!
