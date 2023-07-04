# ResNet과 PlainNet 비교하기

# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이효준
- 리뷰어 : 이동익


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [x] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?  
> 네 데이터셋을 분석하고 인코딩 후, 설계한 RetinaNet을 통해 object detection 테스트를 수행했습니다.
- [x] 2.주석을 보고 작성자의 코드가 이해되었나요?  
```python
'''
Anchor 파헤치기   
우리는 이러한 특징을 통해 aspect ratio를 1:2, 1:1, 2:1 세 종류로 정할 수 있고   
피사체의 거리에 따라 사진에 담기는 객체의 크기(Scale)가 다양해지는 것을 고려해 2^{1}, 2^{1/3}, 2^{2/3}으로 정할 수 있습니다.   
이 과정에서 우리는 총 9가지 anchor box 를 결정하게 됩니다.
'''
'''
FeaturePyramidNetwork 파헤치기
base-model에서 C3 -> C4 -> C5 로 진행되며, Class Activaition이 잘 된 정보(즉 주목할 대상장보)가 잘 압축되었습니다.   
즉, C5는 C3에 비해 해상도가 1/4되며 상당히 많은 정보를 잃었지만 남은 정보들은 컨볼루션을 통해 주목하고자 하는 정보들이 남아있는 샘입니다.   
base-model의 C3, C4, C5는 해상도 뿐만아니라 Channel이 같지 않기에 channel을 맞춰준다고 생각합니다. 그러한 결과가 P3, P4, P5 입니다.  
'''
```  
> 위과 같이 모델의 주요 기법에 대한 설명을 자세히 작성해주셔서 이해가 쉬웠습니다!

- [x] 3.코드가 에러를 유발할 가능성이 있나요?
>  없는 것 같습니다.
- [x] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
![image](https://github.com/domik017123/Aiffel_CV/assets/126870709/d8e54247-28cc-4c1d-a68e-cbd8be8f3562)
> 네, FPN에서 상위 레이어를 업샘플링 후 다시 더해주는 과정이 왜 필요한지 설명해주셨습니다.

- [x] 5.코드가 간결한가요?
```python
    for box, classes in zip(boxes, class_names):
        # 사람이 detect되면 stop
        if classes in ['Pedestrian', 'Person_sitting']:
            return "Stop"

        # 차가 조건에 맞으면 stop
        if classes in ['Car', 'Van', 'Truck']:
            x1,y1,x2,y2 = box
            w, h = x2-x1, y2-y1
            if w >= size_limit or h >= size_limit:
                return "Stop"
    return "Go"
```
> 네 간결합니다.
