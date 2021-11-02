# Detecting and Classifying Car Damage

쏘카 이용자들이 차량 사용 전 업로드한 차량 사진들을 통해 차량의 파손 여뷰, 파손 유형(찌그러짐, 스크래치, 이격) 및 파손 위치를 찾아내는 프로젝트이다. 

## 0. Rules
- Issue 기반 Branch 생성 후 개발
- 1명의 코드 리뷰 후 Merge
- Commit Message: "[What] [State] [Date] [fixed] [issue number]" (ex: "U-Net done 21/11/02 fixed #1")
- [.py] 형식 코드 파일 업로드
- [.ipynb] 로 코드 실행
- 변수명:  lowercase with underscores (ex: epoch_num)
- 함수명: lowercase with underscores (ex: get_features())
- 클래스명: CapWords convention (ex: ToTensor)

## 1. Plan
- 완성된 model을 먼저 만들고 그 이후의 성능을 향상시킨다.
- Base model을 찾기 위해 3주 동안 개인당 3개의 모델을 구현하여, 총 12개의 모델을 비교한다 (100 epoch 기준).
- Base model을 찾으면 성능을 높이기 위해 Data Augmentation 등 data에 대한 접근과 모델의 파라미터들에 대한 접근을 한다.

## 2. Paper Review

|모델|성능|논문링크|요약|작성자|
|---|---|---|---|---|
|U-Net|?|[link](https://arxiv.org/pdf/1505.04597.pdf)|[link](https://kim123.notion.site/U-Net-Convolutional-Networks-for-Biomedical-Image-Segmentation-98ba91df596a4df197ec5e4b93fe323e)|혁|

## 3. Data

|파손유형|Test|Train|Valid|
|---|---|---|---|
|Dent|?|?|?|
|Scratch|?|?|?|
|Spacing|?|?|?|

## 4. Data Preprocessing
- 원본 이미지 추출
- Augmentation 적용
    - 예) Cutmix 

## 5. Train

## 6. Test
