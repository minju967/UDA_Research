# 석사 졸업 연구

Unsupervised Domain Adaptation

## Folder List
### 1. Dataset

  Domain Adaptation data 파일 저장 및 create dataset

    [1] Office-Home

    [2] VisDA-2017

    [3] DomainNet

    [4] Office-31

### 2. Loss Function

  Experiment for Proposed loss function

### 3. Proposed

  Experiment for Proposed method & architecture
  1. DRANet_mine
      DRANet에서 출력한 content, style latent사이의 Mutual Information을 측정하기 위해 
      
      [MINE: Mutual Information Neural Estimator](https://github.com/gtegner/mine-pytorch)를 적용해본다.
      
      23-5-13: statistics network 적용 및 loss 변형.
               MINE의 경우, 두 변수 X, Z사이의 distance가 loss가 된다 (-t + second_term). 
               
               DRANet의 경우, 두 변수 Content, Style은 distance가 클수록 서로간의 상호정보량이 적다는 것을 의미한다.
               
               따라서, distanc가 낮을수록 높은 loss값을 높을수록 낮은 loss값을 계산한다.
      
### 4. Experiment 정리

  실험내용 및 결과 정리
