# Animating Still Frames



MIPAL 2020-Winter Internship Project



### GOAL

- 두 장의 애니메이션 이미지로부터 부드러운 연속적 Sequence of Images 생성


### Formulation

1. 두 장의 이미지 X_1, X_m를 Feature Space로 인코딩 -> Z_1, Z_m
   - pretrained VGG, Inception, ResNET 등 활용.
   - Raw 이미지를 Feature Map으로 만드는 것을 말함. 
   - f(X_1), f(X_m) where f: feature extractor like VGG16

2. StyleGAN의 Mapping Network를 거쳐(8-MLP) Z_1, Z_m 를 W_1, W_m으로 변환 + Path Length Regularization

   - Mapping Network: 단순한 fully connected layer 여러개를 쌓은 것. 
     - 기존 GAN 모델들은 대부분 정규분포에서 샘플링한 z로부터 바로 이미지를 만들었는데, 이 경우 태생적으로 feature entanglement 문제가 생기게 됨
     - StyleGAN: 학습가능한 mapping을 통해 disentangled latent 확보하고자 함

   - [StyleGAN 논문](https://arxiv.org/abs/1812.04948)
     - Mapping Network를 통한 Feature Disentanglement 제안.
     - disentanglement: [사전적 의미] 얽힌 것을 품 / 해방시킴.
     - latent space가 linear한 구조를 가지게 되어서 하나의 latent vector, z를 움직였을 때, 정해진 어떠한 하나의 특성이 변경되게 마들고자 하는 것
     - ex) latent vector, z의 specific한 값을 변경했을 때 생성되ㅡㄴ 이미지 하나의 ㅌ그성들(머리길이, 성별, ...)만 영향을 주게 만듦 
     - > "이 model의 latent space는 disentanglement"

   - [StyleGAN 설명](https://blog.lunit.io/2019/02/25/a-style-based-generator-architecture-for-generative-adversarial-networks/)
      - latent vector(z) -> Normalization -> mapping network -> w
      - mapping network: 8개의 FC layer -> Nonlinear f -> latent vector 간 상관관계 ↓ -> disentanglement 
      
   - [StyleGAN 설명2](https://jayhey.github.io/deep%20learning/2019/01/16/style_based_GAN_2/)
   
   - [StyleGAN v2 논문](https://arxiv.org/abs/1912.04958)
     - Path Length Regularization 개념 등장 + StyleGAN 개선. 
     
   - [Path Length Regularization](https://paperswithcode.com/method/path-length-regularization)

3. 생성된 W_1, W_m를 (선형) interpolate => W_2, W_3, ..., W_(m-1) 생성
   - Linear or Bilinear Interpolation 생각
   - [Linear Interpolation이란] (https://darkpgmr.tistory.com/117)
   - Video Frame Interpolation

4. 이것들을 Latent Code로 하여 Generation Task 진행
   Idea)
   1. Pretrained StyleGAN을 Freeze-D를 이용하여 Fine-Tune 하여 사용
      - [FreezeD 논문](https://arxiv.org/abs/2002.10964)
        - pretrained된 GAN을 새로운 데이터셋으로 fine tuning 하는 방법에 대한 논문
        - StyleGAN은 FFHQ와 같은 Natural Image로 학습되었으므로, 이를 cartoon/animation 도메인으로 적용하려면 fine tuning 필요
        - 손쉽지만 성능은 약간 아쉬울 가능성
        
   2. SLE-GAN a.k.a. FastGAN을 from the scratch 새로 학습 (모델 작고 가벼움)
      - [SLEGAN 논문](https://arxiv.org/abs/2101.04775)
        - 모델 굉장히 가벼운데 반해 높은 resolution 이미지 잘 생성하는 것으로 드러남.
        - Scratch 부터 학습해도 그리 부담되지 않는 사이즈.

5. 두 개의 Discriminator가 각각 개별 Image의 Fidelity, 그리고 Sequence의 자연스러움을 평가
   - Spectral Normalization 사용할 경우 GAN LOSS(Cross-Entropy Loss)
     - [Spectral Normalization 논문](https://arxiv.org/abs/1802.05957)

   - 혹은 WGAN-GP나 LSGAN Loss 사용
     - [WGAN 논문](https://arxiv.org/pdf/1704.00028.pdf)
       - Stable Training 
     - [LSGAN 논문](https://arxiv.org/abs/1611.04076)
       - (Least-Squares GAN)
       - 단순히 Discriminator가 참, 거짓만 보는 것이 아니라 실제 데이터 분포에 가깝도록
   
   - 두 개의 Discriminator를 사용하는 선행 연구
     - [MoCoGAN 논문](https://arxiv.org/abs/1707.04993)
       - Motion과 Content를 분리하여 동영상을 생성하는 법에 대한 논문. 
     - [StoryGAN 논문](https://arxiv.org/abs/1812.02784)
       - 문단(이야기)을 input으로 받아 여러 장의 sequence of images를 생성 (ex: 뽀로로 스토리 시각화)
       
   - SLE-GAN에서처럼 Discriminator에게 보조적인 Self-supervision task를 주어 보다 유의미한 시그널을 Generator에게 전달
     - Self-Supervision: 레이블 없이 스스로 학습. 
     - 보통 Reconstruction(Auto-encoder) 및 문제풀이(rotation, jigsaw...)
     
     - SLE-GAN에서는 Reconstruction을 시켰는데, 이는 decoder가 큰 이미지를 점차 작은 이미지로 축소해나가다가 최종적으로 참/거짓 값을 반환하게 되는데, 이 때 중간 정도 크기의 이미지를 다시 큰 이미지로 복원하는 decoder를 두어, 복원이 잘 되는지 여부를 평가하는 것.
       - 이를 통해 discriminator가 (원래 이미지를 복원할 수 있을 정도로) 좋은 feature들을 뽑고 이를 통해 참/거짓을 판별하도록 함.
       
6. 최종 평가는 (1) 정성적 - 눈으로 보기 + (2) FID Score 정도 확인해보기

   - FID(Frechet Inception Distance): 실제 데이터 분포와 생성된 샘플들의 분포간 거리 (1st, 2nd moment distance)
     
     - [GAN의 평가와 편향](https://velog.io/@tobigs-gm1/evaluationandbias)
    
