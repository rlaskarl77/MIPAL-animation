# Animating Still Frames



MIPAL 2020-Winter Internship Project



### GOAL

- 두 장의 애니메이션 이미지로부터 부드러운 연속적 Sequence of Images 생성


### Formulation

1. 두 장의 이미지 X_1, X_m를 Feature Space로 인코딩 (pretrained VGG, Inception, ResNET): Z_1, Z_m

   - Raw 이미지를 Feature Map으로 만드는 것을 말함. f(X_1), f(X_m) where f: feature extractor like VGG16

2. StyleGAN의 Mapping Network를 거쳐(8-MLP) Z_1, Z_m 를 W_1, W_m으로 변환 + Path Length Regularization

   - Mapping Network: 단순한 fully connected layer 여러개를 쌓은 것. 
     - 기존 GAN 모델들은 대부분 정규분포에서 샘플링한 z로부터 바로 이미지를 만들었는데, 이 경우 태생적으로 feature entanglement 문제가 생기게 되어, 이를 학습가능한 매핑을 통해 disentangled latent 확보하려는 것! (자세한 건 styleGAN 논문 참조)

   - [StyleGAN 논문](https://arxiv.org/abs/1812.04948)
     - Mapping Network를 통한 Feature Disentanglement 제안.

   - [StyleGAN 설명](https://blog.lunit.io/2019/02/25/a-style-based-generator-architecture-for-generative-adversarial-networks/)
   - [StyleGAN 설명2](https://jayhey.github.io/deep%20learning/2019/01/16/style_based_GAN_2/)
   - [StyleGAN v2 논문](https://arxiv.org/abs/1912.04958)
     - Path Length Regularization 개념 등장 + StyleGAN 개선. 우리는 StyleGAN을 직접 training할 것은 아니니 깊게 읽진 않아도 ok.
   - [Path Length Regularization](https://paperswithcode.com/method/path-length-regularization)

3. 생성된 W_1, W_m를 (선형) interpolate ==> W_2, W_3, ..., W_{m-1} 생성

   - Linear, or Bilinear Interpolation 생각
   - [Linear Interpolation이란](https://darkpgmr.tistory.com/117)

4. 이것들을 Latent Code로 하여 Generation Task 진행

   1. (Idea #1) Pretrained StyleGAN을 Freeze-D를 이용하여 Fine-Tune 하여 사용
      - [FreezeD 논문](https://arxiv.org/abs/2002.10964)
        - 사전에 pretrained된 GAN을 새로운 데이터셋으로 fine tuning 하는 방법에 대한 논문
        - StyleGAN은 FFHQ와 같은 Natural Image로 학습되었으므로, 이를 cartoon/animation 도메인으로 적용하려면 fine tuning 필요 - 손쉽지만 성능은 약간 아쉬울 가능성
   2. (Idea #2) SLE-GAN a.k.a. FastGAN을 from the scratch 새로 학습! (모델 작고 가벼움)
      - [SLEGAN 논문](https://arxiv.org/abs/2101.04775)
        - 모델 굉장히 가벼운데 반해 높은 resolution 이미지 잘 생성하는 것으로 드러남.
        - Scratch 부터 학습해도 그리 부담되지 않는 사이즈. (개인적 흥미로 구현해놓았음)

5. 두 개의 Discriminator가 각각 개별 Image의 Fidelity, 그리고 Sequence의 자연스러움을 평가

   - Spectral Normalization 사용할 경우 그냥 GAN LOSS 사용해도 될듯(Cross-Entropy Loss)
     - [Spectral Normalization 논문](https://arxiv.org/abs/1802.05957)
       - 실제 구현은 그냥 nn.utils.spectral_norm(*) 하면 되니까 의미 정도만 이해하면 될 듯 합니다.
   - 혹은 WGAN-GP나 LSGAN Loss 사용도 고려
     - [WGAN 논문](https://arxiv.org/pdf/1704.00028.pdf)
       - Stable Training 
     - [LSGAN 논문](https://arxiv.org/abs/1611.04076)
       - 단순히 Discriminator가 참, 거짓만 보는 것이 아니라 실제 데이터 분포에 가깝도록! (Least-Squares GAN)
   - 두 개의 Discriminator를 사용하는 선행 연구
     - [MoCoGAN 논문](https://arxiv.org/abs/1707.04993)
       - Motion과 Content를 분리하여 동영상을 생성하는 법에 대한 논문. 우리와 유사하면서도 약간 접근이 다름.
       - 관심 간다면 자세히 읽어보고 어떤 방법이 더 좋을지 이야기해보는 것도 Good.
     - [StoryGAN 논문](https://arxiv.org/abs/1812.02784)
       - 문단(이야기)을 input으로 받아 여러 장의 sequence of images를 생성 (ex: 뽀로로 스토리 시각화)
   - SLE-GAN에서처럼 Discriminator에게 보조적인 Self-supervision task를 주어 보다 유의미한 시그널을 Generator에게 전달
     - Self-Supervision: 레이블 없이 스스로 학습. 보통 Reconstruction(Auto-encoder) 및 문제풀이(rotation, jigsaw...)
     - SLE-GAN에서는 Reconstruction을 시켰는데, 이는 decoder가 큰 이미지를 점차 작은 이미지로 축소해나가다가 최종적으로 참/거짓 값을 반환하게 되는데, 이 때 중간 정도 크기의 이미지를 다시 큰 이미지로 복원하는 decoder를 두어, 복원이 잘 되는지 여부를 평가하는 것.
       - 이를 통해 discriminator가 (원래 이미지를 복원할 수 있을 정도로) 좋은 feature들을 뽑고 이를 통해 참/거짓을 판별하도록 함.
       - 자세한 내용은 논문 참조!

6. 최종 평가는 (1) 정성적 - 눈으로 보기 + (2) FID Score 정도 확인해보기

   - FID(Frechet Inception Distance): 실제 데이터 분포와 생성된 샘플들의 분포간 거리 (1st, 2nd moment distance)
     - [GAN의 평가와 편향](https://velog.io/@tobigs-gm1/evaluationandbias)
       - FID 외에도 Truncation Trick 등 생성모델의 중요한 요소들에 대해 간략히/직관적으로 설명하고 있습니다.



### TODO

- 데이터셋 정하기 + 만들기
  - (ex) 디즈니/픽사 애니메이션, 미야자키 하야오/신카이마코토 등...
- 논문 열심히 읽기!
- 결과가 (당연히) 만족스럽지 않을 것을 대비하여 개선할 수 있는 아이디어 자유롭게 생각하기!
