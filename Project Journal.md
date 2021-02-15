# Project 진행 기록



### 1. VGG Encoder + SLE-GAN

- VGG Encoder: Pretrained VGG16 from torchvision.models (마지막 conv layer까지 freeze)
- Trained on Tom&Jerry Dataset exclusively. 30K images, 8 images in sequence
- Spherical Linear Interpolate 1st and 8th image, generate 8 images
  - 1st & 8th = reconstruction task (supervised)
  - 2nd~6th = generation task (Discriminator inspects fidelity)
- D: Hinge Loss + L1 reconstruction loss
  - Single Discriminator used for simplicity (for now): NO sequence discriminator.
- G: Hinge Loss + L1 reconstruction loss (1st and 8th)
- Adam with default hyper params (lr=0.001, betas=(0.9, 0.999))
- batch_size = 8 (fill in mid 6 images)
- Weights Init: N(0, 0.02) for conv and batchnorm

(1) Vanilla SLE-GAN

- Mode Collapse After 10K iterations 
  - ![image-20210212164223596](C:\Users\rin46\AppData\Roaming\Typora\typora-user-images\image-20210212164223596.png)

(2) SLE-GAN with Spectral Norm

- Mode Collapse after 10K iterations
  - ![image-20210212164319966](C:\Users\rin46\AppData\Roaming\Typora\typora-user-images\image-20210212164319966.png)



### 2. Feasibility Check: random screenshot 생성 가능한가?

- Tom&Jerry + LionKing Shuffled
- Gaussian Random Noise --> Generator
- Spectral Normalization Used

(1) latent dimension 256

- Gaussian Random Noise of dim 256
- Adam, lr=0.0002, betas=(0.5, 0.999)
- Diverge after 42K iterations

![image-20210213074619149](C:\Users\rin46\AppData\Roaming\Typora\typora-user-images\image-20210213074619149.png)

(2) latent dimension 1024

- Perhaps the data distribution is too complex, compared to CelebA-HQ or neat Anime face dataset.
- Higher dim to represent higher dim data manifold
- Adam, lr=0.0002, betas=(0.5, 0.999)
- 뭔가가 표현되는 것 같지만 여전히 실제 애니메이션 화면 생성하는 데에는 실패..ㅠㅠ



## 3. Auto-Encoder Pretraining + Fine Tuning 방법

- Lion King data 만으로 우선 학습
- L1 loss를 이용하여 Auto Encoder 모델을 학습시키고, Auto Encoder의 latent에서의 interpolation을 통해 sequence 생성
  - =당연히 충분히 매끄럽지 않고 결과 불만족스러움
  - 하지만 애니메이션의 장면들을 생성해내는데는 성공
- 추가적인 Loss Function 통해 Fine Tuning 학습
  - 아이디어 #1: L1 Loss를 통해 개별 이미지끼리 비교
  - 아이디어 #2: 3D conv 이용하는 Discriminator 통해 sequence의 매끄러움 판별
- (Issue) Image의 변화를 현재는 Fade In/Out 으로 표현 => Object의 움직임을 표현해낼 수 있도록
- (Issue) 새로운 Dataset에 대해 했을 때에도 잘 작동하는지 확인 (Validation Set Approach)