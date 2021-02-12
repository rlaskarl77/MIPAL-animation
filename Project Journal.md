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