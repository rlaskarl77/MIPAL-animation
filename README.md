# Animating Images by Frame Interpolation

## 0. Goal

- Create a short sequence of continuous images from two still images using frame interpolation.
- Generated sequence should be animated in a sense that is natural to human perception.

## 1. Dataset

- Use screen captured Disney Animation Lionking 1, 2 dataset, each consisting of 36000 and 51000 still frames.
- Sequence of length 8 is used. Those that include abrupt scene changes are excluded using pretrained VGG16 feature maps.
- All images were resized to 256x256 scale and basic normalization was conducted.

## 2. Method

### Encoder-Decoder Feature Extractor

- An encoder-decoder network extracts structural features from given images.
- Skip Layer Excitement Module is used for decoder, and multiple residual blocks are used for encoder to enhance expressivity of our model.
- MS-SSIM loss and L1 reconstruction loss were used to extract oveall semantic and circumvent generation ghosting objects.
- 256x256x3 images were compressed to 32x32x128 feature maps by the encoder, and mapped back to original scale.

### U-net based Refinement Network

- Output from Feature Extractor Network often suffers in colors and sharp shapes, as they focus on overall structural cues.
- Use improved Pix2Pix model as refinement network to fine tune our output.
- Spectral normalization was used to stabilize training process, and strided convolutions were replaced by nearest neighbor upsampling + resolution preserving convolution to evade so-called 'checker board effect'.
- Following the precedure of original Pix2Pix, adversarial loss and L1 reconstruction loss were combined to form second stage network objective.

### End-to-End Training

- Put first stage feature extractor and second stage refinement network together to create our final *Generator* network.
- 4 Losses, first stage L1 loss, MS-SSIM loss, second stage L1 loss and adversarial losses are all incorporated in multi-objective loss function.
- Generator trained at lr=1e-4, and discriminator trained at lr=5e-4 following the tradition of TTUR.

## 3. Experiments

### Qualitative Results

![gif1](https://github.com/reyllama/MIPAL-animation/blob/master/output/G1_1093.gif | width=96)
![gif1](https://github.com/reyllama/MIPAL-animation/blob/master/output/G1_1265.gif | width=96)
![gif1](https://github.com/reyllama/MIPAL-animation/blob/master/output/G1_1357.gif | width=96)
![gif1](https://github.com/reyllama/MIPAL-animation/blob/master/output/G1_1429.gif | width=96)
![gif1](https://github.com/reyllama/MIPAL-animation/blob/master/output/G1_1473.gif | width=96)
![gif1](https://github.com/reyllama/MIPAL-animation/blob/master/output/G1_1545.gif | width=96)
![gif1](https://github.com/reyllama/MIPAL-animation/blob/master/output/G1_1573.gif | width=96)
![gif1](https://github.com/reyllama/MIPAL-animation/blob/master/output/G1_1709.gif | width=96)
