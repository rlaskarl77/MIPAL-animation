# Project 개요

최초에 내가 기획한 프로젝트 아이디어는 특정 input image를 받았을 때, 원하는 스타일로 바꿔주는 style transfer를 생각했다. 근데 이게 지금 와서 생각해보니까 기존에 잘 구현된 코드가 있어서 코드 복붙만 해서 학습 데이터 다시 들고 하면 되는거라 프로젝트에서 바라는 "새로운 것"의 취지와는 잘 어울리지 않았다.

결과적으로 선택된 프로젝트 아이디어는 "Animating Images by Frame Interpolation" 이다. 자세한 내용은 [여기](https://github.com/reyllama/MIPAL-animation)에 있으니 이거 참고하면 된다.

내가 생각한 이 프로젝트의 아이디어는 다음과 같다.

> 애니메이션 생성에 있어 프레임 사이의 연속된 동작 사이에는 **어떤 특징**이 있어서 이를 통해 중간 N개의 프레임을 생성해낼 수 있을 것이다.

그러면 중간 N개의 프레임을 생성해내기 위해서는 무엇이 필요한가? 에 대한 고민의 결과는 "딥러닝 모델의 `latent feature`의 `linear interpolation`을 통해 가능하다!"이고 우리 팀원 모두 이러한 방향성을 가지고 논문을 찾아보고 실험을 진행했다.

논문 찾아보는 건 이것저것 찾아보긴 했지만 아직 내 코딩 수준이 허접이라 간단하고 강력한 모델을 찾아보고자 했다.

# 모델1 : [Project-AE slerp](https://www.kaggle.com/chaerink/project-ae-slerp)의 응용
아이디어: AutoEncoder의 구조를 이용해 latent code를 생성하는 모델을 만들어낸 것을 기반으로 Sequence 8장의 이미지 전체를 L1 loss로 직접 비교하는 loss function을 이용해 새롭게 fine-tuning하도록 해보자!

학습 결과 몇 가지 현상들을 발견했다. ([캐글구현](https://www.kaggle.com/wonseok1017/project-ae-slerp-l1-loss?scriptVersionId=54374559))

- 학습을 위해 사용한 8장의 image sequence가 하나의 액션으로 구성되지 않는다
- 학습결과를 판단하는 기준으로 L1 loss를 사용했기 때문에 이미지의 linear interpolation이 발생하는 현상을 캐치하지 못하는 것같다.
- Input frame을 A1 ~ A8이라고 한다면 생성된 이미지 A1' ~ A8'를 비교할 때, 처음과 마지막 frame은 좀 더 원본에 가깝게 복원될 수 있도록 hyperparameter를 선정했다. 하지만 나중에는 이는 별 의미가 없다는 것을 깨달았다. (ver.1부터 ver.5까지는 hyperparameter의 변화를 준 코드들이다.)
- 학습을 진행할 때 Dataloader에서 Shuffle=False로 설정해 연속된 이미지들을 학습할 수 있게 되었지만, 반대로 문제점은 무조건 처음부터 8장씩만 학습을 진행하기 때문에 특정한 순서가 발생한다는 것이다.

# 모델2 : 학습 dataset 늘려보기

위의 현상들 중에 4번째의 순서대로 학습하는 것을 방지하기 위해 Dataloader를 사용하지 않고 random seed를 통해 8장의 Image Sequence를 발생하는 문장을 추가했다. 

```python
start = np.random.randint(36000-8)
data = torch.empty(8,3,256,256)
for i in range(8):
	data[i] = dataset[start+i]
```
조사를 했다면 좀 더 깔쌈한 코드를 찾을 수 있었을 것으로 예상되지만 그정도의 코드 실력과 정보검색능력은 없어서 조잡하지만 저런 코드를 사용했다. ([캐글구현](https://www.kaggle.com/wonseok1017/project-ae-slerp-l1-loss?scriptVersionId=54444200))

학습 결과 다음과 같은 특징을 발견했다.

- 역시나 8장의 이미지가 연속되는지는 판단해 학습하는 방법이 없다.
- Latent vector의 선형보간이 frame interpolation으로 이어지지 않는다면 애초에 가정에서 무언가가 잘못된 것인가?

이를 해결해보기 위해 새로운 loss를 만들어보았다.

# 모델3 : loss function 변경 & 학습 dataset 선별

기존 모델에서의 `loss`는 다음과 같이 설정했다. (`data`는 원본, `zs`는 생성 이미지이다.)
```python
loss = alpha_lerp * (recon_loss(data[0],zs[0]) + recon_loss(data[-1],zs[-1]))
for i in range(7):
	loss += recon_loss(data[i],zs[i])
```

이를 다음과 같이 변경했다.
```python
loss = alpha_lerp * (recon_loss(data[0],zs[0]) + recon_loss(data[-1],zs[-1]))
for i in range(1,7):
	loss += recon_loss(data[i],zs[i]) + alpha_latent * recon_loss(z_in[i],z_out[i])
```

그리고 8장의 이미지가 연속된지 아닌지를 판단하는 함수인 `not_sequence`를 새로 정의했다. 이는 `vgg16` 모델에 첫장과 끝장을 넣어 feature의 차이를 비교함으로서 sequence를 판단해주는 거다.
```python
def not_sequence(img1, img2, check=False):
    temp = torch.cat([img1.unsqueeze(0), img2.unsqueeze(0)], dim=0)
    temp = temp.data.float()
    temp = temp.to(device)
    out = vgg(temp)
    if check:
        print(abs(out[0].mean().detach().item() - out[1].mean().detach().item()))
    if abs(out[0].mean().detach().item() - out[1].mean().detach().item()) > 0.01:
        return True
    else:
        return False
```

최종 결과는 다음과 같다. ([캐글구현](https://www.kaggle.com/wonseok1017/project-ae-slerp-l1-loss?scriptVersionId=54783106))

결과를 한 줄로 요약하면 다음과 같다.
> 뭔 짓을 해도 L1 loss를 사용한 이상 중간 frame이 아닌 image interpolation 현상을 피하지 못한다.

# 모델4 : 새로운 모델의 생성
모델1부터 모델3까지는 [Project-AE](https://www.kaggle.com/chaerink/project-ae)를 기반으로 한 모델이다. 근데 loss function을 바꿔가면서 실험해봐도 발전이 보이지 않아 아예 새로운 모델을 구축할 필요성을 느꼈다. 새로운 딥러닝 모델을 구현하는데 있어 필요한 참고 모델이 필요했고, 최근에 진행한 Generative Study에서 나온 [InfoGAN](https://arxiv.org/pdf/1606.03657.pdf)을 참고해보기로 했다. 여기에 몇 가지 수정사항을 추가해 새로운 모델을 만들었다. ([캐글구현](https://www.kaggle.com/wonseok1017/project-wonseokoriginal?scriptVersionId=54964443))

- Batch Normalization 대신 Spectral Normalization 사용
- Loss function에 latent vector의 L1 loss와 생성된 이미지의 MS-SSIM loss를 이용

100000번의 iteration을 진행하면서 발생한 문제점은 다음과 같았다.
> 아무리 학습을 진행시켜도 뭔가 잘 되는 듯한 느낌이 안들고 회색 이미지가 생성된다.

이 부분을 해결해보고자 추가적으로 100000번 더 돌렸는데 그래도 문제는 여전하다.

그래서 좀 빠르게 이 모델은 잘못 만들었다고 판단하여 새로운 모델을 만드려던 와중 뭐가 문제였는지 파악했다. 가중치 초기화를 안했다. 그래서 backpropagation에서 gradient가 전달되지 않은 것같다.

# 모델5 : 모델4에서 오류 수정
모델4에서 가중치 초기화를 이용하니까 50000번 학습을 진행해도 좋은 성능을 내는 것을 확인할 수 있었다(`lr=1e-3`). 여기서 `lr=1e-4`로 줄인 수 20000번 추가 학습을 진행해 모델을 학습시켰다. ([캐글구현](https://www.kaggle.com/wonseok1017/project-wonseokoriginal?scriptVersionId=55171103))

학습에 사용한 loss function은 다음과 같다. (`1 - ssim_loss(data,zs)` 부분은 그냥 MS-SSIM loss로 보면 된다. 실제로 실험해본 결과 동일한 이미지에서 `ssim_loss=1`이 도출되었다.
```python
loss_ssim = 1 - ssim_loss(data,zs)
loss_recon = recon_loss(data[0],zs[0]) + recon_loss(data[-1],zs[-1])
loss_latent = 0
for i in range(1,7):
	loss_recon += recon_loss(data[i],zs[i])
	loss_latent += recon_loss(z_in[i],z_out[i])

loss = 0.1 * loss_ssim + loss_recon + 0.1 * loss_latent
```
학습 결과는 다음과 같다.

- 모델1~3보다는 얕은 모델을 만들었기 때문에 이미지가 좀 약하게 생성되는 경향성이 발견된다.
- 실제로 중간 frame에 해당하는 이미지와 매우 유사한 이미지를 생성하는 경우가 간혹 존재하지만, 대부분의 경우에는 image interpolation과 큰 차이가 보이지 않는다.

# 해결하지 못하거나 추가적으로 생각난 아이디어
정보검색능력이나 프로그래밍 실력이 상대적으로 떨어져서 생각을 완벽하게 구현하지 못한 점이 아쉬웠지만, 그래도 이번 프로젝트를 진행하면서 해결하지 못한 부분들이나 이 프로젝트의 연장선상으로 진행할 수 있을만한 아이디어들을 적어두면 나중에 이걸 다시 봤을 때 또 다른 영감을 얻을 수 있을까해서 적어둔다.

- Optical Flow에 대한 정보를 프로젝트 막바지에 접하게 되었는데 이게 정확히 우리 프로젝트와 일치하는 task라고 생각된다. Optical flow를 좀 응용하거나 경량화해서 우리의 프로젝트를 발전시킬 수 있지 않을까 생각한다.
- 우리의 프로젝트는 input frame `X(1)`, `X(n)`을 바탕으로 `X(2)`, ... , `X(n-1)`를 생성하는 거다. 그러면 만약 latent space의 linear interpolation을 통해 중간 단계의 이미지를 생성하는 게 성공하면 동일한 방법으로 `X(n+1)`, `X(n+2)`, ....을 생성해보면 어떤 결과가 나올지 기대된다. 
- ex) 사람이 오른쪽에서 왼쪽으로 이동하는 프레임을 가지고 애니메이션을 생성했을 때, 이후 영상을 계속 생성하면 이 사람은 프레임 바깥으로 탈출할 것인가?
- 현재 프로젝트의 아이디어를 기반으로 하면 물체의 `형상`과 `위치`에 대한 정보를 얻을 수 있다고 생각한다. 그렇기 때문에 선형보간을 했을 때, `형상`의 변화와 `위치`에 대한 정보가 중간값을 얻을 수 있을 텐데, 그러면 중간 위치는 얻을 수 있지만, 그 사이의 `액션`정보는 학습하는데 어려움이 있지 않을 까 생각한다. [ex) 자전거를 타고 이동하는 움직임] 그렇다면 이러한 현상을 발견하는 것도 꽤나 재미있는 과정이 될 것 같다.
- 실제 애니메이션을 frame 단위로 보면 인접한 두 frame 사이의 차이는 크게 발생하지 않으니 중간 frame을 상대적으로 정확하게 생성할 수 있을 것으로 예상된다. 그렇다면 우리의 프로젝트는 기존에 존재하는 애니메이션의 fps를 높여줄 수 있는 강력한 tool이 되지 않을까 생각한다.

# 프로젝트 후기
딥러닝 공부한지 2달밖에 안된 나로써는 너무나 많이 빡센 과정이었지만, 그래도 말로 표현할 수 없는 무언가를 얻어가지 않았다 생각된다. 좀 더 열심히 할 수 있었을 것 같긴 하지만 그래도 꽤나 많은 공부가 되었다.

나중에는 내가 직접 기획한 프로젝트를 구현하는 날이 왔으면 좋겠다.
