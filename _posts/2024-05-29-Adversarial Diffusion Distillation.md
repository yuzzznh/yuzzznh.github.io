---
layout: post
title: Adversarial Diffusion Distillation
description: 
sitemap: false
hide_last_modified: false
category: "[2023, 멘토님 추천]"
---
### Abstract

#### <span style='color: #f4acb6' name='PinkBlush3'>Adversarial Diffusion Distillation = ADD</span>
- **특징**
	- <span style='color: #f4acb6'>large-scale foundational image diffusion model</span>을 <span style='color: #f4acb6' name='PinkBlush3'>1~4 step만에 효율적으로 샘플링</span>하면서도 high image quality는 유지
- **구현**
	- <span style='color: #f4acb6' name='PinkBlush3'>score distillation</span> 사용
		- <span style='color: #f4acb6' name='TouchOfTurquoise3'>기성(off the shelf, pretrained) large-scale image diffusion model</span>을 teacher signal로 활용
	- <span style='color: #f4acb6' name='PinkBlush3'>adversarial loss</span> 사용
		- <span style='color: #a6a6a6'>(generator와 discriminator 각각의 목표를 표현)</span>
		- 1~2 sampling step만 사용하는 low-step regime에서도 high image fidelity 보장
			- GAN은 원래 1 step에서 sampling하던 모델이니까.
			- <span style='color: #a6a6a6'>fidelity = 생성된 결과물이 원본 데이터나 목표로 하는 기준에 얼마나 충실한지를 나타내는 척도</span>
- **성능**
	- 1 step → 기존의 few-step methods (GAN, Latent Consistency Model)을 확실히 능가
	- 4 step → sota diffusion model (SDXL) 성능에 도달
- **의의**
	- ADD는 foundation model을 활용해 single-step, real-time image synthesis을 하는 최초의 method

### 1. Introduction
- **Diffusion models (DMs)**
	- 설명
		- 생성모델 트렌드의 중심. 고퀄리티 이미지와 비디오 합성에 눈부신 발전 가져왔다.
	- 특징
		- scalability (확장성)
		- <span style='color: #f8dd74' name='MilkyYellow3'>iterative nature (반복적 특성)</span>
			- sampling step이 상당히 많이 필요 → <span style='color: #f8dd74' name='MilkyYellow3'>실시간 application에 방해가 된다</span>
- **Generative Adversarial Networks (GANs)**
	- 특징
		- single-step formulation
		- inherent speed (본질적으로 빠른 속도)
		- <span style='color: #f8dd74'>large dataset으로 scale하는 경우 DM에 비해 sample quality가 낮다.</span>
- **ADD 논문의 목표**
	- 장점 결합
		- <span style='color: #f4acb6'>DM의 우수한 sample quality</span>
		- <span style='color: #f4acb6'>GAN의 inherent speed (본질적으로 빠른 속도)</span>
- **ADD 논문의 approach**
	- training objective = 2가지의 combination
		1. <span style='color: #f4acb6'>adversarial loss</span>
			- 모델이 각 forward pass에서 <span style='color: #f4acb6'>manifold</span> of real images 상에 있는 sample을 directly generate하도록 강제한다
			- 덕분에 (다른 distillation method에서 흔히 보이는) blurriness 및 artifact를 방지하게 된다
		2. <span style='color: #f4acb6'>distillation loss</span>
			- <span style='color: #ee7152'>score distillation sampling (SDS)</span> 에 해당
			- another pretrained (and fixed) DM을 teacher로 사용
				- pretrained DM의 광범위한 지식을 효과적으로 활용
				- large DM에서 관찰되는 강력한<span style='color: #ee7152'>compositionality (구성성)</span>을 보존
	- inference
		- <span style='color: #ee7152'>classifier-free guidance 안 쓴다</span> → memory requirements 더욱 줄였다
	- DM과 GAN의 장점 결합
		- <span style='color: #f4acb6'>iterative refinement</span> (개선) 통해 결과를 향상시킬 수 있는 model ability를 유지한다 → 기존 one-step GAN-based approaches와의 차별점
- **ADD 논문의 기여**
	- ADD는 pretrained 디퓨전 모델을, <span style='color: #f4acb6'>high-fidelity + real-time image generators (w/ only 1~4 sampling steps)</span>로 바꿔주는 method이다!
	- ADD는 <span style='color: #f4acb6'>adversarial training</span>과 <span style='color: #f4acb6'>score distillation</span>의 새로운 조합을 사용한다!
		- design choices → ablate
	- ADD는 여러 strong baseline(기준선) 들을 크게 능가한다
		- <span style='color: #bdd2ea'>LCM, LCM-XL</span>
		- <span style='color: #bdd2ea'>single-step GANs</span>
	- ADD는 single inference step만으로 복잡한 image composition (구성) 을 처리하면서 high image realism을 유지할 수 있다.
	- 4 sampling step 기준으로 ADD-XL은 / teacher model로 사용된 SDXL-Base를 / 512 x 512 resolution (픽셀 해상도) 에서 / 능가하는 성능을 보여준다.

### 2. Background
- Diffusion model의 문제와 이를 해결하기 위한 노력들
	- <span style='color: #f8dd74'>Diffusion model</span>
		- <span style='color: #f8dd74'>iterative nature → real-time application 지연시킨다</span>
	- <span style='color: #f8dd74'>Latent diffusion model</span>
		- iterative nature의 문제를 완화하고자 more computationally feasible latent space 도입
		- 하지만 <span style='color: #f8dd74'>여전히 billion대의 파라미터를 가진 large model의 iterative application에 의존</span>
	- <span style='color: #f8dd74'>diffusion model에 faster sampler 사용하는 연구</span>
	- <span style='color: #f8dd74'>model distillation에 대한 연구</span>
		- progressive distillation
		- guidance distillation
		- iterative sampling step을 4~8 수준으로 줄여주지만, <span style='color: #f8dd74'>성능 저하가 심각했다</span>
		- <span style='color: #f8dd74'>iterative training process</span>를 요하기도 했다
	- c<span style='color: #f8dd74'>onsistency model</span>
		- <span style='color: #d6e399'>ODE trajectory에 consistency regularization 적용</span>
		- model distillation이 <span style='color: #d6e399'>iterative training process를 요하는 문제를 해결</span>
		- few-shot setting에서 pixel-based model에 대해 좋은 성능을 보여줬다
	- L<span style='color: #f8dd74'>CM (Latent consistency model)</span>
		- <span style='color: #d6e399'>LDM distilling</span>에 초점을 맞췄다
		- <span style='color: #d6e399'>4 sampling step</span>에서 좋은 성능을 보였다
	- <span style='color: #f8dd74'>LCM-LoRA</span>
		- LCM 모듈을 효율적으로 learn하기 위한 <span style='color: #d6e399'>LoRA</span> (low-rank adaption) training 제안
		- 그 LCM 모듈은 SD와 SDXL의 다양한 checkpoint에 plug in 될 수 있다.
	- <span style='color: #f8dd74'>InstaFlow</span>
		- distillation process를 개선하기 위해 Rectified Flows 사용할 것을 제안
	- **위 method들의 공통적인 결점**
		- 4 step → synthesize된 sample은 종종 <span style='color: #f8dd74'>blurry</span>해 보이며 눈에 보이는 <span style='color: #f8dd74'>artifact</span>를 만든다.
		- fewer step → 더욱 심각
- **GAN의 강점과 한계**
	- 강점
		- standalone single step model로 학습될 수 있다
		- sampling speed ↑
	- 한계
		- performance는 DM based models에 비해 ↓
			- adversarial objective의 stable train을 위해 필요한, finely balanced GAN-specific architecture 때문이다
				- 이 balance를 해치지 않으면서 model scaling과 nn architecture 발전을 통합하는 게 매우 어려운 일이 된다
		- (논문 작성 시점 기준) 현재 sota text-to-image gan은 classifier-free guidance 같은 method가 없다
			- DM에선 scale에 classifier-free guidance가 필수 요소
- Score Distillation Sampling (= Score Jacobian Chaining)
	- 본 목적: foundational T2I (text→Image) 모델의 knowledge를 3D synthesis model로 distill하자
	- 활용:
		- 대부분의 SDS-based 연구는 SDS를 3D object에 대한 per-scene optimization에 이용
		- text-to-3D-video-synthesis에도 사용
		- image editing에도 사용
- **DM의 요소를 활용해 GAN의 요소를 개선한 연구들**
	- 최근 score-based model과 GAN의 밀접한 관계가 밝혀지고 Score GAN이 제안되었다
		- GAN의 discriminator 대신, DM의 scored-based diffusion flow를 사용해 train됨
	- DiffInstruct도 유사하다
		- SDS를 generalize
		- pretrained diffusion model을 generator로 distill할 수 있게 해준다. discriminator 없이!
- **GAN의 adversarial training을 통해 Diffusion process를 개선한 연구**
	- Denoising DIffusion Gans
		- for faster sampling. enable sampling with few steps
		- quality 향상을 위해 Adversarial Score Matching의 score matching 목적 함수와 CTM의 consistency 목적 함수에 discriminator loss를 추가했다.
- 이 논문의 <span style='color: #f4acb6'>ADD method</span>는 <span style='color: #f4acb6'>adversarial training과 score distillation을 hybrid objective에서 결합</span>함으로써 <span style='color: #d6e399'>현존하는 few-step generative model의 한계를 극복</span>한다.

### 3. Method
- 목표
	- 최대한 적은 sampling step으로 high-fidelity sample 생성
	- sota model의 quality에 뒤지지 않도록
- <span style='color: #f4acb6'>adversarial objective</span>
	- 1 forward step만으로 image manifold 상의 sample을 생성할 수 있도록 model을 train시키므로, fast generation에 자연스럽게 적용된다
	- 단, GAN scaling 시도들로부터, discriminator에만 의존하지 말고 pretrained classifier 또는 (<span style='color: #f8dd74'>text alignment</span> 개선을 위한) CLIP network를 사용해야 한다는 걸 알 수 있었다.
		- discriminative network를 과하게 써먹으면 artifact가 발생하고 <span style='color: #f8dd74'>image quality</span>가 저하된다
- <span style='color: #f4acb6'>score distillation objective</span>
	- 이걸 통해 <span style='color: #f4acb6'>pretrained diffusion model의 gradient를 활용</span>하여 <span style='color: #d6e399'>text alignment와 sample quality를 개선</span>한다.
- (처음부터 train하는 대신) pretrained diffusion model weights로 우리 모델을 initialize
- 