---
layout: post
title: Adversarial Diffusion Distillation
description: 
sitemap: false
hide_last_modified: false
category: "[Paper, 2023]"
image:
---
# Adversarial Diffusion Distillation

Axel Sauer et al. / 2023 / Stability AI
## Abstract

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

![Figure 1](https://i.imgur.com/yRMI4PM.jpeg)

## 1. Introduction
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

## 2. Background
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
	- <span style='color: #f8dd74'>consistency model</span>
		- <span style='color: #d6e399'>ODE trajectory에 consistency regularization 적용</span>
		- model distillation이 <span style='color: #d6e399'>iterative training process를 요하는 문제를 해결</span>
		- few-shot setting에서 pixel-based model에 대해 좋은 성능을 보여줬다
	- <span style='color: #f8dd74'>LCM (Latent consistency model)</span>
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

## 3. Method
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
	- generator network를 pretrain하는 것은 adversarial loss를 쓰는 training을 상당히 개선하는 것이 알려져 있다.
- GAN training에 쓰이는 decoder-only architecture를 쓰는 대신, standard diffusion model framework를 채택한다.
	- This setup naturally enables iterative refinement.

### 3.1 Training Procedure (학습 절차)

<img src="https://i.imgur.com/heiCsTH.png" alt="Figure 2" width="400" style="aspect-ratio: auto;">
#### Networks:
1. <span style='color: #f4acb6'>ADD-Student</span>
	- initialized from a pretrained Unet-DM with weights $\theta$
	- Train - noisy data $x_s$를 받아 sample $\hat{x}_\theta(x_s, s)$를 generate
		- $x_s$는 real image $x_0$에 forward diffusion process $x_s = \alpha_s x_0$를 적용해 만든 것
		- 이 논문에서는 $\alpha_s$와 $\sigma_s$를 student DM과 동일한 값으로 사용하였다.
		- $s$는, $N$ chosen student timesteps로 이루어진 집합 $T_{student} = \{\tau_1, ..., \tau_n\}$에서 uniform sampling해 얻었다. 논문에선 $N=4$로 설정했다.
		- (중요) $τ_n = 1000$으로 설정하고 train 시 zero-terminal SNR을 enforce하여 inference 시 model이 pure noise에서 시작하도록 하였다. 
			- (Ref) Common diffusion noise schedules and sample steps are flawed, 2023
			- 
1. <span style='color: #f4acb6'>discriminator</span>
	- with trainable weights $\phi$
2. <span style='color: #f4acb6'>DM teacher</span>
	- with frozen weights $\psi$ 





적대적 목적 함수를 위해 생성된 샘플 x^θ와 실제 이미지 x_0가 판별자에 전달되며, 판별자는 이들을 구별하는 것을 목표로 한다. 판별자의 설계와 적대적 손실은 3.2절에서 자세히 설명된다. DM teacher의 지식을 증류하기 위해 우리는 student 샘플 x^θ를 teacher의 순방향 프로세스로 x^θ_t로 확산시키고, teacher의 디노이징 예측 x^ψ(x^θ_t, t)를 증류 손실 L_distill의 재구성 타겟으로 사용한다(3.3절 참조).


그러므로 전체 목적 함수는 다음과 같다:

L = L^G_adv(x^θ(x_s, s), φ) + λL_distill(x^θ(x_s, s), ψ) (1)

우리는 픽셀 공간에서 방법을 공식화하지만, 잠재 공간에서 작동하는 LDM에 이를 적용하는 것은 간단하다. teacher와 student가 공유된 잠재 공간을 가진 LDM을 사용할 때, 증류 손실은 픽셀 공간이나 잠재 공간에서 계산될 수 있다. 잠재 확산 모델을 증류할 때 픽셀 공간에서 증류 손실을 계산하면 더 안정적인 그래디언트를 산출하기 때문에 우리는 픽셀 공간에서 증류 손실을 계산한다[72].

3.2 적대적 손실 판별자의 경우, 우리는 [59]에서 제안된 설계와 학습 절차를 따른다. 이를 간단히 요약하면 다음과 같다. 자세한 내용은 원논문을 참조하기 바란다. 우리는 고정된 사전 학습 특징 네트워크 F와 학습 가능한 경량 판별자 헤드 D_φ,k의 집합을 사용한다. 특징 네트워크 F로는 Sauer et al.[59]이 잘 작동한다고 발견한 vision transformer(ViT)[9]를 사용하며, 4장에서 ViT의 목적 함수와 모델 크기에 대한 다양한 선택을 분석한다. 학습 가능한 판별자 헤드는 특징 네트워크의 서로 다른 레이어에서 특징 F_k에 적용된다.

성능 향상을 위해 판별자는 projection[46]을 통해 추가 정보로 조건화될 수 있다. 일반적으로 텍스트-이미지 설정에서는 텍스트 임베딩 c_text가 사용된다. 그러나 표준 GAN 학습과는 달리, 우리의 학습 구성은 주어진 이미지에 대해서도 조건화할 수 있게 해준다. τ < 1000일 때, ADD-student는 입력 이미지 x_0로부터 일부 신호를 받는다. 따라서 주어진 생성 샘플 x^θ(x_s, s)에 대해, 우리는 x_0의 정보로 판별자를 조건화할 수 있다. 이는 ADD-student가 입력을 효과적으로 활용하도록 장려한다. 실제로 우리는 추가 특징 네트워크를 사용하여 이미지 임베딩 c_img를 추출한다.

[57, 59]를 따라 우리는 적대적 목적 함수로 hinge 손실[32]을 사용한다. 따라서 ADD-student의 적대적 목적 함수 L_adv(x^θ(x_s, s), φ)는 다음과 같다:

L^G_adv(x^θ(x_s, s), φ) = -E_s,ϵ,x_0 [Σ_k D_φ,k(F_k(x^θ(x_s, s)))] (2)

반면 판별자는 다음을 최소화하도록 학습된다:

L^D_adv(x^θ(x_s, s), φ) = E_x_0 [Σ_k max(0, 1 - D_φ,k(F_k(x_0))) + γR1(φ)]

- E_x^θ [Σ_k max(0, 1 + D_φ,k(F_k(x^θ)))] (3)

여기서 R1은 R1 그래디언트 패널티[44]를 나타낸다. 픽셀 값에 대한 그래디언트 패널티를 계산하는 대신, 우리는 각 판별자 헤드 D_φ,k의 입력에 대해 계산한다. 우리는 128^2 픽셀보다 큰 출력 해상도로 학습할 때 R1 패널티가 특히 유익하다는 것을 발견했다.

3.3 Score Distillation 손실 식 (1)의 증류 손실은 다음과 같이 공식화된다:

L_distill(x^θ(x_s, s), ψ) = E_t,ϵ' [c(t)d(x^θ, x^ψ(sg(x^θ_t); t))] (4)

여기서 sg는 stop-gradient 연산을 나타낸다. 직관적으로, 이 손실은 거리 메트릭 d를 사용하여 ADD-student에 의해 생성된 샘플 x_θ와 DM-teacher의 출력 x^ψ(x^θ_t, t) = (x^θ_t - σ_t * ϵ^ψ(x^θ_t, t)) / α_t 사이의 불일치를 타임스텝 t와 노이즈 ϵ'에 대해 평균낸 값을 측정한다.

주목할 점은 teacher가 ADD-student의 생성물 x^θ에 직접 적용되는 것이 아니라 확산된 출력 x^θ_t = α_t * x^θ + σ_t * ϵ'에 적용된다는 것인데, 이는 비확산 입력이 teacher 모델에 대해 out-of-distribution이기 때문이다[68].

다음으로 우리는 거리 함수를 d(x, y) := ||x - y||^2_2로 정의한다. 가중치 함수 c(t)에 대해서는 두 가지 옵션을 고려한다: 지수 가중치(exponential weighting), 여기서 c(t) = α_t(높은 노이즈 수준은 기여도가 낮음), 그리고 score distillation sampling (SDS) 가중치[51]. 보충 자료에서 우리는 d(x, y) = ||x - y||^2_2와 c(t)에 대한 특정 선택을 통해 우리의 증류 손실이 [51]에서 제안된 SDS 목적 함수 L_SDS와 동등해짐을 보인다.

우리 공식화의 장점은 재구성 타겟을 직접 시각화할 수 있다는 점과 여러 연속적인 디노이징 단계를 자연스럽게 수행할 수 있다는 점이다. 마지막으로 우리는 최근 제안된 SDS의 변형인 noise-free score distillation (NFSD) 목적 함수도 평가한다[28].

1. 실험 실험을 위해 우리는 서로 다른 용량의 두 모델인 ADD-M(860M 파라미터)과 ADD-XL(3.1B 파라미터)을 학습시킨다. ADD-M 분석을 위해서는 Stable Diffusion (SD) 2.1 백본[54]을 사용하고, 다른 기준선과의 공정한 비교를 위해서는 SD1.5를 사용한다. ADD-XL은 SDXL[50] 백본을 활용한다. 모든 실험은 512x512 픽셀의 표준화된 해상도로 수행된다. 더 높은 해상도를 생성하는 모델의 출력은 이 크기로 다운샘플링된다.

우리는 모든 실험에서 증류 가중치 계수로 λ = 2.5를 사용한다. 또한 R1 패널티 강도 γ는 10^-5로 설정된다. 판별자 조건화를 위해 우리는 텍스트 임베딩 c_text를 계산하기 위해 사전 학습된 CLIP-ViT-g-14 텍스트 인코더[52]를 사용하고, 이미지 임베딩 c_img를 위해 DINOv2 ViT-L 인코더[47]의 CLS 임베딩을 사용한다.

기준선으로는 공개적으로 이용 가능한 최고 모델들을 사용한다: 잠재 확산 모델[50, 54](SD1.5, SDXL), 캐스케이드 픽셀 확산 모델[55](IF-XL), 증류된 확산 모델[39, 41](LCM-1.5, LCM-1.5-XL), 그리고 OpenMUSE[48], 즉 빠른 추론을 위해 특별히 개발된 트랜스포머 모델인 MUSE[6]의 재구현이다. SDXL-Base-1.0 모델과의 공정한 비교를 위해 추가 리파이너 모델 없이 비교한다는 점에 유의하라. 최첨단 GAN 모델이 공개되어 있지 않기 때문에 우리는 개선된 판별자로 StyleGAN-T[59]를 재학습시킨다.

이 기준선(StyleGAN-T++)은 FID와 CS에서 이전의 최고 GAN을 크게 능가한다(보충 자료 참조). 우리는 FID[18]를 통해 샘플 품질을, CLIP score[17]를 통해 텍스트 정렬을 정량화한다. CLIP score의 경우 LAION-2B[61]에서 학습된 ViT-g-14 모델을 사용한다. 두 메트릭 모두 COCO2017[34]에서 5k 샘플에 대해 평가된다.

4.1 분석 연구 우리의 학습 설정은 적대적 손실, 증류 손실, 초기화 및 손실 상호작용과 관련하여 다양한 설계 공간을 열어준다. 우리는 Table 1에서 여러 선택 사항에 대한 분석 연구를 수행한다. 주요 통찰력은 각 표 아래에 강조되어 있다. 다음에서 각 실험에 대해 논의할 것이다.

판별자 특징 네트워크. (Table 1a) Stein et al.[67]의 최근 통찰에 따르면 CLIP[52] 또는 DINO[5, 47] 목적 함수로 학습된 ViT가 생성 모델의 성능을 평가하는 데 특히 적합하다고 한다. 유사하게 이러한 모델들은 판별자 특징 네트워크로도 효과적인 것으로 보이며, DINOv2가 최선의 선택으로 떠오른다.

판별자 조건화. (Table 1b) 이전 연구와 유사하게 우리는 판별자의 텍스트 조건화가 결과를 향상시킨다는 것을 관찰한다. 특히 이미지 조건화가 텍스트 조건화를 능가하고 c_text와 c_img의 조합이 최상의 결과를 산출한다.

Student 사전학습. (Table 1c) 우리의 실험은 ADD-student의 사전학습 중요성을 입증한다. 사전 학습된 생성기를 사용할 수 있다는 것은 순수 GAN 접근 방식에 비해 상당한 이점이다. GAN의 문제점은 확장성 부족이다. Sauer et al.[59]과 Kang et al.[25] 모두 특정 네트워크 용량에 도달한 후 성능 포화를 관찰한다. 이는 DM의 일반적으로 매끄러운 스케일링 법칙[49]과 대조된다. 그러나 ADD는 더 큰 사전 학습된 DM을 효과적으로 활용할 수 있다(Table 1c 참조).

손실 항. (Table 1d) 우리는 두 손실 모두 필수적임을 발견한다. 증류 손실만으로는 효과적이지 않지만, 적대적 손실과 결합되면 결과가 눈에 띄게 개선된다. 서로 다른 가중치 스케줄은 서로 다른 동작으로 이어진다. 지수 스케줄은 더 다양한 샘플을 산출하는 경향이 있는 반면(낮은 FID로 표시됨), SDS와 NFSD 스케줄은 품질과 텍스트 정렬을 개선한다.

모든 다른 분석에서 기본 설정으로 지수 스케줄을 사용하지만, 최종 모델 학습에는 NFSD 가중치를 선택한다. 최적의 가중치 함수를 선택하는 것은 개선의 기회를 제공한다. 또는 3D 생성 모델링 문헌[23]에서 탐구된 바와 같이 학습 과정에서 증류 가중치를 스케줄링하는 것을 고려할 수 있다.


Teacher 유형. (Table 1e) 흥미롭게도 더 큰 student와 teacher가 반드시 더 나은 FID와 CS를 초래하지는 않는다. 오히려 student는 teacher의 특성을 채택한다. SDXL은 일반적으로 더 높은 FID를 얻는데, 이는 아마도 덜 다양한 출력 때문일 것이다. 그러나 그것은 더 높은 이미지 품질과 텍스트 정렬[50]을 보여준다.

Teacher 단계. (Table 1f) 우리의 증류 손실 공식은 본질적으로 teacher로 여러 연속 단계를 수행할 수 있게 해주지만, 여러 단계가 결정적으로 더 나은 성능을 초래하지는 않는다는 것을 발견했다.

4.2 최첨단 모델과의 정량적 비교 다른 접근 방식과의 주요 비교에서 우리는 자동화된 메트릭에 의존하지 않고 사용자 선호도 연구를 사용한다. 왜냐하면 사용자 선호도 연구가 더 신뢰할 수 있기 때문이다[50]. 이 연구에서 우리는 프롬프트 준수와 전반적인 이미지 모두를 평가하고자 한다. 성능 척도로서 우리는 일대일 비교에 대한 승리 백분율을 계산하고 여러 접근법을 비교할 때 ELO 점수를 계산한다. 보고된 ELO 점수의 경우 프롬프트 준수와 이미지 품질 사이의 평균 점수를 계산한다. ELO 점수 계산과 연구 매개변수에 대한 세부 사항은 보충 자료에 나열되어 있다.

Fig. 5와 Fig. 6은 연구 결과를 보여준다. 가장 중요한 결과는 다음과 같다. 첫째, ADD-XL은 단일 단계로 LCM-XL(4단계)을 능가한다. 둘째, ADD-XL은 4단계로 SDXL(50단계)을 대부분의 비교에서 이길 수 있다. 이는 ADD-XL을 단일 단계와 다중 단계 설정 모두에서 최첨단으로 만든다. Fig. 7은 추론 속도 대비 ELO 점수를 시각화한다. 마지막으로 Table 2는 동일한 기본 모델을 사용하는 다양한 소수 단계 샘플링 및 증류 방법을 비교한다. ADD는 8단계의 표준 DPM solver를 포함한 다른 모든 접근 방식을 능가한다.

4.3 정성적 결과 위의 정량적 연구를 보완하기 위해 이 섹션에서는 정성적 결과를 제시한다. 더 완전한 그림을 제시하기 위해 보충 자료에 추가 샘플과 정성적 비교를 제공한다. Fig. 3는 소수 단계 레짐에서 ADD-XL(1단계)을 현재 최고의 기준선과 비교한다. Fig. 4는 ADD-XL의 반복적 샘플링 과정을 보여준다. 이러한 결과는 초기 샘플을 개선하는 우리 모델의 능력을 보여준다. 이러한 반복적 개선은 StyleGAN-T++와 같은 순수 GAN 접근법에 비해 또 다른 중요한 이점을 나타낸다. 마지막으로 Fig. 8은 ADD-XL을 teacher 모델인 SDXL-Base와 직접 비교한다. 4.2절의 사용자 연구에서 나타난 바와 같이 ADD-XL은 품질과 프롬프트 정렬 모두에서 teacher를 능가한다. 향상된 사실성은 약간 감소된 샘플 다양성을 대가로 한다.

1. 토론 이 연구는 사전 학습된 확산 모델을 빠른 소수 단계 이미지 생성 모델로 증류하는 일반적인 방법인 Adversarial Diffusion Distillation을 소개한다. 우리는 공개된 Stable Diffusion[54]과 SDXL[50] 모델을 증류하기 위해 판별자를 통한 실제 데이터와 확산 teacher를 통한 구조적 이해를 모두 활용하는 적대적 목적 함수와 score distillation 목적 함수를 결합한다. 우리의 접근 방식은 1-2단계의 초고속 샘플링 레짐에서 특히 잘 수행되며, 우리의 분석은 이 레짐에서 그것이 모든 동시 방법을 능가함을 보여준다. 더욱이 우리는 다중 단계를 사용하여 샘플을 정제하는 능력을 유지한다. 실제로 4개의 샘플링 단계를 사용하여 우리 모델은 SDXL, IF, OpenMUSE와 같이 널리 사용되는 다단계 생성기를 능가한다.

우리 모델은 단일 단계에서 고품질 이미지 생성을 가능하게 하여 foundation 모델을 이용한 실시간 생성을 위한 새로운 가능성을 열어준다.

감사의 말 우리는 초안에 대한 피드백을 준 Jonas Müller, 증명과 조판에 대한 피드백을 준 Patrick Esser, 초기 모델 데모를 구축한 Frederic Boesel, 데이터를 생성하고 유익한 토론을 해준 Minguk Kang과 Taesung Park, 컴퓨팅 인프라를 유지 관리해준 Richard Vencu, Harry Saini, Sami Kama, 창의적 샘플링 지원을 해준 Yara Wald, 그리고 Vanessa Sauer의 전반적인 지원에 감사드린다.