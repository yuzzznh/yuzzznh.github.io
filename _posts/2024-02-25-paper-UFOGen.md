---
layout: post
title: "🛸UFOGen: You Forward Once Large Scale Text-to-Image Generation via Diffusion GANs"
# description: >
#   이 논문을 읽어보았습니당
sitemap: true
hide_last_modified: false
category: [paper, diffusion, todo]
---

<a href="https://arxiv.org/abs/2311.09257">arxiv</a>

# 요점
- **ODE trajectory learning** (noise와 image pair들을 준비하고 supervised로 학습) + **GAN discriminator** (이상한 결과 방지) 가 가능함을 보인 논문
- **single inference**만으로 image를 생성 가능하다

# Abstract
- **Text-to-image** *diffusion model은* text prompt를 coherent image로 변환하는 놀라운 능력을 입증했지만, *multi-step* inference의 *computational cost*는 여전히 문제였다
- 해결책으로서 우리는 **UFOGen**이라는 새로운 generative model을 제시한다. 이 모델은 ultra-fast **one-step** text-to-image generation을 위해 design되었다.
- (기존의 conventional approach들은 *sampler improving*이나 diffusion model에의 *distillation* 사용에 중점을 두었지만,) UFOGen은 **diffusion model과 GAN objective를 통합하는 hybrid methology****를 채택한다.
- **diffusion-GAN objective**을 새롭게 도입하고 **pre-trained diffusion models를 통해 initialization**을 진행함으로써 UFOGen은 textual description에 따라 condition된 high-quality image를 **single step**만에 효과적으로 생성한다.
- Traditional text-to-image generation 외에도 UFOGen은 application에서의 versatility(다양성)을 보여준다.
- 특히 UFOGen은 **one-step text-to-image generation**과 다양한 downstream task를 가능하게 해주는 선구적인 모델들 중 하나로서, 효율적인 생성모델의 landscape에서 중요한 발전을 보여준다.

# 1. Introduction
- diffusion model은 최근 generative model의 강력한 한 종류로 부상했으며, 여러 generative modeling task에서 전례없는 결과를 보여주고 있다.
  - 특히, text로 condition된 high-quality image를 합성하는 데 능했다.
  - 이 task 외에도, large-scale의 text-to-image model들은 여러 downstream application에서 foundational(기초적) building block으로 기능한다.
    - 예: personalized generation, controlled generation, image editing
- 하지만 (인상적인 generative quality와 광범위한 utility(유용성)에도 불구하고,) *diffusion model*은 중요한 한계를 가지고 있었는데, 바로 얘네는 final sample을 generate하기 위해 *iterative denoising*에 의존한다는 것이다. 이러면 generation speed는 *slow*일 수밖에 없다.🤨
  - large-scale diffusion model의... 느린 inference와 consequential computational demand는 이 모델을 써먹기 힘들게 만드는 요인이 되었다.
- Song의 seminal work에서, diffusion model에서의 sampling은, 사실 diffusion process와 관련된 PF-ODE를 풀어내는 것과 equilvalent하다는 것이 밝혀졌다.
  - PF-ODE = probability flow ordinary differential equation (확률 흐름 일반 미분방정식)
- 현재, diffusion model의 sampling efficiency를 개선하고자 하는 연구들은 대부분 ODE formulation (공식화) 에 중점을 둔다.
- One line of work (한 분야의 작업) 은 PF-ODE에 대한 numerical solver를 발전시키려고 한다. 이를 통해 discretization size가 더욱 큰 ODE도 해결가능하게 함으로써, 궁극적으로는 requisite (필요한) sampling step 수를 줄이려는 것이 목적이다.
  - 그러나 step size와 accuracy 간의 inherent trade-off는 여전히 존재한다😵
  - PF-ODE의 highly non-linear and complicated trajectory를 고려하면, required sampling step 수를 minimal level(최소 수준)으로 줄이는 것은 겁나 어려울 것이다.
  - 심지어 가장 발전된 solver조차도 image generate에 10~20 sampling step이 필요하며, 이를 더 줄여버리면 image quality가 확연히 떨어져버린다.
- 대안적인 approach는 pre-trained diffusion model로부터 PF-ODE trajectory를 distill하고자 한다.
  - trajectory를 distill한다는게 무슨뜻인지 모르겠음. TODO
  - distillation의 그 distill인 듯!
  - 예를 들어, progressive distillation 기법은 / PF-ODE solver의 multiple discretizatio step들을 / single step으로 압축하려고 시도한다 / by explicitly aligning (일치시키기) with the solver's output!
  - 마찬가지로, consistency distillation 기법은 ODE trajectory를 따라 point consistency를 유지하는, consistency mapping을 learn하는 방식이다.
  - 이와 같은 기법들은 sampling step 수를 상당히 줄일 수 있는 potential을 보여주었다.
  - 하지만, ODE trajectory의 intrinsic(본질적인) complexity로 인해, extremely small step regime에서는 다들 어려움을 겪고 있다. 특히 large-scale text-to-image diffusion model에서는.
  - 

# 2. Related Works

## (2.1) Text-to-image Diffusion Models

## (2.2) Accelerating Diffusion Models

## (2.3) Text-to-image GANs

## (2.4) Recent Progress on Few-step Text-to-image Generation

# 3. Background

## (3.1) Diffusion Models

## (3.2) Diffusion-GAN Hybrids

# 4. Methods
- 이 섹션은 우리가 diffusion-GAN hybrid model에서 개선하여 UFOGen model을 탄생시킨 부분들에 대해 다룬다.
- 이러한 개선사항들은 두 가지 부분에 초점이 맞춰져 있다.
1. enabling **one step sampling** (section 4.1에서 후술)
2. **scaling-up** for text-to-image generation (section 4.2에서 후술)
  - **pre-trained diffusion model**을 활용한 initialization!

## 4.1 Enabling One-step Sampling for UFOGen
- **diffusion-GAN hybrid model**은 **training을 large denoising step size로** 하도록 맞춤제작되었다 (tailored)
- 그러나 이 모델을 *single denoising step만으로 train*하려는 시도는, 사실상 model training을 conventional GAN의 training으로 reduce해버리겠다는 것이다😟
  - single denoising step으로 model을 train하자는 건 $$x_{T-1} = x_0$$을 의미한다.
  - 결과적으로 그동안의 (prior) diffusion-GAN model들은 one-step sampling을 달성할 수 없었다😟
    - 왜??? TODO 👻
- 이러한 challenge를 고려하여, 우리는 **SIDDM formulation**을 심층검토하고, **generator의 parameterization**과 **objective 속의 reconstruction term**을 수정하였다.
- 이러한 조정은 UFOGen이, 여전히 **train은 several denoising step**을 통해 진행하면서도, **samling은 one-step으로** 할 수 있게 해주었다.

### (4.1.1) Parameterization of the Generator
- diffusion-GAN model에서, generator는 $$x_{t-1}$$의 sample을 생성해야 한다.
- 그러나, *($$x_{t-1}$$을 directly outputting하는 대신)*, DDGAN과 SIDDN의 generator는 
  $$ p_\theta(x_{t-1} | x_t) = q(x_{t-1}|x_t, x_0 = G_\theta(x_t, t)) $$ 로 parameterized된다.
  - 즉, 먼저 $$ x_0 $$ 가 denoising generator $$G_\theta(x_t, t)$$를 통해 predict되고, 
    그 다음에 $$x_{t-1}$$ 은 $$q(x_{t-1}|x_t, x_0$$라는 Gaussian posterior distribution을 통해 sampling된다.
  - 참고로, 이 parameterization은 실용적인 목적을 위한 것이며, alternative parameterization은 model formulation을 break하지 (깨뜨리지) 않는다.
- 우리는 generator에 대한 또다른 타당한 parameterization을 제안한다 - 
  $$p_\theta(x_{t-1}) = q(x_{t-1} | x_0 = G_\theta(x_t, t))$$
  - generator는 여전히 $$x_0$$를 predict하지만, 우리는 $$x_{t-1}$$을 (posterior가 아니라) 
    forward diffusion process, 즉 $$q(x_{t-1}|x_0)$$를 통해 sampling한다.
  - 후술하겠지만, 이러한 설계는 $$x_0$$에서 distribution matching을 허용함으로써 one-step sampling에 이바지한다.
    - distribution matching 뭐지? TODO

### (4.1.2) Improved Reconstruction Loss at $$x_0$$
- (4.1.1의) 새로운 generator parameterization을 쓰게 되면, Equation 4에 등장하는 SIDDM의 objective는 $$x_0$$에서의 distribution과 indirecty match된다.
- 이를 확인하기 위해 우리는, adversarial objective와 KL objective를 별도로 분석한다. (둘 다 Equation 4에 등장)
- (Equation 4의) first term은 adversarial divergence 
  $$D_{adv}(q(x_{t-1}||p_\theta(x'_{t-1})))$$
  을 minimize한다.
  - 여기 나오는 두 분포 $$q_(x_{t-1})$$과 $$p_\theta(x'_{t-1})$$은, 각각 $$x_0$$의 distribution이 (서로 동일한) Gaussian kernel에 의해 corrput된 것이라고 볼 수 있다\dots
  - 

### (4.1.3) Training and Sampling of UFOGen

## 4.2 Leverage Pre-trained Diffusion Models (-> scaling-up for text-to-image generation)
### 문제
- 우리의 목표는 ultra-fast text-to-img 모델을 만드는 것이다.
- 근데 effective UFOGen recipe에서 **web-scale data**로의 transition은 challenging하다.
- (text-to-img 생성을 위한 diffusion-GAN hybrid model에서) **training**이 여러모로 복잡하기 때문 ㅠㅠ
  - 특히 **discriminator**(판별자)는, text-image alignment에서 가장 중요한 texture(질감)과 semantics(의미) 둘다를 기반으로 판단을 내려야 한다.
  - 이 challenge는 training의 **initial stage**(초기단계)에서 특히 두드러진다.
  - 게다가 text-to-imgae model의 training은 cost가 엄청 높을 수 있다 - 특히 GAN 기반 모델의 경우 discriminator가 추가적으로 parameter를 갖고있다ㅠㅠ
  - 순수 GAN 기반의 text-to-img 모델은 이런 복잡성 때문에 매우 복잡하고 비싼 training을 초래한다
### 해결
- diffusion-GAN hybrid model을 scaling-up하는 challenge를 극복하기 위해, 우리는 **pre-trained text-to-image diffusion model**, 특히 **stable diffusion model**을 활용할 것을 제안한다.
  - 구체적으로, UFOGen 모델은 generator와 discriminator 모두에 일관된 UNet structure를 사용하도록 설계되었다.
  - 이 design은 pre-trained stable diffusion model을 이용한 initialization이 원활하도록 해준다.
  - 우리는 stable diffusion model의 internal feature (내부의 특징) 들이 textual & visual data 사이의 복잡한 interplay (상호작용) 에 대해 rich information을 포함하고 있다고 가정한다.
### 성과
- 이 initialization 전략은 UFOGen의 training을 상당히 간소화해준다.
  - stable diffusion model로 UFOGen의 generator와 discriminator를 initialize한 후, 우리는 stable training dynamics (안정적인 training 동역학) 과 놀랍도록 빠른 convergence (수렴) 을 관찰할 수 있었다.
- UFOGen의 완전한 훈련 전략은 다음의 Fig 3에 나타나 있다.
![Full-width image](/assets/img/blog/2024-02-25-UFOGen-Fig3.png)

# 5. Experiments

# 6. Conclusions
- UFOGen
  - text-to-image synthesis(합성)에서의 엄청난 발전
  - inference efficiency라는 challenge를 효과적으로 해결
- 창의적인 hybrid approach: diffusion model과 GAN objective를 합침
  - UFOGen이 text description으로부터 high-quality image 생성하는걸 ultra-fast & one-step 로 할 수 있게 해준다
- UFOGen이 기존의 accelerated diffusion-based method들보다 우월하다
  - 종합적인 평가들로부터 일관되게 확인됨
- UFOGen은 one-step text-to-image synthesis를 독특하게 잘하고, downstream task들에 있어서도 숙련도가 높다
- ultra-fast 속도의 text-to-image synthesis가 가능하게 해주는 선구자
  - gen. models landscape에서 transformative shift의 길을 열어준다
  - UFOGen의 impact는 academic discourse(논의)를 넘어, 빠르고 고퀄인 이미지를 생성하는 데 있어 혁명적인 실용성을 보여줄 것이다