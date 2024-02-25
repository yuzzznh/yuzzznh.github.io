---
layout: post
title: "UFOGen: You Forward Once Large Scale Text-to-Image Generation via Diffusion GANs"
description: >
  이 논문을 읽어보았습니당
sitemap: true
hide_last_modified: false
category: [paper, diffusion, todo]
---

<a href="https://arxiv.org/abs/2311.09257">arxiv</a>

# 요점
- **ODE trajectory learning** (noise와 image pair들을 준비하고 supervised로 학습) + **GAN discriminator** (이상한 결과 방지) 가 가능함을 보인 논문
- **single inference**만으로 image를 생성 가능하다

# Abstract
- **Text-to-image** diffusion model은 text prompt를 coherent image로 변환하는 놀라운 능력을 입증했지만, *multi-step* inference의 computational cost는 여전히 문제였다
- 해결책으로서 우리는 **UFOGen**이라는 새로운 generative model을 제시한다. 이 모델은 ultra-fast **one-step** text-to-image generation을 위해 design되었다.
- (기존의 conventional approach들은 sampler improving이나 diffusion model에의 distillation 사용에 중점을 두었지만,) UFOGen은 **diffusion model과 GAN objective를 통합하는 hybrid methology****를 채택한다.
- **diffusion-GAN objective**을 새롭게 도입하고 **pre-trained diffusion models를 통해 initialization**을 진행함으로써 UFOGen은 textual description에 따라 condition된 high-quality image를 single step만에 효과적으로 생성한다.
- Traditional text-to-image generation 외에도 UFOGen은 application에서의 versatility(다양성)을 보여준다.
- 특히 UFOGen은 **one-step text-to-image generation**과 다양한 downstream task를 가능하게 해주는 선구적인 모델들 중 하나로서, 효율적인 생성모델의 landscape에서 중요한 발전을 보여준다.

# 1. Introduction

# 2. Related Works

## (2.1) Text-to-image Diffusion Models

## (2.2) Accelerating Diffusion Models

## (2.3) Text-to-image GANs

## (2.4) Recent Progress on Few-step Text-to-image Generation

# 3. Background

## (3.1) Diffusion Models

## (3.2) Diffusion-GAN Hybrids

# 4. Methods

## 4.1 Enabling One-step Sampling for UFOGen

### (4.1.1) Parameterization of th eGenerator

### (4.1.2) Improved Reconstruction Loss at $$x_0$$

### (4.1.3) Training and Sampling of UFOGen

## 4.2 Leverage Pre-trained Diffusion Models
- 우리의 목표는 ultra-fast text-to-img 모델을 만드는 것이다.
- 근데 effective UFOGen recipe에서 web-scale data로의 transition은 challenging하다.
- (text-to-img 생성을 위한 diffusion-GAN hybrid model에서) **training**이 여러모로 복잡하다ㅠㅠ
- 특히 discriminator(판별자)는, text-image alignment에서 가장 중요한 texture(질감)과 semantics(의미) 둘다를 기반으로 판단을 내려야 한다.
- 이 challenge는 training의 initial stage(초기단계)에서 특히 두드러진다.
- 게다가 text-to-imgae model의 training은 cost가 엄청 높을 수 있다 - 특히 GAN 기반 모델의 경우 discriminator가 추가적으로 parameter를 갖고있다ㅠㅠ
  - 순수 GAN 기반의 text-to-img 모델은 이런 복잡성 때문에 매우 복잡하고 비싼 training을 초래한다
  - 

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