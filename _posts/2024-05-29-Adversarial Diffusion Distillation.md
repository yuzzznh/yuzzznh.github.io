---
layout: post
title: Adversarial Diffusion Distillation
description: 
sitemap: false
hide_last_modified: false
category: "[2023, 멘토님 추천]"
---
***
### Abstract

#### <span style='color: #f4acb6' name='PinkBlush3'>Adversarial Diffusion Distillation = ADD</span>
- **특징**
	- <span style='color: #a1d4cf' name='TouchOfTurquoise3'>large-scale foundational image diffusion model</span>을 <span style='color: #f4acb6' name='PinkBlush3'>1~4 step만에 효율적으로 샘플링</span>하면서도 high image quality는 유지
- **구현**
	- <span style='color: #f4acb6' name='PinkBlush3'>score distillation</span> 사용
		- <span style='color: #a1d4cf' name='TouchOfTurquoise3'>기성(off the shelf, pretrained) large-scale image diffusion model</span>을 teacher signal로 활용
	- <span style='color: #f4acb6' name='PinkBlush3'>adversarial loss</span> 사용
		- 1~2 sampling step만 사용하는 low-step regime에서도 high image fidelity 보장
			- (fidelity: 생성된 결과물이 원본 데이터나 목표로 하는 기준에 얼마나 충실한지를 나타내는 척도)
- **성능**
	- 1 step → 기존의 few-step methods (GAN, Latent Consistency Model)을 확실히 능가
	- 4 step → sota diffusion model (SDXL) 성능에 도달
- **의의**
	- ADD는 foundation model을 활용해 single-step, real-time image synthesis을 하는 최초의 method
***
### Introduction
- Diffusion models (DMs)
	- 설명
		- 생성모델 트렌드의 중심. 고퀄리티 이미지와 비디오 합성에 눈부신 발전 가져왔다.
	- 특징
		- scalability (확장성)
		- <span style='color: #f8dd74' name='MilkyYellow3'>iterative nature (반복적 특성)</span>
			- sampling step이 상당히 많이 필요 → - <span style='color: #f8dd74' name='MilkyYellow3'>실시간 application에 방해가 된다</span>
- Generative Adversarial Networks (GANs)
	- 특징
		- single-step formulation
		- inherent speed (본질적으로 빠른 속도)
		- <span style='color: #f8dd74'>large dataset으로 scale하는 경우 DM에 비해 sample quality가 낮다.</span>
- ADD 논문의 목표
	- 장점 결합
		- DM의 우수한 sample quality
		- GAN의 inherent speed (본질적으로 빠른 속도)
	- 


***