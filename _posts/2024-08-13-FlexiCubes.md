---
layout: post
title: 2024-08-08-FlexiCubes-Review
description: ""
sitemap: true
hide_last_modified: false
image: 
related_posts: 
accent_image: 
accent_color: 
theme_color: 
invert_sidebar: false
category: "[paper, 3d, mesh]"
---

## Flexible Isosurface Extraction for Gradient-Based Mesh Optimization

Tianchang Shen et al.

10 Aug 2023

NVIDIA

[arxiv](https://arxiv.org/pdf/2308.05371)

---
## 0. Abstract

### 이 논문의 의의
- <span style='color: #f4acb6'>FlexiCubes</span> 제안

### <span style='color: #f4acb6'>FlexiCubes</span>란?
- 고퀄리티 isosurface representation 기법
- 각종 (geometric, visual, or physical) objective들에 대한 <span style='color: #f4acb6'>gradient-based mesh optimization</span>을 위해 설계됨
- main insight는 representation에 추가적인 parameters를 도입한 것이다
- 다양한 실험을 통해 synthetic 벤치마크와 real-world application 모두에서 mesh quality와 geometric fidelity 면에서 성능이 입증되었다.

### FlexiCubes에 추가된 parameter들

- 의의: extracted mesh의 geometry와 connectivity에 <span style='color: #f4acb6'>local flexible adjustments</span>를 가능하게 해준다
- optimizing (for a downstream task) 과정에서 <span style='color: #f4acb6'>automatic differentiation을 통해 underlying scalar field와 함께 update된다.</span>

### FlexiCubes와 <span style='color: #f4acb6'>Dual Marching Cubes</span>의 관계
- FlexiCubes의 extraction scheme이 <span style='color: #f4acb6'>DMC 기반</span>인데, 그 덕분에 FlexiCubes는 개선된 위상적 특성을 가짐과 동시에 extension에서 사면체이며 hierarchically-adaptive인 mesh를 optional하게 생성할 수 있다.

##### <span style='color: #a6a6a6'>isosurface란?</span>
- <span style='color: #a6a6a6'>3차원 데이터 세트에서 동일한 값들을 가지는 점들을 연결하여 형성한 표현.</span>
- <span style='color: #a6a6a6'>즉, scalar field의 등고선.</span>

### <span style='color: #f4acb6'>gradient-based mesh optimization</span>이란?
- 3D surface mesh를 어떤 scalar field의 isosurface로 표현함으로써 iteratively optimize한다
- 현 구현들은 <span style='color: #f4acb6'>Marching Cubes</span>나 <span style='color: #f4acb6'>Dual Contouring</span> 등의 classic isosurface extraction algorithm들을 적용하는데, 이것들은 mesh를 fixed, known fields에서 extract하도록 설계되었으며, 이에 따라 <span style='color: #f8dd74'>optimization에서 high-quality feature-preserving mesh를 표현하기 위한 충분한 degree of freedom을 갖지 못하거나, numerical instability의 문제를 갖는다</span>.

---
## 1. Introduction

### Surface Mesh란?
- 3D geometry의 representation/transmission/generation에서 중요한 역할을 한다
- 임의의 surface에 대해 명확하고 정확한 encoding을 제공하여, efficient hardware accelerated rendering의 수혜를 받는다.
- 단, 이러한 장점들은 high quality mesh에 국한되는 경우가 많다. 따라서 <span style='color: #f4acb6'>high quality mesh</span>를 만드는 것은 중요하지만, 어려운 일이다.
- <span style='color: #f8dd74'>저퀄리티 mesh = element 수가 너무 많거나 / self-intersection이 있거나 / sliver element가 있거나 / underlying geometry를 poorly capture하거나</span>
- 원래는 mesh 생성이 사람이 맡던 일이었지만 최근 automatic mesh generation의 필요성이 대두되고 있는데, 이는 종종 <span style='color: #f4acb6'>differentiable mesh generation</span>을 기반으로 이루어진다.

### <span style='color: #f4acb6'>differentiable mesh generation</span>이란?
- 3D surface mesh의 space를 parameterize하고, gradient-based techniques를 통해 그것을 다양한 objective 함수에 대해 optimize할 수 있게 하는 것.
- <span style='color: #a6a6a6'>세상이 단순했더라면 한 mesh representation와 objective에 대해 naive gradient descent만 하면 됐겠지만, 역시나</span> 여러 난관이 존재하여 저퀄리티 mesh를 만들게 한다.
- <span style='color: #a1d4cf'>난관 1 = topology가 다양한 mesh를 어떻게 optimize할 것인가?</span>
- <span style='color: #a1d4cf'>난관 2 = 기존 formulation의 stability와 robustness가 부족한 문제</span>
- <span style='color: #f4acb6'>FlexiCubes</span>는 이러한 난관을 극복하게 도와주는 new formulation으로서, 여러 downstream task에 있어 differentiable mesh generation을 보다 용이하고 퀄리티 좋게 수행할 수 있게 해준다.
- <span style='color: #a6a6a6'>mesh의 vertex 위치를 directly optimize → (매우 조심스럽게 initialization / remeshing / regularization 하지 않는 한) degeneracy와 local minima의 함정에 빠질 수 있다</span>
- 일반적인 방법은 space에 <span style='color: #f4acb6'>scalar field 또는 signed distance function (SDF)</span> 를 하나 정의해서 <span style='color: #f4acb6'>optimize</span>하고, <span style='color: #f4acb6'>그 function의 level set (함숫값이 동일한 점들의 집합) 을 근사하는 triangle mesh를 extract</span>하는 것.
- 그 경우, scalar function representation과 mesh extraction scheme을 뭘로 하는지가 optimization pipeline 전반의 성능에 지대한 영향을 미친다.
- <span style='color: #a1d4cf'>작지만 중요한 난관 3 = scalar field에서 mesh를 extract할 때, possible generated meshes의 space가 restricted일 수 있다.</span> (후술: triangle mesh를 추출할 알고리즘을 고르는 게 generated shape의 특성을 결정짓는다.)
- 난관 1/2/3에 대한 해결책으로, 저자들은 mesh generation 절차가 downstream task에 쉽고/효율적이고/고퀄리티 optimization을 가능케 해주기 위해서 만족시켜야 할 두 가지 조건을 제시한다.
- <span style='color: #f4acb6'>조건 1 = Grad : 이 mesh에 대한 differentiation이 well defined이며, 실제로 gradient-based optimization이 효율적으로 수렴한다.</span>
- <span style='color: #f4acb6'>조건 2 = Flexible : mesh vertex들이 (surface feature에 fit하고 적은 수의 element로도 고퀄리티 mesh를 만들기 위해) individually & locally 조정되는 것이 가능하다.</span>
- <span style='color: #a1d4cf'>근데 조건 1과 조건 2가 본질적으로 상충된다ㅠㅠ</span> 조건 2를 만족시키기 위해 flexibility를 늘리면 → 원치 않았던 degenerate geometry / self-intersection도 표현가능해지는 바람에 → gradient-based optimization에서 수렴에 방해가 돼서 조건 1 만족이 어려워진다
- 그래서 <span style='color: #f8dd74'>기존의 방법들은 주로 조건 1/2 중 하나는 잘 만족시키지 못했다.</span>

### <span style='color: #f8dd74'>기존의 differentiable mesh generation 방법들은 조건 1/2 중 하나만 충족시킨다.</span>
#### <span style='color: #f4acb6'>Marching Cubes (MC)</span>
- <span style='color: #f8dd74'>조건 2인 Flexible을 포기</span>하고 조건 1인 Grad를 챙겼다.
- 안 Flexible한 이유 = <span style='color: #f8dd74'>vertex들이 fixed lattice 위에만 존재하게 만들어졌다</span> → 만들어진 mesh는 <span style='color: #f8dd74'>절대로 non-axis-aligned sharp feature들과 align할 수 없다ㅠㅠ</span> Figure 1 참고.
#### <span style='color: #a6a6a6'>Generalized Marching</span>
- underlying grid를 변형할 수는 있지만, 여기서도 각 vertex에 대한 위치 조정은 허용하지않는다 → sliver elements, imperfect fits ㅠㅠ
#### <span style='color: #f4acb6'>Dual Contouring (DC)</span>
- 조건 2인 Flexible을 챙겨서 sharp feature도 capture할 수 있는 것으로 유명하지만, <span style='color: #f8dd74'>조건 1인 Grad에서 취약</span>하다
- <span style='color: #f8dd74'>vertex를 위치시키는 데 쓰이는 linear system이 unstable & ineffective optimization의 문제로 이어진다.</span>

### 반면 <span style='color: #f4acb6'>FlexiCubes</span>는 조건 1/2를 모두 충족시킨다!
#### Insight
- <span style='color: #f4acb6'>Dual Marching Cubes의 특정 formulation</span>을 적용했다
- <span style='color: #f4acb6'>추가적인 degrees of freedom</span>을 도입해 각각의 extracted vertex들을 그것의 dual cell 안에 <span style='color: #f4acb6'>flexible</span>하게 배치할 수 있도록 하는 방식으로 <span style='color: #f4acb6'>조건 2인 Flexibility</span>를 충족시킨다.
- 저자들은 formulation에 careful한 <span style='color: #f4acb6'>constraint</span>을 가하여, 생성되는 mesh가 <span style='color: #f4acb6'>manifold + watertight + 대부분 intersection-free</span> 이도록 만듦으로써, underlying mesh에 대해 well-behaved differentiation이 가능하게 하는 방식으로 <span style='color: #f4acb6'>조건 1인 Grad</span>를 충족시킨다.
- <span style='color: #f4acb6'>실제로 mesh에 대한 gradient-based optimization이 꾸준히 성공한다</span>는 점이 조건 1 Grad를 충족시켜주는 중요한 부분이다.
##### <span style='color: #a6a6a6'>manifold</span>
- <span style='color: #a6a6a6'>모든 면이 잘 연결되어 있어 연속적</span>
##### <span style='color: #a6a6a6'>watertight</span>
- <span style='color: #a6a6a6'>표면에 구멍이나 틈이 없어 완전히 밀폐됨</span>
#### Evaluation

- 조건 1/2와 같은, 본질적으로 경험적인 부분을 평가하기 위해, 이 논문에서는 몇몇 downstream task에 대해 다양하게 FlexiCubes의 evaluation을 진행했으며, 실험 결과 다양한 mesh 생성 응용 과제에 있어 유의미한 benefit이 있었다.
- task = inverse rendering, optimizing physical and geometric energies, and generative 3D modeling
- 생성된 mesh는 적은 element 수로도 desired geometry를 간명히 capture하였으며, gradient descent를 통해 잘 optimize되었다. (조건 2/1 충족)
#### <span style='color: #d5b6d4'>FlexiCubes의</span> <span style='color: #d5b6d4'>extension</span>
- <span style='color: #d5b6d4'>extension 1 = hierarchical refinement를 통해 mesh의 resolution을 adaptively adjusting하기</span>
- <span style='color: #d5b6d4'>extension 2 = domain 내부를 자동으로 사면체화 하기 (tetrahedralizing)</span>

---
## 2. Related Work

### <span style='color: #f4acb6'>Isosurface Extraction</span> 방법에 대해 알아보자
- tradional method들은 scalar function의 level set을 나타내는 polygonal mesh를 extract하며, 크게 3가지로 분류할 수 있다.
- scalar field로부터의 extraction엔 진전이 꽤 있었지만, isosurfacing method들을 gradient-based mesh optimization에 적용하는 건 아직 난제로 남아있다.
#### (1) <span style='color: #f4acb6'>Spatial Decomposition</span> 방식으로 isosurface 얻기
- space를 정육면체/사면체 등의 cell 여러개로 나눈 다음, 그 cell 안에서 polygon을 만든다.
##### <span style='color: #f4acb6'>Marching Cubes(MC)</span>는 Spatial Decomposition을 통해 isosurface를 얻는 방법이다
- 단점: <span style='color: #f8dd74'>topological 모호함 + sharp features 표현 잘 못함</span>
##### <span style='color: #f4acb6'>Dual Contouring(DC)</span>도 Spatial Decomposition을 통해 isosurface를 얻는 방법이다
- <span style='color: #a6a6a6'>MC가 sharp features 표현 잘 못하는데,</span> DC는 sharp feature를 잘 capture하고자 <span style='color: #f4acb6'>mesh vertex들이 per-cell로 extract되는</span> <span style='color: #f4acb6'>dual representation</span>으로 옮겨갔다
- vertex position을 local isosurface detail에 따라 estimate하기로 했다
##### <span style='color: #f4acb6'>Dual Marching Cubes(DMC)</span>는 MC와 DC의 장점을 모두 가지고간다

#### (2) <span style='color: #a6a6a6'>Surface Tracking</span> 방식으로 isosurface 얻기
- surface sample들 간의 neighboring information을 활용해서 isosurface를 얻는다.
- 이 방식들에 gradient-based mesh optimization을 적용하는 건 discrete & iterative update process에 미분을 적용해야 되기 때문에 또다른 난제가 된다.
##### Marching Triangles

#### (3) <span style='color: #a6a6a6'>Shrink Wrapping</span>으로 isosurface 얻기
- 구형(spherical) mesh를 shrinking하거나, critical point들을 팽창시킴으로써 isosurface를 얻는다,
- 제한된 topological cases에만 적용되며, 임의의 topology에 적용시키기 위해서는 critical point를 직접 골라줘야 하는 방법들이다.
- shrinking에 대한 미분도 명확하지 않기 때문에 gradient-based optimization에 맞지 않는 방법이다.
### ML에서의 Gradient-Based Mesh Optimization
- 몇몇 연구들은 neural network로 3D mesh를 생성하는 걸 다룬다. 여기서 parameter들은 특정 loss function에 대해 gradient-based optimization 된다.
- 초기 연구들은 generated shape의 topology를 predefine했으나, 이 방식은 complex topology를 가지는 objects에 대한 일반화 능력에 한계가 있었다.
- 이를 해결하기 위해 AtlasNet, Mesh R-CNN, PolyGen, CvxNet, BSPNet 등이 제안되었으나 각각의 한계를 가지고 있었다.
- 최근 들어 diffenetiable mesh reconstruction scheme에 대한 연구들이 많아졌는데, (convolutional network나 implicit neural fields로 encode된) implicit function에서 isosurface를 extract하는 것들이었다.
- Deep Marching Cube와 MeshSDF, DefTet 등이 제안되었다.
- FlexiCubes와 가장 비슷한 것은 <span style='color: #f4acb6'>DMTet (Deep Marching Tetrahedra)</span>이다. 이것은 differentiable Marching Tetrahedra(사면체) layer를 이용해 mesh를 extract하기 때문이다. 이 논문에 대해선 Section 3에서 더 자세하게 다루고 있다.

---
## 3. Background and Motivation
- 기존의 많이쓰이는 isosurface extraction scheme들과 각각의 단점에 대해 알아보자. 이것은 FlexiCubes의 motivation으로 이어진다.
### Problem Statement
- 우리가 원하는 건 differentiable mesh optimization에 써먹을 representation이고, 그 basic pipeline은 { (1) space에 <span style='color: #f4acb6'>scalar signed-distance function </span>정의하기 (2) 그 함수의 <span style='color: #f4acb6'>0-isosurface를 triangle mesh로 extract</span>하기 (3) <span style='color: #f4acb6'>그 mesh에 대해 objective functions를 evaluate하기</span> (4) <span style='color: #f4acb6'>underlying scalar function까지 gradients를 back-propagate하기</span> } 로 이루어진다.
- <span style='color: #f8dd74'>gradient-based optimization의 효과가 isosurface extraction 방법의 선택에 좌우된다는 게 가장 큰 난관</span>이 된다. 왜냐하면 isosurface extraction에 많이 쓰이는 알고리즘 중 몇몇 개는 미분과 관련해 이슈가 있기 때문. 예를 들어 <span style='color: #f8dd74'>restrictive parameterizations, numerically unstable expressions, 그리고 topological obstructions</span> 얘네들은 gradient-based optimization에 쓰일 때 실패와 artifacts를 불러오는 놈들이다.
- <span style='color: #f4acb6'>FlexiCubes</span>라는 representation은 (기존 연구에서 집중하던) fixed, known scalar fields에서의 isosurface extraction을 위한 것이 아니다. 오히려 저자들은 <span style='color: #f4acb6'>underlying scalar field이 unknown인 상태에서의 differentiable mesh optimization</span>을 생각하는데, 여기서 <span style='color: #f4acb6'>extraction은 gradient-based optimization 동안에 여러 번 수행된다.</span> 따라서 새로운 challenge가 생겨나며 specialized approach가 필요해진 것이다.
### Notation
- 우리가 고려하는 모든 방법론은 isosurface를 scalar function $$s: \mathbb{R}^3$$ → $$\mathbb{R}$$에서 extract하는 것이다. 그 scalar function은 regular grid(정규 격자)의 vertex들에서 sampling되고, 각 cell 내부에서 interpolate된다.<span style='color: #a6a6a6'>(scalar function은 grid vertex들에서의 value들로 정의될 수도 있고, underlying neural network의 evaulation으로 정의될 수도 있다. 어쨌든 <span style='color: #a6a6a6'>s</span>의 exact parameterization은 isosurface extraction에 있어선 아무 상관도 없다.)</span>
- set X = grid의 vertex들
- cells C = grid의 cell들
- M=(V, F) = extracted mesh. V는 vertex들, F는 face(면)들.
- v와 x로 각각 logical vertex와 그 vertex의 space에서의 position을 나타낼 것이다.
### <span style='color: #f4acb6'>Marching Cubes(MC)</span>와 Marching Tetrahedra(사면체)
- 가장 direct한 방식. grid의 vertex들이랑 (1개 또는 그 이상의, 각 grid cell 안에 들어있는) mesh faces를 가지고 mesh를 extract하는 것이다.
- <span style='color: #f4acb6'>mesh vertex들은 grid edge(모서리) 위의 점들 중에서 linearly-interpolated scalar function의 부가 바뀌는 지점</span>으로 extract된다. 
$$ u_e = \frac{x_a \cdot s(x_b) - x_b \cdot s(x_a)}{s(x_b) - s(x_a)} $$
- 즉, MC에서는 grid의 edge를 따라 vertex를 생성하고, 그것들을 연결하여 mesh를 형성한다. 따라서 MC에 의해 만들어진 mesh는 격자의 primal connectivity (<-> dual connectivity) 를 따른다.
- <span style='color: #a6a6a6'>이 식은</span> $$s(v_a)=s(v_b)$$<span style='color: #a6a6a6'>일 떄 (differential optimization을 방해할 수 있는) singularity를 가진다고 알려져있지만, extraction 동안엔 이 singular condition이 발생하지 않음이 밝혀졌다.</span>
- 이 방법으로 생성되는 mesh는 <span style='color: #f4acb6'>언제나 self-intersection-free이며 manifold</span>인 것이 보장된다.
- 하지만 이런 marching extraction 방법에서는 <span style='color: #f8dd74'>생성되는 mesh vertex들이 오로지 grid edge라는 작은 부분집합에만 속할 수 있는데, 이는 mesh가 sharp feature에 fitting하지 못하게 한다. 따라서, isosurface가 vertex 근처를 지날 때는 피치 못하게 저퀄리티의 sliver triangles가 생긴다.</span>
- 이 문제에 대한 유망한 해결책은, <span style='color: #f4acb6'>underlying grid vertices의 deform을 허용</span>해주는 것인데, 이러면 performance가 유의미하게 좋아진다. 다만 이때도 extract된 mesh vertex들이 independent하 움직일 수는 없기 때문에, mesh vertex들이 grid상의 degree of freedom 주변으로 몰리는 경우 별 모양의 얇은 삼각형 artifact가 생기는 문제가 있다. <span style='color: #f4acb6'>Deep Marching Tetrahedra (DMTet)</span> 얘기다.
- <span style='color: #f4acb6'>FlexiCubes</span>는 방금의 해결책에서와 마찬가지로 <span style='color: #f4acb6'>grid deformation을 활용</span>하는데, 여기에 더해 <span style='color: #f4acb6'>representation에 degree of freedom을 추가함으로써 vertex들의 independent repositioning을 가능하게 한다!</span> Figure 4에서 확인 가능.
### <span style='color: #f4acb6'>Dual Contouring(DC)</span>
- dual representation을 사용하며, 여기서 extract하는 <span style='color: #f4acb6'>mesh vertex는 일반적으로 grid cell 내부의 어느 곳에든지 위치할 수 있다. 따라서 sharp geometric feature들을 더 잘 캡쳐</span>할 수 있는 것은 당연지사.
- 각 mesh vertex의 position은 다음과 같은 <span style='color: #f4acb6'>local quadratic error function (QEF, 이차 에러함수)를 minimizing하는 값</span>으로 결정된다.
$$\begin{align}
v_d = \arg\min_{v_d} \sum_{u_e \in Z_e} \nabla s(u_e) \cdot (v_d - u_e).
\end{align}$$
- 여기서 $$u_e \in Z_e$$들은 cell edge 위의 점들 중 linearly-interpolated scalar function의 부호가 바뀌는 지점, 일명 zero-crossings이다.
- <span style='color: #f4acb6'>Figure 3의 DC 그림에서, 하얀 점들이 <span style='color: #f4acb6'>u_e</span>에 해당하고 이것들이 QEF의 계산에 사용되는 입력값이다. 한편 QEF를 minimize하여 계산된 vertex들, 즉 v_d가 바로 초록 점들에 해당하며, 이를 dual vertex라고 부른다. → DC에서는 바로 이 초록 vertex들이 grid의 dual connectivity를 형성하며 mesh를 이루게 된다. == DC에서 vertex들은 grid의 dual connectivity를 따른다. == DC에서 QEF에 의해 최적화된 위치에 배치된 vertex들이 서로 연결되어 mesh를 이룬다.</span>
- DC는 fixed scalar function으로부터 mesh 한개를 extract할 땐 <span style='color: #f4acb6'>sharp feature의 fitting에 매우 뛰어나지만</span>, 몇몇 특성이 DC를 <span style='color: #f8dd74'>differential optimization에 쓰기 힘들게</span> <span style='color: #f8dd74'>만든다.</span> 가장 중요한 건 위의 <span style='color: #f8dd74'>Equation 2가 extract된 vertex가 grid cell 안에 있다는 걸 보장해주지 않는다</span>는 점이다.
- 각 $$u_e \in Z_e$$에 대응되는 gradient vector $$\nabla s(u_e)$$들은 모두 같은 평면 위에 있는데 (co-planar) 이것들이 <span style='color: #f8dd74'>degenerate configuration</span>을 만든다. 
- <span style='color: #f8dd74'>degenerate configuration이란, vertex가 멀리 떨어진 위치로 폭발적으로 이동하여 self-intersection과 (formulation을 미분하는 과정에서의) numerically unstable optimization으로 이어지는 상태를 일컫는다.</span>
- 근데 (Equation 2만으로 vertex가 grid cell 안에 있는 걸 보장해주지 못한다고 해서)<span style='color: #f8dd74'>vertex가 cell 안에만 위치하도록 explicitly constrain하면, gradient가 0이 되는 문제가 생기고, 이걸 해결할려고 Equation 2를 regularize(정규화)하면 이번엔 sharp feature에 fit하는 능력이 사라진다.</span> 논문에서는 Figure 2&4를 참고하라고 언급한다. Figure 2의 경우, (1)에선 vertex가 grid cell 밖에 위치할 수 있음을 보여주고, (2)에선 gradient가 0이 되는 문제 상황을 보여주고, (3)에선 strong regularization으로 인해 DC의 flexibility란 장점 (즉 sharp feature에 fit하는 능력)이 줄어든 것을 보여준다. (4)는 FlexiCubes를 보여주는데, 여기선 additional degree of freedom을 도입하여 dual vertex가 그림의 초록색 삼각형 위의 어떤 곳에든 위치할 수 있도록 하고 있다.
- <span style='color: #f8dd74'>한편, 생성된 mesh의 connectivity는 nonmanifold일 수 있으며, output mesh는 (삼각형으로 분할될 때 오류를 유발하는) non-planar quadrilateral(비평면 사변형)들을 포함할 수 있다. (Figure 3 참고)</span>
- DC를 일반화한 최근의 연구들은 Equation 2를 learned neural network로 대체함으로써, 불완전하지만 "fix된" scalar function부터의 extraction quality를 개선하였다. 하지만, <span style='color: #f8dd74'>underlying function에 대해 optimizing을 할 때, 추가적인 neural network에 대해 미분하는 것은 optimization을 더욱 복잡하게 만들어 수렴을 저해하였다.</span> (Chen et al. 2022b 연구 / Figure 4 참고)
- <span style='color: #f4acb6'>FlexiCubes</span>는 이러한 방법론들을 참고하여 각 vertex를 cell 안에서 자유롭게 배치하는 것의 중요성을 깨달았다. 하지만, extracted vertex를 explicit하게 scalar field만의 함수로 배치하는 대신, 저자들은 <span style='color: #f4acb6'>vertex position을 locally 조정하도록 optimize된 additional (carefully-chosen) degrees of freedom을 도입</span>하였다. <span style='color: #f4acb6'>저자들은 manifold(가 아닐 수 있다는) 문제를 해결하기 위해, 그들의 scheme을 DC와 비슷하지만 덜 알려진 Dual Marching Cubes(DMC)를 기반으로 하기로 하였다.</span>
### <span style='color: #f4acb6'>Dual Marching Cubes (DMC)</span>
- DMC도 DC처럼 <span style='color: #f4acb6'>extract된 vertex들이 grid cell 안에 위치한다.</span> (edge 위에 국한 X)
- 하지만, mesh를 <span style='color: #a6a6a6'>(DC에서처럼) "grid의" dual connectivity에 따라 extract하는 대신,</span> <span style='color: #f4acb6'>"Marching Cubes에 의해 extract될 mesh의" dual connectivity에 따라 extract한다</span>는 차이가 있다.
- <span style='color: #f4acb6'>DMC는 필요할 때 하나의 grid cell 내부에 여러 개의 mesh vertex를 생성함으로써 모든 configuration들에 대해 manifold mesh output을 허용한다.</span>
- extract된 vertex의 위치는 { (DC와 비슷한) QEF의 minimizer / 기본 mesh 기하학의 geometric function (면의 중심점 등) } 등으로 정의된다.
- 일반적으로 DMC는 DC에 비해 extract된 mesh의 connectivity를 개선하지만, <span style='color: #f8dd74'>vertex 배치에 QEF를 사용하는 경우 DC의 단점들 중 꽤나 많은 것들을 마주하게 된다.</span>
- <span style='color: #f8dd74'>만약 vertex들이 primal mesh(즉, MC에 의해 만들어지는 mesh)의 centroid(정점) 위치에 배치된다면, freedom이 부족해 개별적인 sharp feature에 fit하지 못하게 된다.</span> <span style='color: #f4acb6'>어쨌든, 이후의 내용에서 저자들이 별도의 설명 없이 DMC를 언급하는 경우 이 centroid approach를 얘기하는 것이다.</span>
- 이게 무슨 말이냐면, <span style='color: #f4acb6'>DMC에서 생성된 vertex들은 MC에서 생성될 mesh의 중심점들에 해당하는 위치에 배치되고, 이 vertex들 간의 연결 관계에 따라 새로운 mesh가 형성된다</span>는 것. <span style='color: #f4acb6'>MC가 생성한 mesh의 vertex들이 가지는 connectivity를 기반으로 새로운 vertex를 배치</span>하므로, <span style='color: #f4acb6'>MC에 의해 extract될 mesh의 dual connectivity에 따라 새로운 vertex를 배치하는 것</span>이라 할 수 있다.
- <span style='color: #f4acb6'>저자들은 DMC를 기반으로 FlexiCubes를 만들었으며, vertex 배치를 위해 additional parameters를 도입함으로써 DMC의 centroid approach를 일반화하였다. difficult configuration에서도 올바른 topology를 생성할 수 있는 scheme를 기반으로 한 것이 FlexiCubes의 성공요인 중 하나이다.</span>

---
## 4. FlexiCubes의 Method
- 저자들은 <span style='color: #f4acb6'>differential mesh optimization</span>을 위해 FlexiCubes라는 representation을 제안하였다.
- 이 method의 핵심은 <span style='color: #f4acb6'>a grid 위에 정의된 a scalar function</span>이며, 저자들은 <span style='color: #f4acb6'>이 함수로부터 DMC를 통해 a triangle mesh를 extract</span>한다.
- 저자들의 main contribution은 <span style='color: #f4acb6'>세 가지 추가적인 parameter set을 제안하여 mesh representation에 flexibility를 더하면서도 robustness와 optimization 용이성은 유지</span>한 것이다.
- Parameter 1 = <span style='color: #f4acb6'>Interpolation weights</span> $$\alpha \in \mathbb{R}^8_{>0}$$, $$\beta \in \mathbb{R}^{12}_{>0}$$ / per grid cell / dual vertex들을 배치하기 위함
- Parameter 2 = <span style='color: #f4acb6'>Splitting weights</span> $$\gamma \in \mathbb{R}_{>0}$$ / per grid cell / quadrilateral(사각형)이 삼각형으로 분할되는 방법을 결정하기 위함
- Parameter 3 = <span style='color: #f4acb6'>Deformation vectors</span> $$\delta \in \mathbb{R}^3$$ / per vertex of the underlying grid / spatial alignment를 위함
- 이 파라미터들은 scalar function $$s$$와 함께 auto-differentiation을 통해 optimize됨으로써 mesh를 desired objective에 fit되게 한다.
- 또한 저자들은 FlexiCubes representation의 extension을 통해 volume에 대해 사면체형(thetrahedral) mesh를 extract하거나 adaptive resolution을 가지는 hierarchical mesh를 represent하는 방법을 제시하였다.

### 4.1 Dual Marching Cubes Mesh Extraction
- FlexiCubes의 시작은, 각 grid vertex $$x$$에서의 scalar function value인 $$s(x)$$를 기반으로 <span style='color: #f4acb6'>DMC mesh의 connectivity를 extract</span>하는 것이다. ($$v_d$$의 connectivity)
- Figure 7에서 볼 수 있듯, $$\color[rgb]{0.956, 0.675, 0.714}{s(x)}$$<span style='color: #f4acb6'>의 </span><span style='color: #f4acb6'>cube corners에서의 부호</span>가 conectivity와 adjacency 관계를 결정한다. MC에서 그랬던 것처럼 edge 상에서의 interpolation을 통해 $$u_e$$ 점들을 얻는 것으로 시작한다.
- $$u_e$$들의 위치가 결정된 후 → $$u_e$$ 여러개를 포함하는 primal face들이 정해지고 → 그 primal face의 centroid 지점이 $$v_d$$가 된다. (= Equation 4)
- <span style='color: #a6a6a6'>(vertex를 grid edge 위에서만 extract하는 MC와 달리)</span> <span style='color: #f4acb6'>DMC는 vertex를 cell의 각 primal face마다 1개씩 뽑는다. 이때 primal face는 grid의 한 면을 의미하는 것이 아니라 Figure 7에서 색칠된 다각형들에 해당하고, cell마다 보통 1개, 최대 4개 존재한다. 따라서 cell 한개당 vertex 개수도 1~4개가 된다.</span> (cell 한개에 primal face가 4개인 경우는 Figure 7의 case C13 참고)
- <span style='color: #f4acb6'>adjacent cells 안의 extracted vertice(v_d)들은 edge로 link되는데, 결과적으로 4개의 neighboring dual vertices들로 이루어진 사각형의 면들이 dual mesh를 이루게 된다.</span> (Figure 5를 보면 초록색 $$v_d$$ 점 4개가 모여 사각형 면을 이루고 있다.)
- 결과적으로 생성된 mesh는 manifold임이 보장된다. 다만, 후술할 additional degrees of freedom으로 인해 mesh는 드물게 self-intersection을 포함할 수 있다. (Section 7.2 참고)
### 4.2 Flexible Dual Vertex Positioning & <span style='color: #f4acb6'>Interpolation Weights</span> $$\color[rgb]{0.956, 0.675, 0.714}{\alpha}$$<span style='color: #f4acb6'>, </span>$$\color[rgb]{0.956, 0.675, 0.714}{\beta}$$

- FlexiCubes는 extracted mesh vertex의 위치를 계산하는 방식에 있어서 DMC를 일반화한다.
- 먼저 <span style='color: #a1d4cf'>일반화 전의 그냥 DMC</span>에 대해 복습해보자. MC에서 primal vertex $$u_e$$들이 grid cell edge 상의 scalar zero-crossing 지점에 위치했었다. 한편, original DMC에서는 {MC에서 extract되는 mesh의 dual connectivity}에 따라 mesh를 extract했다. 다시 말해, original DMC에서는 MC 방식으로 구한 $$u_e$$와 그에 따라 정해진 primal face를 가지고, 각각의 extracted vertex $$v_d$$를 primal face의 centroid로 결정한다. 이 과정은 다음의 Equation 4와 같으며, 이때 $$V_E$$는 crossings $$u_e$$의 집합, 즉 primal face vertices 집합이다. 
$$v_d = \frac{1}{|V_E|} \sum_{u_e \in V_E} u_e$$
- <span style='color: #f4acb6'>DMC에 additional flexibility를 도입</span>하기 위해, 저자들은 먼저 <span style='color: #f4acb6'>각 grid cell마다 </span>$$\color[rgb]{0.956, 0.675, 0.714}{\alpha \in \mathbb{R}^8_{>0}}$$<span style='color: #f4acb6'>라는 weight 집합을 정의한다. </span>$$\color[rgb]{0.956, 0.675, 0.714}{\alpha}$$<span style='color: #f4acb6'>는 </span><span style='color: #f4acb6'>각 cube corner에 양의 스칼라 값을 하나씩 연결짓는다. </span>따라서 $$\alpha$$는 각 edge 위에 있는 <span style='color: #f4acb6'>crossing point</span> $$\color[rgb]{0.956, 0.675, 0.714}{c_e}$$<span style='color: #f4acb6'>의 위치를 조정한다.</span> 그러면 $$\color[rgb]{0.956, 0.675, 0.714}{u_e}$$<span style='color: #f4acb6'>의 위치</span>를 결정하던 Equation 3은 다음의 Equation 5로 대체된다.
$$ u_e = \frac{s(x_i) \alpha_i x_j - s(x_j) \alpha_j x_i}{s(x_i) \alpha_i - s(x_j) \alpha_j}
$$
- 저자들의 구현에서는, $$\alpha$$를 [0,2]로 제한하기 위해 $$\tanh(\cdot) + 1$$ activation function을 적용했으며, degeneracy(퇴화)로 인한 convergence problem은 관찰되지 않았다고 한다.
- 한편, 위와 마찬가지로, <span style='color: #a6a6a6'>original DMC에서처럼 dual vertex를 그냥 naive하게 primal face의 centroid에 위치시키기보다는,</span> 저자들은 <span style='color: #f4acb6'>각 grid cell마다 </span>$$\color[rgb]{0.956, 0.675, 0.714}{\beta \in \mathbb{R}^{12}_{>0}}$$<span style='color: #f4acb6'>라는 weight 집합을 정의한다. </span>$$\color[rgb]{0.956, 0.675, 0.714}{\beta}$$<span style='color: #f4acb6'>는 각 cube edge에 양의 스칼라 값을 하나씩 연결짓는다. (</span>$$\color[rgb]{0.956, 0.675, 0.714}{\alpha}$$<span style='color: #f4acb6'>는 cube corner에, </span>$$\color[rgb]{0.956, 0.675, 0.714}{\beta}$$<span style='color: #f4acb6'>는 cube edge에!) </span>따라서 $$\color[rgb]{0.956, 0.675, 0.714}{\beta}$$<span style='color: #f4acb6'>는 </span><span style='color: #f4acb6'>각 face 안에서 dual vertex의 위치를 조정</span>하는 역할을 한다. 그러면 $$\color[rgb]{0.956, 0.675, 0.714}{v_d}$$<span style='color: #f4acb6'>의 위치</span>를 결정하던 Equation 4는 다음의 Equation 6으로 대체된다.
$$v_d = \frac{1}{\Sigma_{u_e \in V_E} \beta_e} \sum_{u_e \in V_E} \beta_e u_e$$
- 실제로, 저자들은 $$\alpha$$에 대해 그랬던 것처럼 $$\beta$$에 대해서도 $$\tanh(\cdot) + 1$$ activation을 적용하여 range를 제한하였다.
- 종합하면, 추가적으로 가중치 $$\alpha \in \mathbb{R}^8_{>0}$$와 $$\beta \in \mathbb{R}^{12}_{>0}$$를 도입하는 것은 각 grid cell마다 scalar parameter를 20개씩 추가하는 것이다. 또, $$\alpha$$와 $$\beta$$ 둘다에 있어서 모든 weight들은 cell마다 independent하게 정의되어 adjacent corners or edges에서도 공유되지 "않는다". 이러한 independent weights는 더 많은 flexibility를 제공하며, 우리의 dual setting에서는 adjacent elements들 간에 유지해야 할 continuity condition 같은 건 존재하지 않는다.
- Equation 5와 Equation 6은 둘다 의도적으로 convex combination으로 parameterize된 것이다. 따라서, 결과적으로 추출된 vertex position은 반드시 {그 vertex가 위치하는 grid cell의 vertex들의} convex hull (볼록 껍질) 내에 있게 된다.
- 또, Figure 7에서와 같이 convex cell이 multiple dual vertices를 생성할 때, dual vertices가 위치하는 corresponding primal faces는 서로 교차하지 않는데, 그에 따라 resulting mesh에서는 거의 모든 self-intersection이 방지된다. (Section 7 및 Supplement 참고)
### 4.3 Flexible Quad Splitting & <span style='color: #f4acb6'>Splitting Weights</span> $$\color[rgb]{0.956, 0.675, 0.714}{\gamma}$$

- Dual Marching Cubes, 그리고 따라서 FlexiCubes도, 순수한 <span style='color: #f4acb6'>사각형(quadrilateral) 메쉬</span>를 추출하는데, 이 메쉬는 <span style='color: #f4acb6'>non-planar 면들로 구성되어 있으며, 이는 보통 후속 application에서 처리하기 위해 삼각형으로 분할된다.</span> 
- 임의의 대각선으로 단순히 분할하면 곡면 영역에서 상당한 아티팩트를 발생시킬 수 있으며(Figure 8 참고), non-planar 사각형을 분할하여 unknown geometry를 표현하기 위한 단일한 이상적인 방법은 일반적으로 존재하지 않는다. <span style='color: #f4acb6'>우리의 다음 parameter는 split의 선택을 유연하게 만들고, 이를 continuous degree of freedom의 하나로 optimize하기 위해 도입되었다.</span>
- 우리는 <span style='color: #f4acb6'>각 grid cell마다 weight </span>$$\color[rgb]{0.956, 0.675, 0.714}{\gamma \in \mathbb{R}_{>0}}$$<span style='color: #f4acb6'>를 정의</span>하며, $$\gamma$$는 (추출된 mesh에 포함되는) <span style='color: #f4acb6'>생성된 vertex들(cell마다 1~4개)로 propagate된다.</span> optimization-time "한정으로", mesh의 각 사각형 면은 하나의 midpoint vertex $$\overline{v_d}$$를 insert함으로써 4개의 삼각형으로 쪼개진다. (Figure 8 참고) 이때 이 midpoint의 위치는 다음의 Equation 7에 따라 결정되며, notation은 Figure 8에서와 동일하다.
$$\color[rgb]{0.956, 0.675, 0.714}{ \overline{v_d} = \frac{\gamma_{c_1} \gamma_{c_3} \left( v_d^{c_1} + v_d^{c_3} \right) / 2 + \gamma_{c_2} \gamma_{c_4} \left( v_d^{c_2} + v_d^{c_4} \right) / 2}{\gamma_{c_1} \gamma_{c_3} + \gamma_{c_2} \gamma_{c_4}} }$$
- 이는 면의 두 가지 가능한 대각선의 중간점들의 weighted combination이며, weights는 해당 vertex에 대한 $$\gamma$$ 매개변수에서 가져온다. 직관적으로, $$\gamma$$ 가중치를 조정하면 두 가지 가능한 분할에서 발생하는 geometry들의 사이를 smoothly interpolate하게 된다. $$\gamma$$ 값을 optimize하면 objective of interest에 fit하는 split을 선택하는 데 도움이 된다. Optimization이 완료된 후, <span style='color: #f4acb6'>final extraction 시에는, midpoint vertex</span> $$\overline{v_d}$$<span style='color: #f4acb6'>를 삽입하지 "않고", 단순히 $$\gamma$$ 값의 곱이 더 큰 대각선을 따라 각 사각형을 분할한다.</span>
### 4.4 Flexible Grid Deformation & <span style='color: #f4acb6'>Deformation Vectors</span> $$\color[rgb]{0.956, 0.675, 0.714}{\delta}$$

- DefTet와 DMTet에서 영감을 받아, 저자들은 또한 각 grid vertex들에서 $$\delta \in \mathbb{R}^3$$ 변위에 따라 underlying grid의 vertex들이 변형되도록 허용한다. 
- 이러한 변형은 grid가 thin feature들과 locally align될 수 있게 하며, vertex 위치 설정에 추가적인 유연성을 제공한다. 
- 저자들은 grid cell이 절대로 invert되지 않도록, 변형을 최대, grid spacing의 절반으로 제한한다.
### <span style='color: #a6a6a6'>4.5 (Extension) Tetrahedral(사면체형) Mesh를 Extract하는 법</span>

### <span style='color: #a6a6a6'>4.6 (Extension) Adaptive Resolution을 가지는 Hierarchical Mesh를 Extract하는 법</span>

## <span style='color: #a6a6a6'>5. Experiments</span>
---
## 6. Applications
### 6.3 <span style='color: #f4acb6'>3D Mesh Generation</span>
- 3D 컨텐츠 생성 촉진을 목표로 하는 3D 메쉬 생성은 컴퓨터 그래픽스와 비전에서 중요한 작업이며, 게임 및 소셜 플랫폼과 같은 산업에 이익을 준다. 
- 최근의 3D 생성 모델들은 3D 표현을 2D 이미지로 미분 가능하게 렌더링하며, 고전적인 GAN 프레임워크와 결합하여 2D 이미지 supervision만으로 3D 컨텐츠를 합성한다. 
- 최신의 GET3D는 미분 가능한 isosurfacing 모듈인 DMTet에 의해 고품질의 텍스처가 포함된 3D 메쉬를 직접 합성한다.
- 이 application에서, 저자들은 FlexiCubes가 3D 생성 모델에서 plug-and-play 방식으로 사용할 수 있는 미분가능한 메쉬 추출 모듈로 작용할 수 있으며, 메쉬 품질을 크게 향상시킬 수 있음을 시연한다. 
- 구체적으로, 저자들은 GET3D를 사용하고 메쉬 추출 단계에서 DMTet을 FlexiCubes로 대체한다. 
- 저자들은 GET3D의 3D 생성기에서 마지막 레이어만 수정하여 FlexiCubes의 각 cube에 대해 21개의 가중치를 추가로 생성하도록 했다. (<span style='color: #f4acb6'>$$\alpha$$, $$\beta$$ and $$\gamma$$</span>)
- 훈련 절차, 데이터셋(ShapeNet 사용) 및 GET3D의 다른 하이퍼파라미터들은 변경하지 않았다.
- Figure 23과 Table 6을 통해 질적 비교와 정량적 결과가 제시되어 있다. FlexiCubes는 모든 카테고리에서 더 나은 FID 점수를 달성하여 3D 모델 생성에서 더 높은 역량을 입증한다. 질적으로는, FlexiCubes 버전의 GET3D를 사용하여 생성된 형태들이 훨씬 더 높은 품질을 가지며, 디테일이 더 많고, sliver 삼각형은 덜 포함하고 있다.

---
## <span style='color: #a6a6a6'>7. Discussion</span>
