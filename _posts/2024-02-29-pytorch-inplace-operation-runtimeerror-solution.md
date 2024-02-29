---
layout: post
title: "Solution to Pytorch's RuntimeError about 'inplace operation'"
# description: >
#   이 논문을 읽어보았습니당
sitemap: true
hide_last_modified: false
category: [error]
---

파이토치를 이용해 코드를 수정해가며 실험을 하는 과정에서 다음과 같은 에러가 발생하였다.

> RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [1, 3, 32, 32]] is at version 10; expected version 9 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
{:.lead}

Chat GPT와 copilot, 구글링 등의 수단을 총동원해 보았지만, 주로 in place operation을 제거하라는 조언밖에 얻을 수 없었다.

내 코드에서는 in place operation을 사용하지 않고 있었기에 사실상 무의미한 조언이었다.

detach를 써보라는 블로그가 있었지만, 내 코드 상 detach를 추가할 만한 곳은 없었다. 또, detach의 역할이 computational graph를 끝내버리는 것임을 고려하면 에러를 피하고자 무지성으로 detach를 난사하는 것은 위험해 보이기도 했다.

결국 예전에 딥러닝의 기초 과목 과제를 하다가 비슷한 상황에서 `.clone()`을을 통해 문제를 해결했던 게 기억났다. 왜 `.clone()`을 해야만 에러가 안 뜨는 건지 qna도 올렸었던 기억이 난다. 아래는 그때 조교님께서 남겨주신 답변을 찾아와 요약한 것이다. 과제 마감이 얼마 남지 않았을 때 올린 질문인데 길고 자세한 답변을 남겨주셔서 정말 죄송하고도 감사했다.

- pytorch의 backprop과 in-place operation에 대한 내용이 매우 복잡하고 그것에 비해 documentation이 너무 부족하다
- `.clone()`은 tensor의 값 뿐 아니라 gradient에 대한 정보까지 모두 복사하기 때문에, `.clone()`을 통해 backprop이 끊어지는 것은 아니다
- 내가 첨부한 코드에서 `.clone()`을 제거하였을 때, 즉 에러를 유발하는 상태일 때, h[batch]와 그것을 계산하는 데에 사용되는 pre-activation(self.nonlinearity를 통과하기 전 linear module forward 결과)의 memory address를 확인해보자. 정확한 원인은 모르겠지만, h[batch]에서 pre-activation을 계산할 때 memory address가 바뀌지 않는 in-place operation이 일어나는 것을 확인할 수 있다. 아마 이 부분에서 해당 에러가 발생했을 것이다.
![image](/assets/img/blog/2024-02-29-inplace/before.png){:.lead loading="lazy"}
- 반면, `.clone()`을 추가한 경우, h[batch]의 address는 바뀌지 않지만, pre-activation의 address는 모두 다르게 저장된다. Backprop 과정에서는 h[batch]`.clone()`만이 사용되기 때문에 이 경우 에러가 해결되는 듯하다.
![image](/assets/img/blog/2024-02-29-inplace/after.png){:.lead loading="lazy"}

결과적으로, 이번에도 코드에서 사용된 모든 텐서에 `.clone()`을 붙였더니 이 에러가 사라졌다. 그 후, `.clone()`을 횟수를 최소화하기 위해 코드를 여러 번 실행시켜 보면서 소거법을 통해 어떤 위치에서 `.clone()`을 뗐을 때 에러가 생기는지를 파악하고 그 밖의 `.clone()`을은 모두 제거했다. (이때 epoch, hidden_dim 등의 하이퍼파라미터를 최소화하면 무의미한 연산량을 줄일 수 있었다!) 내 코드에서는 단 한 번의 `.clone()`을이 에러 여부를 판가름하는 key였다.

어찌보면 단순무식한 해결방법이지만, 솔직히 원인불명의 에러에 명확한 해결방법은 존재한다면, 어제의 나처럼 마냥 답답해하며 스트레스 속에 시간을 허비하기보다 이렇게라도 해결방법을 찾아내 다음 단계로 넘어가는 것이 조금 더 시간을 절약할 수 있는 방법이 아닌가 싶어 기록을 남긴다.

