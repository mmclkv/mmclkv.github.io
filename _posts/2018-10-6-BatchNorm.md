---
layout: post
title: Batch Normalization梯度公式推导（附Pytorch实现解析）
date: 2018-10-06 17:05:00
categories:
- 深度学习
tags:
- Batch Normalization
- 深度学习
---

Batch Normalization的梯度推导相对有点复杂，原论文写得比较简略，只是直接给出了结果（而且还有一个为0的项作者忘了约掉），其实画个DAG，推起来相当地舒服。以下8张图，完整地展示了BN梯度推导的过程（为求简化省略了加在方差上的无穷小量\\(\epsilon\\)）：
![bn1](/public/images/bn1.jpg)
![bn2](/public/images/bn2.jpg)
![bn3](/public/images/bn3.jpg)
![bn4](/public/images/bn4.jpg)
![bn5](/public/images/bn5.jpg)
![bn6](/public/images/bn6.jpg)
![bn7](/public/images/bn7.jpg)
![bn8](/public/images/bn8.jpg)
  
其中会被用到的是\\(\frac{\partial L}{\partial x_i}\\)，\\(\frac{\partial L}{\partial w}\\)，\\(\frac{\partial L}{\partial b}\\)三个项。看完了公式推导之后，我们再来看看Pytorch里的实现：

```javascript

void THNN_(BatchNormalization_backward)(
  THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradInput,
  THTensor *gradWeight, THTensor *gradBias, THTensor *weight,
  THTensor *running_mean, THTensor *running_var,
  THTensor *save_mean, THTensor *save_std,
  bool train, double scale, double eps)
{
  THNN_CHECK_SHAPE(input, gradOutput);
  int64_t nInput = THTensor_(size)(input, 1);
  int64_t f;
  ptrdiff_t n = THTensor_(nElement)(input) / nInput;

  if (gradInput) {
    THTensor_(resizeAs)(gradInput, input);
  }

  #pragma omp parallel for
  for (f = 0; f < nInput; ++f) {
    THTensor *in = THTensor_(newSelect)(input, 1, f);
    THTensor *gradOut = THTensor_(newSelect)(gradOutput, 1, f);
    scalar_t w = weight ? THTensor_(get1d)(weight, f) : 1;
    scalar_t mean, invstd;
    if (train) {
      mean = THTensor_(get1d)(save_mean, f);
      invstd = THTensor_(get1d)(save_std, f);
    } else {
      mean = THTensor_(get1d)(running_mean, f);
      invstd = 1 / sqrt(THTensor_(get1d)(running_var, f) + eps);
    }

    // sum over all gradOutput in feature plane
    accreal sum = 0;
    TH_TENSOR_APPLY(scalar_t, gradOut, sum += *gradOut_data;);

    // dot product of the Q(X) and gradOuput
    accreal dotp = 0;
    TH_TENSOR_APPLY2(scalar_t, in, scalar_t, gradOut,
      dotp += (*in_data - mean) * (*gradOut_data););

    if (gradInput) {
      THTensor *gradIn = THTensor_(newSelect)(gradInput, 1, f);

      if (train) {
        // when in training mode
        // Q(X) = X - E[x] ; i.e. input centered to zero mean
        // Y = Q(X) / σ    ; i.e. BN output before weight and bias
        // dL/dX = (Q(dL/dY) - dot(Y, dL/dY) * Y) / σ * w

        // projection of gradOutput on to output scaled by std
        scalar_t k = (scalar_t) dotp * invstd * invstd / n;
        TH_TENSOR_APPLY2(scalar_t, gradIn, scalar_t, in,
          *gradIn_data = (*in_data - mean) * k;);

        accreal gradMean = sum / n;
        TH_TENSOR_APPLY2(scalar_t, gradIn, scalar_t, gradOut,
          *gradIn_data = (*gradOut_data - gradMean - *gradIn_data) * invstd * w;);

      } else {
        // when in evaluation mode
        // Q(X) = X - running_mean  ; i.e. input centered to zero mean
        // Y = Q(X) / running_std    ; i.e. BN output before weight and bias
        // dL/dX = w / running_std
        TH_TENSOR_APPLY2(scalar_t, gradIn, scalar_t, gradOut,
          *gradIn_data = *gradOut_data * invstd * w;);
      }

      c10::raw::intrusive_ptr::decref(gradIn);
    }

    if (gradWeight) {
      scalar_t val = THTensor_(get1d)(gradWeight, f);
      THTensor_(set1d)(gradWeight, f, val + scale * dotp * invstd);
    }

    if (gradBias) {
      scalar_t val = THTensor_(get1d)(gradBias, f);
      THTensor_(set1d)(gradBias, f, val + scale * sum);
    }

    c10::raw::intrusive_ptr::decref(gradOut);
    c10::raw::intrusive_ptr::decref(in);
  }
}

```
首先，对于图像分类等任务，每一个Batch的输入是一个四维的Tensor，其shape为(N, C, H, W)。BN是在C这个维度上做的，在代码中用变量nInput表示。因此，整个函数的主要计算逻辑都包在一个循环之中，循环变量f从0遍历到nInput - 1，每次取出Input[:, f, :, :]和gradOutput[:, f, :, :]，计算\\(\frac{\partial L}{\partial x_i\\)（这里的x就是Input[:, f, :, :]，i的取值范围为从0到NHW-1），\\(\frac{\partial L}{\partial w_f}\\)，\\(\frac{\partial L}{\partial b_f}\\)。在计算这三个变量的梯度的时候有一个小优化：注意到

$\frac{\partial L}{\partial w}=\sum_{j=1}^{n}\frac{\partial L}{\partial y_j}\cdot \hat{x}_j=\frac{1}{\sigma}\sum_{j=1}^{n}\frac{\partial L}{\partial y_j}\cdot(x_j - \mu)$

此时可以看到在\\(\frac{\partial L}{\partial x_i}\\)和\\(\frac{\partial L}{\partial b_f}\\)中均出现了\\(\sum_{j=1}^{n}\frac{\partial L}{\partial y_j}\\)这个量，在\\(\frac{\partial L}{\partial x_i}\\)和\\(\frac{\partial L}{\partial b_f}\\)中均出现了\\(\sum_{j=1}^{n}\frac{\partial L}{\partial y_j}\cdot(x_j - \mu)\\)这个量，因此先提前计算好这两个量，分别存放在sum变量和dotp变量中：

``` javascript
// sum over all gradOutput in feature plane
accreal sum = 0;
TH_TENSOR_APPLY(scalar_t, gradOut, sum += *gradOut_data;);

// dot product of the Q(X) and gradOuput
accreal dotp = 0;
TH_TENSOR_APPLY2(scalar_t, in, scalar_t, gradOut,
  dotp += (*in_data - mean) * (*gradOut_data););
```

在计算梯度的时候直接使用sum和dotp，减少了一次重复计算：

``` javascript
// 计算x_i的梯度，使用了dotp和sum
scalar_t k = (scalar_t) dotp * invstd * invstd / n;
TH_TENSOR_APPLY2(scalar_t, gradIn, scalar_t, in,
  *gradIn_data = (*in_data - mean) * k;);

accreal gradMean = sum / n;
TH_TENSOR_APPLY2(scalar_t, gradIn, scalar_t, gradOut,
  *gradIn_data = (*gradOut_data - gradMean - *gradIn_data) * invstd * w;);
```

``` javascript
// 计算w的梯度，使用了dotp
scalar_t val = THTensor_(get1d)(gradWeight, f);
  THTensor_(set1d)(gradWeight, f, val + scale * dotp * invstd);
```

``` javascript
// 计算w的梯度，使用了sum
scalar_t val = THTensor_(get1d)(gradWeight, f);
  THTensor_(set1d)(gradWeight, f, val + scale * dotp * invstd);
```