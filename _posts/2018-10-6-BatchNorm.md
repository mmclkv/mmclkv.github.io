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
  
  
  