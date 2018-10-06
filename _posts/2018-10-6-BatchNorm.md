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

Batch Normalization的梯度推导相对有点复杂，原论文写得比较简略，只是直接给出了结果（而且还有一个为0的项作者忘了约掉）。其实画个DAG，推起来相当地舒服：
![bn1](/_images/bn1.jpg)
![bn2](/_images/bn2.jpg)
![bn3](/_images/bn3.jpg)
![bn4](/_images/bn4.jpg)
![bn5](/_images/bn5.jpg)
![bn6](/_images/bn6.jpg)
![bn7](/_images/bn7.jpg)
![bn8](/_images/bn8.jpg)
  
  
  