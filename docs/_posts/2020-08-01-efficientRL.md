---
title: "Efficient Value-Based RL"
excerpt: "Discussion on several recent works trying to improve sample efficiency of reinforcement learning algorithms."
categories:
  - Reinforcement Learning
tags:
  - Value-Based RL
---

## Introduction

We briefly summarize several papers trying to improve data efficiency in RL. Notice that these methods are favored when the number of environment interactions is limited (e.g. 100K environment steps). In long terms, these methods may eventually impair the performance.

## [CURL](#ref1)

CURL employs two techniques to speed up existing model-free algorithms: 1) random cropping, which randomly crops an \\(84\times 84\\) images from a \\(100\times 100\\) input image, and 2) contrastive learning, which trains another head on top of the image embedding using InfoNCE from [CPC]({{ site.baseurl }}{% post_url 2018-09-27-CPC %}).

CURL employs the bi-linear inner-product \\(sim(q,k)=q^TWk\\), where \\(q\\) and \\(k\\) are the anchor and target(a.k.a., query and key), \\(W\\) is a learnable parameter matrix. Following the same vein as MoCo, \\(q\\) is obtained from an online encoder while \\(k\\) is obtained from a target network with momentum update. Additionally, when cooperating with random cropping, we use different crops as the inputs to the online and target networks. The following code demonstrates the process

```python
x_q = aug(x)	# apply data augmentation
x_k = aug(x)	# apply different data augmentation
z_q = encoder(x_q)
z_k = target_encoder(x_k)
z_q = mlp(z_q)
z_k = mlp(z_k)
z_k = tf.stop_gradient(z_k)
Wk = tf.matmul(W, tf.transpose(z_k))
logits = tf.matmul(z_q, Wk)			# logits is a [B, B] matrix, where positives are on the main diagonal
labels = tf.range(batch_size)		# index of elements on the main diagonal
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
loss = tf.reduce_mean(loss)
```

Notice that when we train the network using augmented images, it is important to apply the same data augmentation method to images during rollouts. Otherwise, the agent may suffer from distribution mismatch. This is unnoticed in the CURL paper, but is later studied by several works.

## [RAD](#ref2)

The authors experiment a variety of data augmentation techniques on Deepmind control suite and find random crop performs best. It's worth noting that RAD with only random crop performs even better than CURL, suggesting that data augmentation alone is sufficient for efficient reinforcement learning.

## [DrQ](#ref3)

The authors experiment a variety of data augmentation techniques on Deepmind control suite and find random shift(with nearest padding) performs best. Notice that random shift differs from random crop in that it keeps the image size unchanged and pads points outside the boundaries using nearest points.

In order to reduce the variance introduced by data augmentations, DrQ modifies the \\(Q\\) loss as follows

$$
\begin{align}
\mathcal L=\mathbb E_{s,a,s'}\left[{1\over N}\sum_{i=1}^N(Q(f^i(s),a)-(r+\gamma{1\over M}\sum_{j=1}^M \max _{a'}Q(f^j(s'), a')))\right]
\end{align}
$$

where \\(f\\) applies a data augmentation technique to \\(s\\). Experiments shows that \\(N=2, M=2\\) indeed reduce the variance on several environments.

## [MPR](#ref4)

Momentum Predictive Representations(MPR) adds a convolutional transition model upon the convolutional encoder. The transition model \\(h\\) consists of two convolutional layers with 64 \\(3\times 3\\) filters, taking as inputs the spatial representation learned by the encoder and actions one-hot encoded as a sets of planes. The output of the transition model \\(z=h(x)\\) is then passed to a projection head, mapping into a smaller latent space \\(\hat y=g(z)\\), where \\(g\\) is an MLP. The authors reuse the FC layers after the encoder in their experiments, i.e., \\(\hat y\\) is the concatenation of outputs of the first layers of value and advantage heads. The prediction loss is computed from cosine similarities between predicted and observed representations in the next \\(k\\) steps

$$
\begin{align}
\mathcal L(s_{t:t+K},a_{t:t:K})=-\sum_{k=1}^K \ell_2(y_{t+k})^\top \ell_2(\hat y_{t+k})
\end{align}
$$

 where \\(\ell_2\\) is the l2 normalization function and \\(y\\) is the computed from a momentum encoder and projection head:

$$
\begin{align}
h_m=\tau h_m+(1-\tau)h\\\
g_m=\tau g_m+(1-\tau)g
\end{align}
$$

MPR also uses the same set of image augmentations as DrQ, comprised of small random shifts and color jitter. They found it important to normalize activations to lie in \\([0,1]\\) at the output of the convolutional encoder and transition model when using augmentation.

## References

<a name="ref1"></a>Srinivas, Aravind, Michael Laskin, and Pieter Abbeel. 2020. “CURL: Contrastive Unsupervised Representations for Reinforcement Learning.” http://arxiv.org/abs/2004.04136.

<a name="ref2"></a>Kostrikov, Ilya, Denis Yarats, and Rob Fergus. 2020. “Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels.” http://arxiv.org/abs/2004.13649.

<a name="ref3"></a>Laskin, Michael, Kimin Lee, Adam Stooke, Lerrel Pinto, Pieter Abbeel, and Aravind Srinivas. 2020. “Reinforcement Learning with Augmented Data.” http://arxiv.org/abs/2004.14990.

<a name="ref4"></a>Schwarzer, Max, Ankesh Anand, Rishab Goel, R Devon Hjelm, Aaron Courville, and Philip Bachman. 2020. “Data-Efficient Reinforcement Learning with Momentum Predictive Representations.” http://arxiv.org/abs/2007.05929.