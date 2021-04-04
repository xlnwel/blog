---
title: "Probability Measures in Deep Learning"
excerpt: "We scratch several probability measures commonly seen in deep learning."
categories:
  - Mathematics
tags:
  - Mathematics
---

## Kullback-Leibler Divergence


$$
\begin{align}
D_{KL}(p\Vert q) = \int p(x) \log{p(x)\over q(x)}dx
\end{align}
$$


Forward KL—moment-matching: Drawing \\(q(x)\\) close to \\(p(x)\\)

Reverse KL—mode-seeking: Drawing \\(p(x)\\) close to \\(q(x)\\)

https://wiseodd.github.io/techblog/2016/12/21/forward-reverse-kl/

A noticeable attribute of KL divergence is that when \\(p\\) closes to zero, and \\(q\\) is significantly non-zero, the effect of \\(q\\) is negligible. 

- In the case of forward KL, this will cause \\(q\\) to cover wherever \\(p\\) is large, resulting in a diffuse shape. That's why forward KL is known as *zero avoiding*, as it is avoiding \\(q(x)=0\\) whenever \\(p(x)>0\\). 

- On the other hand, in the case of reverse KL, this pushes \\(p(x)\\) close to \\(q(x)\\) when \\(p(x)>0\\) and \\(p(x)=0\\) elsewhere. Therefore, \\(p(x)\\) will put most of its mass on the mode of \\(q(x)\\) and zero elsewhere. That's why backward KL is known as *zero forcing*, as it force \\(p(x)\\) to be zero on somewhere.

## Jensen-Shannon Divergence


$$
\begin{align}
JS(p, q)={1\over 2}D_{KL}\left(p\Vert {(p+q)\over 2}\right)+{1\over 2}D_{KL}\left(q\Vert {(p+q)\over 2}\right)
\end{align}
$$


### JS in GANs

One of the most common use of JS divergence in deep learning is GANs, in which we have the following loss

$$
\begin{align}
L(G,D)=\int_x\Big(p_r(x)\log D(x)+p_g(x)\log(1-D(x))\Big)dx
\end{align}
$$

when we have the optimal discriminator \\(D^*={p_r(x)\over p_r(x)+p_g(x)}\\), \\(L(G, D^*)\\) becomes

$$
\begin{align}
L(G,D^*)&=\int_x\left(p_r(x)\log {p_r(x)\over p_r(x)+p_g(x)}+p_g(x)\log{p_g(x)\over p_r(x)+p_g(x)}\right)dx\\\
&=\int_x\left(p_r(x)\log {p_r(x)\over {(p_r(x)+p_g(x))\over 2}}+p_g(x)\log{p_g(x)\over {(p_r(x)+p_g(x))\over 2}}\right)dx-2\log 2\\\
&=2JS(p,q)-2\log 2
\end{align}
$$


## Total Variation

TV divergence is the maximum probability difference between two distribution

$$
\begin{align}
\delta(p, q)=\sup_{A\in\Sigma}|p(A)-q(A)|
\end{align}
$$


## Wasserstein Distance

Wasserstein distance is the \\(L_k\\) between points on inverse cumulative distribution function(CDF)

$$
\begin{align}
W_k(U,V):=\left(\int_0^1\left|F_U^{-1}(z)-F_V^{-1}(z)\right|^kdz\right)^{1/p}
\end{align}
$$

Where \\(U\\) and \\(V\\) are random variables, \\(z\\) is the cumulative probability. \\(F^{-1}_U:z\rightarrow U\\) is the quantile function(inverse CDF). The following figure demonstrates the 1-Wasserstein distance between a random distribution and a uniform Dirachlet distribution. The 

<figure>
  <img src="{{ '/images/distributional/wasserstein.png' | absolute_url }}" alt="" width="1000">
  <figcaption></figcaption>
  <style>
    figure figcaption {
    text-align: center;
    }
  </style>
</figure> 