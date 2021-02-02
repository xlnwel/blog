---
title: "Generalization in RL"
excerpt: "In which we discuss several recent works trying to improve generalization in deep reinforcement learning."
categories:
  - Reinforcement Learning
tags:
  - Reinforcement Learning
  - Generalization in Reinforcement Learning
---

## Introduction

We briefly summarize several recent papers that focus on generalization in reinforcement learning

## IBAC-SNI

[Igl et al. 2019](#ref1) propose two methods to improve generalization in reinforcement learning. We briefly discuss them sequentially in the following sub-sections

### Selective Noise Injection(SNI)

Igl et al. first argue that noise injected by regularization methods such as dropout and batch normalization can deteriorate the agent's performance and therefore affect the data distribution. Furthermore, such stochasticity can destabilize the training through the target critic or importance sampling. Therefore, Igl et al. propose to remove such noise at the rollout time and from the training targets. For example, when batch normalization is involved, we use the moving average statistics when computing actions during rollouts and the targets during training. In addition, they also propose to use a mixture policy gradients

$$
\begin{align}
\mathcal G^{SNI}(\pi^r,\pi,V)=\lambda \mathcal G(\bar\pi^r,\bar\pi,\bar V)+(1-\lambda)\mathcal G(\bar\pi^r,\pi,\bar V)
\end{align}
$$

where $$\pi^r$$ is the rollout policy(as they use AC algorithms in their experiments), $$\pi$$ is the network to update, $$V$$ is the value network, and $$\mathcal G$$ denotes the gradient function. The bar above the symbol implies that noises are suspended when computing the quantity. Therefore, the first term is the gradient function without noise injected and the second is with noise injected in the training policy network $$\pi$$.

$$\mathcal G^{SNI}$$ consists of two terms interpolated by $$\lambda\in[0,1]$$. The first computes the gradients w.r.t. the denoised network $$\bar \pi$$, which is used to reduce the variance. This term is especially important early on in training when the network has not yet learned to compensate for the injected noise. Experiments also justify the effectiveness of such interpolation -- they found $$\lambda=.5$$ outperform $$\lambda=1$$ and $$\lambda=0$$ in most cases.

### Information Bottleneck Actor Critic(IBAC)

IBAC applies information bottleneck to the AC network, which minimizes $$\mathcal I(o;z)$$ and maximizes $$\mathcal I(z;a)$$, where $$z=f_\theta(o,\epsilon)$$ is the output of the encoder parameterized by $$\theta$$. The architecture thus becomes similar to a $$\beta$$-VAE. As now the encoder $$p(z\vert o)$$ is already regularized, they only apply the policy entropy term to the action heads. The final loss becomes

$$
\begin{align}
\mathcal L=\mathcal L_{AC}-\lambda \mathcal H(\pi(\cdot|z))+\beta\mathcal L_{KL}
\end{align}
$$


where $$\mathcal L_{AC}$$ is the loss function of the AC algorithm, $$\mathcal H(\pi(\cdot\vert z))$$ is the entropy loss, and $$\mathcal L_{KL}=D_{KL}(p_\theta(z\vert o)\Vert q(z))$$. 

## DrAC

[Raileanu et al. 2020](#ref2) experiments a collection of data augmentation techniques in RL. Similar work has been down by [Laskin et al. 2020](#ref3) before, which directly applied data augmentation to the PPO objective. This could be problematic as it changes $$\pi(a\vert s)$$ to $$\pi(a\vert f(s))$$, where $$f$$ applies data augmentation to $$s$$. Instead, Raileanu et al. leave the PPO objective as it is and add two additional loss terms to regularize the policy and value functions:

$$
\begin{align}
\mathcal J&=\mathcal J_{PPO} - \alpha(\mathcal L_\pi+\mathcal L_V)\\\
\mathcal L_\pi&=D_{KL}[\pi_\theta(a|s)\Vert \pi(a|f(s))]\\\
\mathcal L_V&={1\over 2}(V(f(s)) - V(s))^2
\end{align}
$$

As a result, they call their algorithm data-regularized actor-critic method, or DrAC.

## mixreg

[Wang et al. 2020](#ref4) propose generating augmented observations by linearly interpolating two observations

$$
\begin{align}
\tilde s=\lambda s_i+(1-\lambda)s_j
\end{align}
$$

Where $$\lambda\sim Beta(\alpha,\alpha)$$ with $$\alpha=0.2$$ in their experiments. 

Because the new observation becomes a convex combination of two random observations, they also mix training signals accordingly. For policy gradient method, the objective for augmented observations becomes

$$
\begin{align}
\mathcal J=\mathbb E\left[\log\pi_\theta(\tilde a|\tilde s)\tilde A\right]
\end{align}
$$

where $$\tilde A=\lambda A_i+(1-\lambda)A_j$$, and $$\tilde a$$ is $$a_i$$ if $$\lambda\ge 0.5$$ or $$a_j$$ otherwise.

For Q-learning, the objective for augmented observations becomes

$$
\begin{align}
\mathcal L=\mathbb E\left[\left(r+\gamma\max_{a'}Q(\tilde s,a')-Q(\tilde s,\tilde a)\right)^2\right]
\end{align}
$$

Where $$r=\lambda r_i+(1-\lambda)r_j$$, $$Q(\tilde s',a')=\lambda Q(s_i',a_i')+(1-\lambda)Q(s_j',a_j')$$,  and $$\tilde a$$ is $$a_i$$ if $$\lambda\ge 0.5$$ or $$a_j$$ otherwise.

It is quite astonishing that, during the test time, mixreg performs better than regular regularization techniques such as data augmentation, l2 regularization, and batch normalization. Although the authors demonstrates that mixing training signals is important to mixreg, it is still unclear why this method works. One possible explanation is that mixreg imposes piece-wise linearity regularization to the learned policy and value functions w.r.t. the states. Such regularization encourages the agent to learn a smoother policy with better generalization performance.

## References

<a name='ref1'></a>Igl, Maximilian, Kamil Ciosek, Yingzhen Li, Sebastian Tschiatschek, Cheng Zhang, Sam Devlin, and Katja Hofmann. 2019. “Generalization in Reinforcement Learning with Selective Noise Injection and Information Bottleneck,” no. NeurIPS. http://arxiv.org/abs/1910.12911.

<a name='ref2'></a>Raileanu, Roberta, Max Goldstein, Denis Yarats, and Rob Fergus. n.d. “Automatic Data Augmentation for Generalization in Deep Reinforcement Learning.”

<a name='ref3'></a>Laskin, Michael, Kimin Lee, Adam Stooke, Lerrel Pinto, Pieter Abbeel, and Aravind Srinivas. 2020. “Reinforcement Learning with Augmented Data.” http://arxiv.org/abs/2004.14990.

<a name='ref4'></a>Wang, Kaixin, Bingyi Kang, Jie Shao, and Jiashi Feng. 2020. “Improving Generalization in Reinforcement Learning with Mixture Regularization,” no. NeurIPS: 1–21.