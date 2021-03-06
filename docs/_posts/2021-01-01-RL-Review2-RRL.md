---
title: "RL Reviews 2 — Model-Free RL"
excerpt: "The second post in a series of reviews of algorithms in reinforcement learning"
categories:
  - Reinforcement Learning
tags:
  - RL Review
---

## Introduction

In this post, we will review KL-regularized reinforcement learning algorithms. 

**Caveat**: For the sake of simplicity, this series is made to be as concise as possible and only serve as the purpose of review. Links to the post with detailed discussion are provided for the algorithms that have been studied in this blog. 

## <a name='dir'></a>Table of Contents

- [Basics](#basics)
- [Maximum Entropy RL](#maxent)
  - [SVI](#svi)
  - [SAC](#sac)
  - [PCL](#pcl)
  - [Trust PCL](#tpcl)
- [KL-Regularzed RL](#kl)
  - [MIRL](#mirl)
  - [M-RL](#mrl)
  - [MPO](#mpo)

## <a name="basics"></a>Basics

All KL-regularized RL algorithms are essentially built upon the following objective

$$
\begin{align}
\max \sum_{t=1}^T\gamma^{t-1}\mathbb E_{s_t,a_t\sim\pi}[r(s_t,a_t)-\alpha D_{KL}(\pi(\cdot|s_t)\Vert\pi_{0}(\cdot|s_t))]\tag{1}\label{eq:1}
\end{align}
$$

and the associated update rules

$$
\begin{align}
Q(s_t,a_t)&=r(s_t,a_t)+\gamma V(s_{t+1})\tag{2}\label{eq:2}\\\
V(s_{t})&=Q(s_t,a_t)-\alpha D_{KL}(\pi(\cdot|s_t)\Vert\pi_0(\cdot|s_t))\tag{3}\label{eq:3}\\\
&=\alpha\log\int\exp\pi_0(a_t|s_t)\left({1\over\alpha}Q(s_t,a_t)\right)da_t\tag{4}\label{eq:4}\\\
\pi(a_t|s_t)&=\pi_0(a_t|s_t)\exp\left({1\over\alpha}\big(Q(s_t,a_t)-V(s_t)\big)\right)\tag{5}\label{eq:5}
\end{align}
$$


Maximum entropy RL(MaxEnt RL) assumes a uniform prior and therefore reduces the KL term to an entropy term. In MaxEnt RL, all $$\pi_0$$s in the above update rules are omitted.

## <a name="maxent"></a>Maximum Entropy RL

### <a name='svi'></a>[Soft Value Iteration]({{ site.baseurl }}{% post_url 2019-01-21-SVI %})

SVI is an dynamic programming algorithm built upon the MaxEnt RL update rules

$$
\begin{align}
&V(s_{T+1})=0\\\
&\mathbf {for}\ t=T\ \mathbf{to}\ 1:\\\
&\quad Q(s_t,a_t)=r(s_t,a_t)+\mathbb E_{p(s_{t+1}|s_t,a_t)}[V(s_{t+1})]\\\
&\quad V(s_t)=\alpha\log \sum_{a_t}\exp\left({1\over\alpha}Q(s_t,a_t)\right)
\end{align}
$$

with the softmax policy

$$
\begin{align}
\pi(a_t|s_t)=\exp\left({1\over\alpha}(Q(s_t,a_t)-V(s_t))\right)
\end{align}
$$


[Elevator back to directory](#dir)

### <a name="sac"></a>[Soft Actor-Critic]({{ site.baseurl }}{% post_url 2019-01-27-SAC %})

SAC derives the following losses from the MaxEnt RL update rules


$$
\begin{align}
\mathcal L(Q)=&\mathbb E_{s_t,a_t\sim\mathcal D}\left[\left(r(s_t,a_t)+\gamma V^-(s_{t+1})-Q(s_t,a_t)\right)^2\right]\\\
\mathcal L(V)=&\mathbb E_{s_t\sim\mathcal D}\left[\Big(\mathbb E_{a_t\sim\pi(a_t|s_t)}[Q(s_t,a_t)-\alpha\log\pi(a_t|s_t)]-V(s_t)\Big)^2\right]\\\
\mathcal L(\pi)=&\mathbb E_{s_t\sim\mathcal D}\left[D_{KL}\bigg(\pi(\cdot|s_t)\Big\Vert\exp\Big({1\over\alpha}\big(Q(s_t,\cdot)-V(s_t)\big)\Big)\bigg)\right]\\\
\propto&\mathbb E_{s_t\sim\mathcal D,\epsilon\sim\mathcal N}\left[\alpha\log\pi(f(\epsilon;s_t)|s_t)-Q(s_t,f(\epsilon;s_t))\right]
\end{align}
$$

where the value network is not necessary and can be replaced by $$Q(s_t,a_t)-\alpha\log\pi(a_t\vert s_t)$$.

There are three additional tricks used in practice for continuous control environments:

1. We clip the logarithm of the standard deviation of the policy to be in range $$[-20, 2]$$

2. To get bounded actions, we apply $$\tanh$$ to the sampled actions, which gives us the log-likelihoood as
   
$$
   \log \pi(a|s)=\log\mu(a|s)+\sum_{i=1}^D(\log 4+2u_i-2\mathrm{softmax}(2u_i))
   $$


3. We can define the MaxEnt RL objective as an constraint optimization problem
   
$$
   \begin{align}
   \max_{\pi}\sum_{t=0}^T\mathbb E_{\pi}\left[r(s_t,a_t)\right]\\\
   s.t.\mathbb E_{\pi}[-\log\pi(a|s)]\ge\mathcal H
   \end{align}
   $$

   Solving the associated generalized Lagrangian, we obtain
   
$$
   \alpha^*=\arg\min_\alpha\alpha(\mathcal H_\pi-\mathcal H)
   $$

   where $$\mathcal H_\pi=-\mathbb E_\pi[\log\pi(a\vert s)]$$ is the policy entropy at state $$s$$ and $$\mathcal H$$ is the desired target entropy.

[Elevator back to directory](#dir)

### <a name='pcl'></a>[Path Consistency Learning]({{ site.baseurl }}{% post_url 2019-02-07-PCL %})

PCL optimizes a value network and a policy network by minimizing the following loss

$$
\begin{align}
\mathcal L(V,\pi)=\sum_{s_{t:t+d}}{1\over 2}C(s_{t:t+d},V,\pi)^2\\\
where\quad C(s_{t:t+d},V,\pi)=V(s_{t+d})-\gamma^dV(s_t)+\sum_{i=0}^{d-1}\gamma^i\big(r(s_{t+i},a_{t+i})-\alpha\log\pi(a_{t+i}|s_{t+i})\big)
\end{align}
$$

where $$C$$ is obtained by plugging $$Q$$ into the softmax policy and adding consecutive time steps.

[Elevator back to directory](#dir)

### <a name="tpcl"></a>[Trust Path Consistency Learning]({{ site.baseurl }}{% post_url 2019-02-07-PCL %})

PCL imposes small policy update by introducing a KL constraint to the policy, from which we obtain the generalized Lagrangian

$$
\begin{align}
&\max \sum_{t=1}^T\gamma^{t-1}\mathbb E_{s_t,a_t\sim\pi}[r(s_t,a_t)+\alpha\mathcal H(\pi(a_t|s_t))]-\lambda D_{KL}(\pi(\cdot|s_t)\Vert\pi_{\tilde\theta}(\cdot|s_t))\\\
=&\max\sum_{t=1}^T\gamma^{t-1}\mathbb E_{s_t,a_t\sim\pi}[\tilde r(s_t,a_t)+(\alpha-\lambda)\mathcal H(\pi(a_t|s_t))]\\\
where&\quad \tilde r(s_t,a_t)=r(s_t,a_t)+\lambda\log\pi_{\tilde\theta}(a_t|s_t)
\end{align}
$$

where $$\pi_{\tilde\theta}$$ is a lagged policy, which is updated by $$\tilde \theta=(1-\tau)\tilde\theta+\tau\theta)$$, in which $$\theta$$ is the parameter of the current policy $$\pi$$. This gives us a similar loss as PCL with the transformed reward function

$$
\begin{align}
\mathcal L(V,\pi)=\sum_{s_{t:t+d}}{1\over 2}C(s_{t:t+d},V,\pi)^2\\\
where\quad C(s_{t:t+d},V,\pi)=V(s_{t+d})-\gamma^dV(s_t)+\sum_{i=0}^{d-1}\gamma^i\big(\tilde r(s_{t+i},a_{t+i})-(\alpha+\lambda)\log\pi(a_{t+i}|s_{t+i})\big)
\end{align}
$$


[Elevator back to directory](#dir)

### <a name="mrl"></a>[Munchausen Reinforcement Learning]({{ site.baseurl }}{% post_url 2020-11-01-M-RL %})

M-DQN modifies DQN by first augmenting with a soft target and then adding a scaled log-policy to the immediate reward. We can write the target of M-DQN as follows

$$
\begin{align}
\hat q_{\text{m-dqn}}(s,a)=&r(s,a)\color{red}{+\nu\alpha\log\pi(a|s)}+\gamma\sum_{a'}\pi(a'|s')(q(s',a')\color{blue}{-\alpha\log\pi(a'|s')})\\\
\pi(a|s)=&\text{softmax}({1\over\alpha}Q(s,a))
\end{align}
$$

M-DQN connects to several previous methods

- M-DQN can be written in value iteration form, which gives Munchausen Value Iteration($$\text{M-VI}(\alpha, \tau)$$)
  
$$
  \text{M-VI}(\alpha, \tau)=\begin{cases}
  \pi_{k+1}&=\arg\max_\pi\left<\pi, q_k\right>\color{blue}{+\tau\mathcal H(\pi)}\\\
  q_{k+1}&=r\color{red}{+\alpha\tau\log\pi_{k+1}}+\gamma P\left<\pi_{k+1},q_{k}\color{blue}{-\tau\log\pi_{k+1}}\right>+\epsilon_{k+1}
  \end{cases}
  $$


- M-DQN is also equivalent to Mirror Descent Value Iteration(MD-VI). Specifically we have $$\text{M-VI}(\alpha, \tau)=\text{MD-VI}(\alpha\tau,(1-\alpha)\tau)$$:
  
$$
  \text{MD-VI}(\alpha\tau,(1-\alpha)\tau)=\begin{cases}
  \pi_{k+1}&=\arg\max_\pi\left<\pi,q'_k\right>-\alpha\tau D_{KL}(\pi\Vert\pi_k)+(1-\alpha)\tau\mathcal H(\pi)\\\
  q'_{k+1}&=r+\gamma P\big(\left<\pi_{k+1},q'_k\right> -\alpha\tau D_{KL}[\pi_{k+1}\Vert\pi_k] +(1-\alpha)\tau\mathcal H(\pi_{k+1})\big)+\epsilon_{k+1}
  \end{cases}
  $$

  This shows that M-DQN performs KL regularization between successive policies

- When $$\tau\rightarrow 0$$, M-DQN is reduced to Advantage Learning($$AL$$)
  
$$
  q_{k+1}=r+\alpha(q_k-\langle\pi_{k+1},q_k\rangle)+\gamma P\langle\pi_{k+1},q_k\rangle+\epsilon_{k+1}\\\
  \pi_{k+1}=\arg\max_a q_k
  $$

  AL adds a small penalty to rewards when the selected action is suboptimal, i.e., $$a\ne \arg\max_a q_k$$. This can increase the action-gap and mitigate the undesirable effects of approximation and estimation errors made on $$q$$ on the induced greedy policies.

[Elevator back to directory](#dir)

## <a name="KL"></a>KL-Regularized RL

### <a name="mirl"></a>[Mutual Information Reinforcement Learning]({{ site.baseurl }}{% post_url 2019-08-14-MIRL %})

MIRL regularizes the policy against a state-independent action prior, which is initialized by minimizing $$\mathbb E_sD_{KL}(\pi(a\vert s)\Vert\pi_0(a))$$ and updated by an exponential moving average $$\pi_0(a)=(1-\alpha_{\pi_0})\pi_0(a)+\alpha_{\pi_0} \pi(a\vert s)$$, where $$\alpha_{\pi_0}$$ is the step size. For discrete action setting, MIRL defines the target $$Q$$ as

$$
\begin{align}
\mathcal L(\theta)=&\mathbb E_{s,a,r,s'\sim replay}\left[\left(q(s,a)-Q(s,a)\right)^2\right]\\\
q(s,a)=&r(s,a)+\gamma \alpha\log\sum_{a'}\exp\pi_0(a'|s')\left({1\over\alpha}Q(s',a')\right)
\end{align}
$$

where the inverse of $$\alpha$$, $$\beta={1\over\alpha}$$ is updated according to the inverse of the empirical loss of the $$Q$$-function

$$
\begin{align}
\beta=(1-\alpha_\beta)\beta+\alpha_{\beta}\left({1\over \mathcal L(\theta)}\right)
\end{align}
$$

[Elevator back to directory](#dir)

### <a name="mpo"></a>[Maximum a posterior Policy Optimization]({{ site.baseurl }}{% post_url 2020-09-14-MPO %})

MPO performs KL-regularized RL following the Expectation Maximization(EM) framework. In the E-step, it minimizes the $$Q$$-function using the Retrace algorithm. In the M-step, MPO computes a sample-based policy($$\pi$$ in Equation $$\eqref{eq:1}$$), against which it learns a parametric policy($$\pi_0$$ in Equation $$\eqref{eq:1}$$) via supervised learning. Several constraints are imposed to mitigate overfitting:

- When learning $$\pi_0$$, we add a KL constraint to avoid $$\pi_0$$ deviate from the previous policy too much
- We replace the KL regularization in Equation $$\eqref{eq:1}$$ with a KL constraint, which demonstrates a way to learn the temperature $$\alpha$$. We can then use the learned $$\alpha$$ to compute the sample-based policy $$\pi$$.
- To avoid premature convergence of Gaussian policies, MPO optimizes the mean and covariance separately.

### <a name="bp"></a>[Behavior Priors]({{ site.baseurl }}{% post_url 2020-10-27-behavior-prior %})

Information asymmetry is important for a good behavior prior. Excessive information may hinder the agents' ability to learn in a new task, while too little information results in meaningless priors.

When it comes to model multi-modal distributions, we may need latent variable models to increase the capacity to model complex distribution. This changes Equation $$\eqref{eq:1}$$ to

$$
\begin{align}
\max \sum_{t=1}^T\gamma^{t-1}\mathbb E_{s_t,a_t\sim\pi}[r(s_t,a_t)-\alpha \big(D_{KL}(\pi(z|x_t)\Vert\pi_{0}(z|x_t))+D_{KL}(\pi(a|x_t,z_t)\Vert\pi_0(a|x_t,z_t))\big)]
\end{align}
$$

[Elevator back to directory](#dir)

