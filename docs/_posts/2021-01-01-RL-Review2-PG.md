---
title: "RL Reviews 2 — Policy-Gradient RL"
excerpt: "The second post in a series of reviews of algorithms in reinforcement learning"
categories:
  - Reinforcement Learning
tags:
  - RL Review
---

## Introduction

In this post, we will review policy-gradient reinforcement learning algorithms. 

**Caveat**: For the sake of simplicity, this series is made to be as concise as possible and only serve as the purpose of review. Links to the post with detailed discussion are provided for the algorithms that have been studied in this blog. 

## <a name="dir"></a>Table of Contents

- [Stochastic Policy Gradient](#spg)
    - [REINFORCE](#reinforce)
    - [TRPO](#trpo)
    - [PPO](#ppo)
    - [PPG](#ppg)
    - [GAE](#gae)
    - [NAE](#nae)
    - [P3O](#p3o)
    - [Reactor](#reactor)
    - [IMPALA](#impala)
    - [STAC](#stac)
    - [Decisions in PG](#decisions-pg)

## <a name="spg"></a>Stochastic Policy Gradient

### <a name="reinforce"></a>[REINFORCE]({{ site.baseurl }}{% post_url 2018-10-07-PG %})

Policy objective:

$$
\begin{align}
\max_\theta J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[r(\tau)]
\end{align}
$$

its gradients:

$$
\begin{align}
\nabla_\theta J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[{\nabla\pi_\theta\over \pi_\theta}r(\tau)\right]
\end{align}
$$

REINFORCE uses the surrogate objective

$$
\begin{align}
\max_\theta J(\theta)=\mathbb E_{s,a\sim\pi_\theta}\left[A(s,a){\log\pi_\theta}(a|s)\right]
\end{align}
$$


[Elevator back to directory](#dir)

### <a name="trpo"></a>[Trust Region Policy Optimization]({{ site.baseurl }}{% post_url 2018-11-27-PPO %})

Trust Region Policy Optimization(TRPO) defines the constrained surrogate objective as

$$
\begin{align}
\max_\theta J_{\theta_{old}}(\theta)=\mathbb E_{s,a\sim\pi_{\theta_{old}}}\left[A(s,a){\pi_\theta(a|s)\over \pi_{\theta_{old}}(a|s)}\right]\\\
s.t.\quad \mathbb E_{s\sim\pi_{\theta_{old}}}[D_{KL}(\pi_{\theta_{old}}(\cdot|s)\Vert\pi_\theta(\cdot|s))]\le\delta
\end{align}
$$

TRPO solves the above objective following two steps:

1. Make the linear approximation to the objective and quadratic approximation to the constraints, then solve its Lagrangian using the [conjugate gradient method]({{ site.baseurl }}{% post_url 2018-11-21-CG %})
2. Approximate the maximal step size $$\alpha$$ by solving the KL constraint, then do line search to find $$\alpha$$ that improves objective

[Elevator back to directory](#dir)

### <a name="ppo"></a>[Proximal Policy Optimization]({{ site.baseurl }}{% post_url 2018-11-27-PPO %})

Proximal Policy Optimization(PPO) defines the surrogate objective as

$$
\begin{align}
\max_\theta J(\theta)=\mathbb E_{a,s\sim\pi_{old}}\left[\min(r(\theta)A(s,a), \mathrm{clip}(r(\theta),1-\epsilon,1+\epsilon)A(s,a))+c\mathcal H(\pi_\theta)\right]
\end{align}
$$


where clip prevents the potential negative effect from data way off the current policy.

[Elevator back to directory](#dir)

### <a name="ppg"></a>[Phasic Policy Gradient]({{ site.baseurl }}{% post_url 2020-12-27-PPG %})

PPG trains PPO with a separate value network in two phase:

- In the policy phase, PPG trains the same objective as PPO but with only 1 epoch

- In the auxiliary phase, which is performed every $$N_\pi$$ policy iterations, PPG trains the value network and the convolutional encoder of the policy network with the loss below
  
$$
  \mathcal L^{joint}(\theta)=\mathbb E_t\left[{1\over 2}(V_\theta(s_t)-\hat V_t)^2\right]+\beta \mathbb E_t[D_{KL}(\pi_{\theta_{old}}(\cdot|s_t)\Vert\pi_\theta(\cdot|s_t))]+\mathbb E\left[{1\over 2}(V_\phi(s_t)-\hat V_t)^2\right]
  $$

  where $$V_\theta$$ is the value function that shares the same convolutional network as the policy $$\pi_\theta$$, $$V_\phi$$ is the standalone value network. The KL term ensures that auxiliary training does not corrupt the policy.

[Cobbe et al. 2020](http://arxiv.org/abs/2009.04416) provides extensive study on the design choices. They find

- Training the policy with a single epoch yields best performance for well-tuned hyperparameters
- More auxiliary training(more auxiliary epochs) is beneficial. However, performance decreases when we perform auxiliary training too frequently.
- A separate value network is not necessary, one can obtain similar performance by training a detached value head in the policy phase

[Elevator back to directory](#dir)

### <a name="gae"></a>[Generalized Advantage Estimation]({{ site.baseurl }}{% post_url 2018-12-01-GAE %})

Generalized Advantage Estimation(GAE) provides a way to control the bias-and-variance trade-off for on-policy learning. It defines the advantages as

$$
\begin{align}
A(s_t,a_t)=\sum_{i=0}^\infty(\gamma\lambda)^i\delta_{t+i}
\end{align}
$$

where $$\gamma$$ controls the scale of the return and value function, and $$\lambda$$ governs the weight of the n-step advantage. $$\gamma$$ introduces bias whenever it’s less than 1, while $$\lambda$$ introduces bias only when the value function is inaccurate. GAE is analogous to $$TD(\lambda)$$ in terms of the advantage.

In practice, we will further normalize the advantage to have zero mean and standard deviation to reduce the variance when using it in the policy objective. The target values for the critic is defined as

$$
\begin{align}
V(s_t)=V_{old}(s_t)+A(s_t,a_t)
\end{align}
$$


[Elevator back to directory](#dir)

### <a name="nae"></a>Normalized Advantage Estimation

Normalized Advantage Estimation(NAE) normalizes the value and advantage function to reduce the variance. The normalization function used in practice is generally defined as

$$
\begin{align}
norm(x)={x-\mu(x)\over \sigma(x)}
\end{align}
$$

The advantages and the target values are then defined as follows

1. Normalize old values $$V_{old}$$ to have the same mean and standard deviation as the returns $$G$$
   
$$
   V_{old}=norm(V_{old})*\sigma(G)+\mu(G)
   $$


2. Use the normalized advantages in the policy objective
   
$$
   A=norm(G-V_{old})
   $$


3. The target values are the normalized returns
   
$$
   G=norm(G)
   $$


[Elevator back to directory](#dir)

### <a name="p3o"></a>[Policy-on Policy-off Policy Optimization]({{ site.baseurl }}{% post_url 2020-10-07-P3O %})

Policy-on Policy-off Policy Optimization(P3O) uses Effective Sampling Size (ESS) to measure the validity of the off-policy data to the current policy. Specifically, P3O defines the following normalized ESS

$$
\begin{align}
ESS={1\over N}{(\sum_xw(x))^2\over\sum_xw(x)^2}
\end{align}
$$

where $$w(x)$$ is the importance ratio of sample $$x$$.

With the definition of ESS, P3O define the objective function as follows

$$
\begin{align}
\mathcal J(\pi)=&\mathbb E_{_\pi}[A(s,a)+\alpha\mathcal H(\pi)]+\mathbb E_{\mu}[\min(\rho,c)A(s,a)]-\lambda\mathbb E_\mu[D_{KL}(\mu\Vert \pi)]\\\
where\quad c=&ESS\\\
\lambda=&1-ESS
\end{align}
$$

where the first expected term is the general policy gradient objective. The second term is a policy gradient objective conditioned on off-policy data. The third term penalizes the deviation of the current policy $$\pi$$ from the behavior policy $$\mu$$. $$c=ESS$$ truncates the importance ratio based on the efficacy of the data. However, the efficacy of this choice is not validated in practice. $$\lambda=1-ESS$$ weights the KL term based on the extent to which $$\mu$$ is from $$\pi$$. The KL term is strong when $$\mu$$ deviates far from $$\pi$$.

[Elevator back to directory](#dir)

### <a name="reactor"></a>[Reactor]({{ site.baseurl }}{% post_url 2020-10-01-REACTOR %})

$$\beta$$-LOO policy gradient estimate interpolates Reinforce and $$Q$$-learning-based policy gradient

$$
\begin{align}
\mathcal G_{\beta-LOO}=\beta(R(s,\hat a)-Q(s,\hat a))\nabla\pi(\hat a|s)+\sum_aQ(s,a)\nabla\pi(a|s)
\end{align}
$$

where $$\beta=\min(c,{1\over \mu(a\vert s)})$$. $$\mathbb E_{\mu}\mathcal G_{\beta-LOO}$$ is unbiased when $$c\rightarrow\infty$$ and $$R(s,\hat a)$$ is an accurate estimate of the return, regardless of the accuracy of $$Q(s,\hat a)$$. In practice, $$R(s,\hat a)$$ is computed by the Retrace algorithm.

[Elevator back to directory](#dir)

### <a name="impala"></a>[Importance Weighted Actor-Learner Architecture]({{ site.baseurl }}{% post_url 2019-11-14-IMPALA %})

IMPALA trains with nearly on-policy data using the following losses

$$
\begin{align}
\mathcal L(V_\psi)=&\mathbb E_\mu[(v(x_t)-V_\psi(x_t))^2]\\\
\mathcal L(\pi_\theta)=&-\mathbb E_{(x_t,a_t,x_{t+1})\sim\mu}[\rho_t(r_t+\gamma v(x_{t+1})-V_\psi(x_t))\log\pi_\theta(a_t|x_t)]-\mathcal H(\pi_\theta)\\\
where\quad v(x_t) :=& V(x_t)+\sum_{k=t}^{t+n-1}\gamma^{k-t}\left(\prod_{i=t}^{k-1}c_i\right)\delta_kV\tag{1}\label{eq:1}\\\
\delta_kV:=&\rho_k(r_k+\gamma V(x_{k+1})-V(x_k))\\\
c_{i}:=&\lambda \min\left(\bar c, {\pi(a_i|x_i)\over \mu(a_i|x_i)}\right)\\\
\rho_k:=&\min\left(\bar\rho, {\pi(a_k|x_k)\over \mu(a_k|x_k)}\right)
\end{align}
$$

where $$\bar c$$ impacts the contraction rate while $$\bar\rho$$ impacts the fixed point of $$V_\psi$$ and $$\pi_\theta$$. 

### <a name="stac"></a>[Self-Tuning Actor Critic]({{ site.baseurl }}{% post_url 2020-11-27-STAC %})

STAC employs meta-gradient to tune the set of tunable hyperparameters in IMPALA. Besides that it makes two additional changes. First, observing that truncated importance ratio reduces the effect of later TD errors, it uses leaky V-trace that interpolates between the truncated and the canonical importance samplings

$$
\begin{align}
c_{i}:=&\lambda \big(\alpha_c\min(\bar c, \text{IS}_t)+(1-\alpha_c)\text{IS}_t\big)\\\
\rho_k:=&\alpha_\rho\min(\bar\rho, \text{IS}_t)+(1-\alpha_\rho)\text{IS}_t\\\
where\quad \text{IS}_t=&{\pi(a_i|x_i)\over \mu(a_i|x_i)}
\end{align}
$$

Second, it introduces auxiliary tasks that trains additional policy and value heads with different hyperparameters.

### <a name="decisions_pg"></a>Decisions in Policy Gradients

[Engstrom et al. 2019](https://arxiv.org/abs/2005.12729) study the implementation details of PPO from OpenAI's baseline, showing that

- Learning rate annealing, reward normalization significantly improves the performance

[Andrychowicz et al. 2020]({{ site.baseurl }}{% post_url 2020-09-27-On-Policy-Choices %}) conduct an extensive hyperparameter search on 5 continuous control environments from OpenAI's Gym. They find

- PPO loss performs best when performing multiple passes on the collected data. V-trace policy loss does not work well with this paradigm
- For the control environments tested, $$tanh$$ performs significantly better than $$ReLU$$. Initializing the policy head to small value is important.
- Observation normalization is crucial but clipping afterwards is unnecessary. 
- GAE and V-trace yield similar performance when using them to compute the advantage and value target. 
- It is helpful to recompute the advantage and shuffle transitions for every pass over the data.
- The number of transitions gathered in each iteration influences the performance significantly; Too many transitions impairs the performance. 

[Liu et al. 2021]({{ site.baseurl }}{% post_url 2021-02-07-Net-regularization-in-PG %}) study the effect of network regularization in policy optimization in continuous control environments, finding that

- Adding regularization generally improves the performance
- $$L_2$$ is more effective than entropy regularization in most cases. 
- Regularizations is especially important for hard tasks.