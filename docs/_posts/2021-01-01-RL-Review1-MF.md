---
title: "RL Reviews 1 — Model-Free RL"
excerpt: "The first post in a series of reviews of algorithms in reinforcement learning"
categories:
  - Reinforcement Learning
tags:
  - RL Review
---

## Introduction

In this post, we will review model-free reinforcement learning algorithms. 

**Caveat**: For the sake of simplicity, this series is made to be as concise as possible and only serve as the purpose of review. Links to the post with detailed discussion are provided for the algorithms that have been studied in this blog. 

## <a name='dir'></a>Table of Contents

- [Deep Q-Learning](#dql)
    - [DQN](#dqn)
    - [Distributional Q-Learning](#dist)
        - [c51](#c51)
        - [QR-DQN](#qrdqn)
        - [IQN](#iqn)
        - [FQF](#fqf)
    - [Rainbow](#rainbow)
        - [DDQN](#ddqn)
        - [Dueling DQN](#duel)
        - [Noisy Networks](#noisy)
        - [PER](#per)
    - [Retrace](#retrace)
    - [Decisions in DQN](#decisions_dqn)
- [Deterministic Policy Gradient](#dpg)
    - [DDPG](#ddpg)
    - [TD3](#td3)
    - [D4PG](#d4pg)
- [Stochastic Policy Gradient](#pg)
    - [REINFORCE](#reinforce)
    - [TRPO](#trpo)
    - [PPO](#ppo)
    - [PPG](#ppg)
    - [GAE](#gae)
    - [NAE](#nae)
    - [P3O](#p3o)
    - [IMPALA](#impala)

## <a name="dql"></a>Deep Q-Learning

### <a name="dqn"></a>[Deep Q-Network]({{ site.baseurl }}{% post_url 2018-09-27-DQN %})

Deep Q-Network(DQN) adopts three auxiliary techniques to stablize training with deep neural networks:

1. Target network
2. Experience Replay
3. Huber loss and gradient clipping

[Elevator back to directory](#dir)

### <a name="dist"></a>Distributional Q-Learning

#### <a name="c51"></a>[c51]({{ site.baseurl }}{% post_url 2018-10-22-c51 %})

C51 models a value distribution $$Z(s,a)$$ whose expectation is the action-value, i.e., $$Q(s,a)=\mathbb E[Z(s,a)]$$. Specifically, the value distribution is modeled by a discrete distribution parameterized by $$N\in\mathbb N$$ and $$V_{\min},V_{\max}\in \mathbb R$$, and whose support is the set of atoms

$$
\begin{align}
\left\{z_i=V_{\min}+i\Delta z:0\le i<N\right\}\\\
\Delta z:={V_{\max}-V_{\min}\over N-1}
\end{align}
$$

The corresponding atom probabilities are given by a parameterized model $$\theta=S\times A\rightarrow\mathbb R^N$$

$$
\begin{align}
Z_\theta(s,a)=z_i\\\
\mathrm{w.p.}\quad p_i(s,a):=\mathrm{softmax}(\theta_i(s,a))={\exp({\theta_i(s,a)})\over\sum_j\exp(\theta_j(s,a))}
\end{align}
$$

The target value distribution is defined as the projected Bellman update

$$
\begin{align}
\left(\Phi \mathcal{\hat T}Z_\theta(s, a)\right)_i=\sum_{j=0}^{N-1}\left[1-{\left|\left[\mathcal{\hat T}z_j\right]_{V_{\min}}^{V_{\max}}-z_i\right|\over\Delta z} \right]_0^1p_j(s', \pi(s'))
\end{align}
$$

and we optimize the value distribution by minimizing the KL divergence

$$
\begin{align}
D_{KL}\left(\Phi\mathcal{\hat T}Z_{\tilde \theta}(s, a)\Vert Z_\theta(s,a)\right)
\end{align}
$$


[Elevator back to directory](#dir)

#### <a name="qrdqn"></a>[Quantile Regression Deep Q Network]({{ site.baseurl }}{% post_url 2019-03-27-IQN %})

Quantile Regression Deep Q Network(QR-DQN) optimizes the quantile functions at points onto which the projection of the target distribution has the minimal Wasserstein distance. For $$N$$ equally spaced intervals $$\{[\tau_i,\tau_{i+1}]_{i=0}^{N-1}\}$$, it is easy to show that these points are the midpoints

$$
\begin{align}
\{\tau_i={\tau_i+\tau_{i+1}\over 2};0\le i< N\}\\\
\end{align}
$$

The corresponding quantile functions $$\theta$$ are computed by minimizing the quantile Huber loss

$$
\begin{align}
\mathcal L(\theta)&=\sum_{i=0}^{N-1}{1\over N}\sum_{j=0}^{N-1}\rho_{\tau_i}^\kappa(\delta_{ij}(s,a))\\\
where\quad
\rho_{\tau_i}^\kappa(u)&=\mathcal L_\kappa(u)|\tau_i-\mathbf 1_{u<0}|\\\
\quad \mathcal L_\kappa(u)&=\begin{cases}{1\over 2}u^2,&\mathrm{if\ }|u|\le\kappa\\\
\kappa(|u|-{1\over 2}\kappa),&\mathrm{otherwise}\end{cases}\\\
\delta_{ij}(s,a)&=r(s,a)+\gamma\theta_j^-(s',a^*)-\theta_i(s,a)\\\
a^*&=\underset{a'}{\arg\max}Q(s',a')\\\
Q(s,a)&=\mathbb E[\theta(s,a)]={1\over N}\sum_{i=0}^{N-1}\theta_i(s,a)\\\
\end{align}
$$


[Elevator back to directory](#dir)

#### <a name="iqn"></a>[Implicit Quantile Networks]({{ site.baseurl }}{% post_url 2019-03-27-IQN %})

Implicit Quantile Networks learns the whole quantile functions, and takes an additional input to query the quantile value at specified locations.

<figure>
  <img src="{{ '/images/distributional/iqn.png' | absolute_url }}" alt="" width="1000">
  <figcaption></figcaption>
  <style>
    figure figcaption {
    text-align: center;
    }
  </style>
</figure>

The loss is similar to that of QR-DQN, but the policy is a little different

$$
\begin{align}
\pi_\beta(s) = \underset{a\in\mathcal A}{\arg\max}{1\over K}\sum_{k=1}^K\theta_{\tilde\tau_k}(s,a),\quad\tilde\tau_k\sim\beta(\tau),\ \tau\sim U([0,1])
\end{align}
$$

where $$\beta:[0,1]\rightarrow[0,1]$$ is known as a distortion risk measure, by choosing which we can establish a risk-sensitive behavior for the agent.

[Elevator back to directory](#dir)

#### <a name="fqf"></a>[Fully Parameterized Quantile Function]({{ site.baseurl }}{% post_url 2019-04-01-FQF %})

FQF introduces a fraction proposal network for proposing probabilities $$\{\tau_i\}_{i=0}^{N-1}$$. The fraction proposal network is trained jointly with the quantile function to minimize the 1-Wasserstein distance between the target and the current distributions. This gives the following gradient

$$
\begin{align}
\forall i\in(0,N),\quad{\partial W_1\over\partial\tau_i}=2F^{-1}_Z(\tau_i)-F^{-1}_Z(\hat\tau_i)-F^{-1}_Z(\hat\tau_{i+1})
\end{align}
$$

where $$F^{-1}_Z$$ is the quantile function. Because now $$\tau_i$$s no longer uniformly spread between $$0$$ and $$1$$, we change the action-value function accordingly

$$
\begin{align}
Q(s,a)=\sum_{i=0}^{N-1}(\tau_{i+1}-\tau_i)F_Z^{-1}(\hat\tau_i)
\end{align}
$$


[Elevator back to directory](#dir)

### <a name="rainbow"></a>[Rainbow]({{ site.baseurl }}{% post_url 2018-10-27-Rainbow %})

Rainbow combines 6 improvements on DQN:

1. Distributional DQN
2. Multi-step learning
3. DDQN
4. Dueling DQN
5. PER
6. Noisy Networks

#### [Double DQN]({{ site.baseurl }}{% post_url 2018-10-27-Rainbow %})

DQN suffers the overestimation bias introduced by two facts: 1) $$Q$$-function is noisy, and 2) the expected value of the maximum of a set of variables is greater than or equal to the maximum of their expected values, i.e.,

$$
\begin{align}
r(s,a)+\gamma \mathbb E\left[\max_{a'}Q^-(s',a')\right]\ge r(s,a)+\gamma\max_{a'}\mathbb E[Q^-(s',a')]
\end{align}
$$

which is obtained from Jensen's inequality and the convex property of $$\max$$.

Double DQN(DDQN) mitigates the overestimation bias of DQN by having the online network pick the best action for the next state, which gives the target value as

$$
\begin{align}
r(s,a)+\gamma Q^-\left(s',\underset{a'}{\arg\max}Q(s',a')\right)
\end{align}
$$


[Elevator back to directory](#dir)

#### <a name="duel"></a>[Dueling DQN]({{ site.baseurl }}{% post_url 2018-10-27-Rainbow %})

Dueling DQN divides $$Q$$-function into value and advantage functions:

$$
\begin{align}
Q(s,a)=V(f(s))+A(f(s),a)-{\sum_{a'}A(f(s),a')\over |\mathcal A|}
\end{align}
$$

The last term is introduced since it is not sufficient, with only the first two terms, to uniquely recover $$V$$ and $$A$$ for a given $$Q$$. If we replace the last term with a maximum, then $$V$$ can be nicely interpreted as the optimal action-value $$\max_{a}Q(s,a)$$. The reason we use the average is to increase the stability of the optimization.

[Elevator back to directory](#dir)

#### <a name="noisy"></a>[Noisy Networks]({{ site.baseurl }}{% post_url 2018-10-27-Rainbow %})

Noisy networks introduces extra layers in addition to traditional linear layers to add noise to the parameters, whereby providing some explorations:

$$
\begin{align}
y&=(W+W_{noisy}\cdot\epsilon^w)x+(b+b_{noisy}\cdot\epsilon^b)\\\
&=(Wx+b)+((W_{noisy}\cdot\epsilon^w)x+b_{noisy}\cdot\epsilon^b)
\end{align}
$$


[Elevator back to directory](#dir)

#### <a name="per"></a>[Prioritized Experience Replay]({{ site.baseurl }}{% post_url 2018-10-15-PER %})

Prioritized Experience Replay(PER) samples transitions with probability $$P(i)$$ proportional to their priority $$p_i^\alpha$$, where $$\alpha$$ determines how much prioritization is used. Two variants are proposed: 

1. Proportional prioritization: a sum-tree, $$p_i=\vert \delta_i\vert +\epsilon$$
2. Rank-based prioritization: a priority queue, $$p_i={1\over\mathrm{rank}(i)}$$.

To correct bias towards transitions of high priority, we apply weighted importance sampling, which results in the loss

$$
\begin{align}
\mathcal L&={1\over \max_i w_i}w_i\delta_i^2\\\
w_i&=\left({1\over N}{1\over P(i)}\right)^\beta
\end{align}
$$

where $$1\over \max_iw_i$$ normalizes ratios to avoid scaling the update upward, $$\beta$$ in general anneals from its initial value to $$1$$ towards the end of learning.

[Elevator back to directory](#dir)

### <a name="retrace"></a> [Retrace]({{ site.baseurl }}{% post_url 2020-10-01-Retrace %})

Retrace defines the following truncated return-based target for off-policy learning

$$
\begin{align}
\mathcal RQ(x,a)=&Q(x,a)+\mathbb E_\mu\left[\sum_{k=t}^{t+n-1}\gamma^{k-t}\left(\prod_{i=t+1}^{k}c_i\right)\delta_k\right]\\\
where\quad c_{i}=&\lambda\min\left(1,{\pi(a_i|x_{i})\over\mu(a_i|x_i)}\right)\\\
\delta_k=&r_k+\gamma\mathbb E_{\pi}Q(x_{k+1},\cdot)-Q(x_{k},a_{k})
\end{align}
$$

Retrace enjoys the following advantages

1. Retrace does not suffer from variance explosion as importance ratios are truncated.
2. It does not cut in on-policy case, making it possible to benefit from the full returns
3. In the evaluation setting, Retrace is a $$\gamma$$-contraction around $$Q^{\pi}$$ for arbitrary policies $$\mu$$ and $$\pi$$.
4. In the control setting, Retrace converges almost surely to $$Q^*$$, without requiring the GLIE assumption.

[Elevator back to directory](#dir)

### <a name="decisions_dqn"></a>Decisions in DQN

[Hasselt et al. 2018]({{ site.baseurl }}{% post_url 2020-07-27-deadly-triad %}) conduct a large scale study on Atari57 and find that

1. The divergence of the $$Q$$ function often results in poor performance 
   1. The target network, double Q, and multi-step returns can effectively reduce the occurrence of divergence and improve the performance
   2.  Strong prioritization can lead to instability
2. A large network is more likely to increase divergence, but still yields better performance. One hypothesis is that when the network is large, the update of the value function may not be generalized to the next state, which helps stabilize the target value.

[Fu et al. 2019]({{ site.baseurl }}{% post_url 2019-12-01-DQN %}) find that

1. A large network not only is crucial for representing a better solution but also make it easier to train using bootstrapping.
2. It is less likely to overfit when training with data of high entropy. This stresses the importance of the experience replay in value-based algorithms

## <a name="dpg"></a> Deterministic Policy Gradient

### <a name="ddpg"></a>[Deep Deterministic Policy Gradient]({{ site.baseurl }}{% post_url 2018-10-01-PG %})

In Deep Deterministic Policy Gradient(DDPG), policy($$\mu(s)$$) deterministically selects an action that maximize the $$Q$$-value. It uses target networks and experience replay to stablize deep neural network training as DQN does.

[Elevator back to directory](#dir)

### <a name="td3"></a>Twin Delayed Deep Deterministic Policy Gradient Algorithm

Twin Delayed Deep Deterministic policy gradient algorithm(TD3) addresses the overestimation bias problem in DDPG by simultaneously training two $$Q$$-functions. In TD3, we always have the actor maximizes $$Q_1$$ but define the target $$Q$$-value as

$$
\begin{align}
r(s,a)+\gamma\min_{i=1, 2}Q^-_{i}(s',\mu^-(s')+\epsilon)
\end{align}
$$


where a small noise $$\epsilon$$ is added to $$\mu^{-}(s')$$ to regularize the $$Q$$ functions, making them robust to small errors. 

[Elevator back to directory](#dir)

### <a name="d4pg"></a>Distributed Distributional DDPG

Distributed Distributional DDPG(D4PG) combines four improvements on DDPG:

1. Ape-X
2. Distributional DQN
3. Multi-step learning
4. PER

[Elevator back to directory](#dir)

## <a name="pg"></a>Stochastic Policy Gradient

### <a name="reinforce"></a>[REINFORCE]({{ site.baseurl }}{% post_url 2018-10-01-PG %})

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

### <a name="trpo"></a>[Trust Region Policy Optimization]({{ site.baseurl }}{% post_url 2018-11-21-PPO %})

Trust Region Policy Optimization(TRPO) defines the constrained surrogate objective as

$$
\begin{align}
\max_\theta J_{\theta_{old}}(\theta)=\mathbb E_{s,a\sim\pi_{\theta_{old}}}\left[A(s,a){\pi_\theta(a|s)\over \pi_{\theta_{old}}(a|s)}\right]\\\
s.t.\quad \mathbb E_{s\sim\pi_{\theta_{old}}}[D_{KL}(\pi_{\theta_{old}}(\cdot|s)\Vert\pi_\theta(\cdot|s))]\le\delta
\end{align}
$$

TRPO solves the above objective following two steps:

1. Make the linear approximation to the objective and quadratic approximation to the constraints, then solve its Lagrangian using the [conjugate gradient method]({{ site.baseurl }}{% post_url 2018-08-27-CG %})
2. Approximate the maximal step size $$\alpha$$ by solving the KL constraint, then do line search to find $$\alpha$$ that improves objective

[Elevator back to directory](#dir)

### <a name="ppo"></a>[Proximal Policy Optimization]({{ site.baseurl }}{% post_url 2018-11-21-PPO %})

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

[Cobbe et al.](http://arxiv.org/abs/2009.04416) provides extensive study on the design choices. They find

- Training the policy with a single epoch yields best performance for well-tuned hyperparameters
- More auxiliary training(more auxiliary epochs) is beneficial. However, performance decreases when we perform auxiliary training too frequently.
- A separate value network is not necessary, one can obtain similar performance by training a detached value head in the policy phase

[Elevator back to directory](#dir)

### <a name="gae"></a>[Generalized Advantage Estimation]({{ site.baseurl }}{% post_url 2018-11-27-GAE %})

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

STAC employs meta-gradient to tune the set of tunable hyperparameters in IMPALA. Besides that it makes two additional changes. First, observing that truncated importance ratio reduces the effect of later TD errors, it uses leaky V-trace that interpolate between the truncated and the canonical importance samplings

$$
\begin{align}
c_{i}:=&\lambda \big(\alpha_c\min(\bar c, \text{IS}_t)+(1-\alpha_c)\text{IS}_t\big)\\\
\rho_k:=&\alpha_\rho\min(\bar\rho, \text{IS}_t)+(1-\alpha_\rho)\text{IS}_t\\\
where\quad \text{IS}_t=&{\pi(a_i|x_i)\over \mu(a_i|x_i)}
\end{align}
$$

Second, it introduces auxiliary tasks that trains additional policy and value heads with different hyperparameters.

### <a name="decisions_pg"></a>Decisions in Policy Gradients

[Andrychowicz et al. 2020]({{ site.baseurl }}{% post_url 2020-09-27-On-Policy-Choices %}) conduct an extensive hyperparameter search on 5 continuous control environments from OpenAI's Gym. They find

- PPO loss performs best when performing multiple passes on the collected data. V-trace policy loss does not work well with these data
- For the control environments tested, $$tanh$$ performs significantly better than $$ReLU$$. Initializing policy head to small value is important.
- Observation normalization is crucial but clipping afterwards is unnecessary. 
- GAE and V-trace yield similar performance when using them to compute the advantage and value target. 
- It is helpful to recompute the advantage and shuffle transitions for every pass over the data.
- The number of transitions gathered in each iteration influences the performance significantly.

