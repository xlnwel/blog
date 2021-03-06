---
title: "RL Reviews 0 — Basics"
excerpt: "Reinforcement learning basics"
categories:
  - Reinforcement Learning
tags:
---

### Prove the convergence of the Bellman optimality update

The Bellman optimality update:

$$
\begin{align}
\mathcal TV(s)=r+\gamma\max_a\mathbb E_{p(s'|s,a)}[V(s')]
\end{align}
$$

If we can prove $$\vert \mathcal TV(s)-\mathcal TV^*(s)\vert <\vert V(s)-V^*(s)\vert $$, where $$V^*(s)$$ is the optimal value function, then we can say the Bellman optimality update converges

$$
\begin{align}
|\mathcal TV(s)-\mathcal TV^*(s)|&=\left|\gamma\max_a\mathbb E_{p(s'|s,a)}[V(s')]-\gamma\max_a\mathbb E_{p(s'|s,a)}[V^*(s')]\right|\\\
&\qquad\color{red}{\text{because } |\max_x f(x)-\max_x g(x)|\le\max_x|f(x)-g(x)|}\\\
&\le\gamma\max_a\mathbb E_{p(s'|s,a)}\left|[V(s')]-[V^*(s')]\right|\\\
&\le\gamma\max_s|V(s)-V^*(s)|
\end{align}
$$

the first ineuqality follows from property

$$
\begin{align}
|\max_x f(x)-\max_x g(x)|\le\max_x|f(x)-g(x)|
\end{align}
$$

the last inequality holds because $$p(s'\vert s,a)$$ are non-negative and sums to one

### What makes function approximation methods unstable in RL?

1. General neural networks and statistical methods all assume a static training set over which multiple passes are made. In RL, the agent interacts with the environment or with a model of its environment, which requires function approximation methods to be able to learn efficiently from incrementally acquired data
2. RL, in addition, requires function approximation methods to be able to handle non-stationary target function

### Deadly triad

Deadly triad refers to the combination of bootstrapping, function approximation, and off-policy learning.

### Prove that baseline does not introduce bias to the policy gradient


$$
\begin{align}
\mathbb E\left[ b(s)\nabla_\theta \log \pi_\theta(a|s)\right]&=\int_{s} b(s)\int_{a}{\nabla_\theta \pi_\theta(a|s)} da ds\\\ 
&=\int_s b(s)\nabla_\theta \int_a\pi_\theta(a|s)dads\\\ 
&=\int_s b(s)\nabla_\theta 1ds\\\ 
&=0
\end{align}
$$


### Explain why baseline may reduce the variance

To see why baseline may reduce the variance, suppose a more general case where we wish to estimate the expect value of the function $$ f: X \rightarrow \mathbb R $$, and we happen to know the value of the integral of another function on the same space $$ g:X\rightarrow \mathbb R $$. We have 


$$
\begin{align}
\mathbb E[f(x)] = \mathbb E[f(x) - g(x)] + \underbrace{\mathbb E[g(x)]}_{a\ known\ constant}=\mathbb E[f(x)-(g(x)-\mathbb E[g(x)])]
\end{align}
$$


Because $$\mathbb E[g(x)]$$ is a known constant, the variance of $$f(x)-(g(x)-\mathbb E[g(x)])$$ is that of $$f - g$$, which is computed as follows


$$
\begin{align}
Var(f - g)&=\mathbb E\Big[ \big(f - g -\mathbb E\left[ f - g \right]\big)^2 \Big]\\\
&=\mathbb E\Big[ \big((f - \mathbb E[f]) - (g-\mathbb E[g])\big)^2 \Big]\\\
&=\mathbb E\Big[ (f - \mathbb E[f])^2 - 2(f-\mathbb E[f])(g-\mathbb E[g]) + (g-\mathbb E[g])^2 \Big]\\\
&=Var(f) -2Cov(f, g) + Var(g)
\end{align}
$$


If $$g$$ and $$ f $$ are strongly correlated so that the covariance term on the right hand side is greater than the variance of $$ g $$, then a variance has been reduced over the original estimation problem. Notice that here we take $$g-\mathbb E[g(x)]$$ as a baseline, which has zero mean and does not introduce any bias.

### Comparison between Monte Carlo and Temporal Difference

1. In practice, TD usually converges faster than MC
2. TC is an online algorithm — it can learn before knowing the outcome. MC, on the other hand, is an offline algorithm — it has to wait until the end of episodes to compute returns. As a result, TD can learn in continuing environments; MC can only learn in episodic environments.
4. TD has low variance, high bias; MC has high variance, zero bias. High variance may result in large update steps which are especially catastrophic for neural networks. That's why we generally use TD target(in off-policy algorithms) or use a large batch size(in on-policy algorithms) when incorporating neural networks into RL algorithms.
5. TD exploits Markov property so that it solves the MDP that best fits the data; MC best fits t the observed return. As a result, TD is usually more efficient in Markov environments; MC on the other hand, usually more effective in non-Markov environments

### Control Variates

To reduce the variance of off-policy $$n$$-step return caused by tiny importance sampling ratios, we introduce an off-policy return with control variates. For state value function, we define the target as

$$
\begin{align}
v(s_t)=\rho_t(r_t+\gamma v(s_{t+1}))+(1-\rho_t)V(s_t)\tag{1}\label{eq:1}
\end{align}
$$

where $$v$$ denotes the off-policy target, $$\rho$$ denotes the importance sampling ratio, and $$V$$ denotes the value function. This reduces the variance since if $$\rho$$ is zero, then instead of the target being zero and causing the estimate to shrink, the target is the same as the estimate and causes no change. The importance sampling ratio being zero means we should ignore the sample, so leaving the estimate unchanged seems appropriate. The second, additional term in Eq.$$\eqref{eq:1}$$ is called a *control variate* (for obscure reasons). Notice that the control variate does not change the expected update; the importance sampling ratio has expected value one(i.e., $$\mathbb E_{b}[{\pi\over b}]=1$$) and is uncorrelated with the estimate($$V(s_t)$$), so the expected value of the control variate is zero. (7.4 Sutton & Barto)

Expanding $$v(s_{t+1})$$, we will have

$$
\begin{align}
v(s_t)&=V(s_t)+\rho_t(r_t-V(s_t)+\gamma v(s_{t+1}))\\\
&=V(s_t)+\rho_t(\underbrace{r_t+\gamma V(s_{t+1})-V(s_t)}_{\delta_t}+\gamma\rho_{t+1}(r_{t+1}+\gamma v(s_{t+2})-V(s_{t+1})))\\\
&=V(s_t)+\rho_t\delta_t+\gamma\rho_t\rho_{t+1}(r_{t+1}+\gamma v(s_{t+2})-V(s_{t+1}))\\\
&=V(s_t)+\sum_{k=t}^{t+n-1}\gamma^{k-t}\prod_{i=t}^{k}\rho_i\delta_k\tag{2}\label{eq:2}\\\
&=V(s_t)+\rho_t\delta_t+\gamma\rho_k\big(v(s_{t+1})-V(s_{t+1})\big)\tag{3}\label{eq:3}
\end{align}
$$

where we take $$v(s_{t+n})=V(s_{t+n})$$ at the last step. Equation $$\eqref{eq:2}$$ is almost the [V-trace introduced in IMPALA]({{ site.baseurl }}{% post_url 2019-11-14-IMPALA %}) except that V-trace truncates $$\rho$$ and treats $$\rho_k$$ and $$\rho_{< k}$$ separately. Equation $$\eqref{eq:3}$$ shows a recursive way to compute $$v(s_t)-V(s_t)$$.

The off-policy target for action values with control variates is 

$$
\begin{align}
q(s_t,a_t)=r_t+\gamma\big(\rho_{t+1}q(s_{t+1},a_{t+1})+ V(s_{t+1})-\rho_{t+1}Q(s_{t+1},a_{t+1})\big)\tag{4}\label{eq:4}\\\
where\quad V(s_t)=\mathbb E_{a\sim\pi(a|s)}[Q(s_t,a_t)]
\end{align}
$$

where $$q$$ denotes the target, $$Q$$ is the action value function. Expanding $$q(s_{t+1},a_{t+1})$$, we will have

$$
\begin{align}
q(s_t,a_t)&=\underbrace{r_t+\gamma V(s_{t+1})-Q(s_t,a_t)}_{\delta_t}+Q(s_t,a_t)+\gamma\rho_{t+1}\big(q(s_{t+1},a_{t+1})-Q(s_{t+1},a_{t+1})\big)\\\
&=\delta_t+Q(s_t,a_t)+\gamma\rho_{t+1}(\delta_{t+1}+\gamma\rho_{t+2}(q(s_{t+2},a_{t+2})-Q(s_{t+2},a_{t+2})))\\\
&=Q(s_t,a_t)+\delta_t+\gamma\rho_{t+1}\big(q(s_{t+1},a_{t+1})-Q(s_{t+1},a_{t+1})\big)\tag{5}\label{eq:5}\\\
&=Q(s_t,a_t)+\sum_{k=t}^{t+n-1}\gamma^{k-t}\prod_{i=t+1}^k\rho_i\delta_k\tag{6}\label{eq:6}
\end{align}
$$

where we take $$q(s_{t+n}, a_{t+n})=Q(s_{t+n}, a_{t+n})$$ at the last step. This is almost the Retrace($$\lambda$$) introduced by R´emi Munos et al. in 2016, except that Retrace($$\lambda$$) truncates $$\rho$$ and additionally multiply $$\rho$$ by $$\lambda$$. Note that the importance sampling starts from $$t+1$$ since action value has specified action at time $$t$$. Equation $$\eqref{eq:6}$$ shows a recursive way to compute $$q(s_t,a_t)-Q(s_t,a_t)$$.

### Compare Policy Gradient Algorithms with Value-Based algorithms

Policy gradient algorithms have some advantages over value-based algorithms:

- Better convergence properties: value-based methods can oscillate even diverge in some cases --- e.g. partially observable Markov decison process(POMDP) since value-based methods are typically built upon Markov property --- while policy-based methods are guaranteed to converge at least to a local optimum if we just directly follow the policy gradient. In addition, in some environments where the agent can be trained on MDP but has to run in POMDP setting, policy gradient provides a way to utilize the MDP by asynchrounously training the value function with the full environment state.

- Effective in high-dimensional or continuous action spaces: value-based methods require a maximization operation to select an action at each step. This operation could be intimidating when action space is large or continuous. Policy-based methods circumvent this operation by adjusting the parameters of the policy directly, and thereby incrementally learn what’s the best action to take at each state

- Can learn stochastic policy: deterministic policy characterized by maximizing value function just doesn’t work in cases, such as rock-paper-scissors and some partially observable environment where two different states may seem the same to the agent. It is necessary to act stochastically in those cases to break the tie

There are also some disadvantages:

- Naive policy-based methods converge very slowly, and have high variance.
- Typically converge to a local rather than global optimum

### Compute Evaluation Value from Behavior Policy Return


$$
\begin{align}
V_{\pi}=\mathbb E_{b}\left[\prod_{i=1}^L{\pi(a_i|s_i)\over b(a_i|s_i)}G_b\right]
\end{align}
$$


with assumption

$$
\begin{align}
b(a|s)>0\quad\text{if }\pi(a|s)>0
\end{align}
$$

As we do in policy gradient, the reward in step $$t$$ does not depend on policy after $$t$$, we could simplify the estimate using

$$
\begin{align}
V_{\pi}=\mathbb E_b\left[\sum_{t=1}^L\gamma^{t-1}R_t\prod_{i=1}^t{\pi(a_i|s_i)\over b(a_i|s_i)}\right]
\end{align}
$$


### Bias and Variance in RL

An estimator $$\hat \theta$$ is biased, suggesting that its expectation does not match the target, i.e., $$\mathbb E[\hat\theta]\ne\theta$$. In contrast, variance refers to how noisy the estimator is.

Monte Carlo is unbiased but of higher variance compared to TD learning. 

- MC is unbiased because by definition the value function is the expectation of returns, i.e, $$v(s)=\mathbb E[G_t]$$. 
- MC has high variance because starting from $$s$$, the return could be very different depending on the action sequence and environment dynamics.
- TD is biased because value function $$V$$ is an estimator not the true value function, so $$r+V(s')$$ is inaccurate
- TD has low variance because $$r+V(s')$$ only depends on the immediate reward $$r$$.

### TD($$\lambda$$)

Write the Bellman operator as

$$
\begin{align}
\mathcal TQ=&r+\gamma PQ\\\
where\quad PQ=&\sum_{x'}\sum_{a'}p(x'|x,a)\pi(a'|x)Q(x',a')
\end{align}
$$

note that we write the Bellman operator in state-based form as we don't consider time steps here. We have $$n$$-step Bellman operator as

$$
\begin{align}
\mathcal T^{n}Q=\mathcal T(\mathcal T^{n-1}Q)=\sum_{i=0}^{n-1}\gamma^{i}r+\gamma^{n} PQ
\end{align}
$$

TD($$\lambda$$) can be written as

$$
\begin{align}
\mathcal T_\lambda Q=&(1-\lambda)\sum_{n=0}^\infty\mathcal \lambda ^n\mathcal T^{n+1}Q\\\
=&(1-\lambda\gamma P)^{-1}(\mathcal T Q-Q)+Q
\end{align}
$$

**Proof**

$$
\begin{align}
\mathcal T_\lambda Q=&(1-\lambda)\sum_{n=0}^\infty\mathcal \lambda ^n\mathcal T^{n+1}Q\\\
=&(1-\lambda)\sum_{n=0}^\infty\lambda ^n\sum_{i=0}^n\gamma^iP^ir +(1-\lambda)\sum_{n=0}^\infty\lambda^n\gamma^{n+1}P^{n+1}Q\\\
=&(1-\lambda)\sum_{i=0}^\infty\gamma^iP^ir\sum_{n=i}^\infty\lambda^n+(1-\lambda)\sum_{n=0}^\infty\lambda^n\gamma^{n+1}P^{n+1}Q\\\
=&(1-\lambda)\sum_{i=0}^\infty\gamma^iP^ir\lambda^i\sum_{n=0}^\infty\lambda^n+(1-\lambda)\sum_{n=0}^\infty\lambda^n\gamma^{n+1}P^{n+1}Q\\\
=&\sum_{i=0}^\infty\gamma^iP^ir\lambda^i++(1-\lambda)\sum_{n=0}^\infty\lambda^n\gamma^{n+1}P^{n+1}Q\\\
=&(1-\lambda\gamma P)^{-1}r+(1-\lambda\gamma P)^{-1}(1-\lambda)\gamma PQ\\\
=&(1-\lambda\gamma P)^{-1}(r+\gamma PQ-\lambda\gamma PQ)\\\
=&(1-\lambda\gamma P)^{-1}(\mathcal T Q-Q+Q-\lambda\gamma PQ)\\\
=&(1-\lambda\gamma P)^{-1}(\mathcal T Q-Q)+Q
\end{align}
$$

We can see that TD($$\lambda$$) converges to the same point as the Bellman operator converges: $$\mathcal TQ=Q\rightarrow \mathcal T_\lambda Q=Q$$

**Recursive version.** We can write TD($$\lambda$$) in a recursive version. As we've seen in the previous proof, we have

$$
\begin{align}
\mathcal T_\lambda Q=&\sum_{i=0}^\infty\gamma^iP^ir\lambda^i+(1-\lambda)\sum_{n=0}^\infty\lambda^n\gamma^{n+1}P^{n+1}Q\\\
=&\sum_{n=0}^\infty (\lambda\gamma)^nP^nr+(1-\lambda)\lambda^n\gamma^{n+1}P^{n+1}Q
\end{align}
$$

If we add time step back to TD($$\lambda$$), we obtain

$$
\begin{align}
\mathcal T_\lambda Q_t
=&\sum_{n=0}^\infty (\lambda\gamma)^nP^nr_{t+n}+(1-\lambda)\lambda^n\gamma^{n+1}P^{n+1}Q_{t+n}\\\
=&r_t+(1-\lambda)\gamma PQ_t+\sum_{n=1}^\infty (\lambda\gamma)^nP^nr_{t+n}+(1-\lambda)\lambda^n\gamma^{n+1}P^{n+1}Q_{t+n}\\\
=&r_t+(1-\lambda)\gamma PQ_t+\lambda\gamma\sum_{n=0}^\infty (\lambda\gamma)^nP^nr_{t+1+n}+(1-\lambda)\lambda^n\gamma^{n+1}P^{n+1}Q_{t+1+n}\\\
=&r_t+(1-\lambda)\gamma PQ_t+\lambda\gamma\mathcal T_\lambda Q_{t+1}
\end{align}
$$
