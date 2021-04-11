---
title: "The Mirage of Action-Dependent Baselines"
excerpt: "Analysis on action-dependent baselines"
categories:
  - Reinforcement Learning
tags:
  - Methodology
  - Policy-Gradient RL
---

## Introduction

We follow the work of [Tucker et al.](#ref1), showing that action-dependent baselines do not reduce the variance in practice and the "success" of many recent works on that topic is due to the bias-variance tradeoff made in their implementation.

## State-Dependent Baselines

Policy gradient reinforcement learning maximizes the following objective

$$
\begin{align}
\mathcal J=\mathbb E[\hat A(s,a,\tau)]\tag 1
\end{align}
$$

where \\(\hat A\\) is an estimator of the advantage function up to a state-dependent constant computed from trajectory \\(\tau\\) (e.g., \\(\sum_t\gamma^tr_t\\) and the GAE estimator).

This objective in Equation \\((1)\\) often results in a policy gradient estimator of high variance due to the variance of \\(\hat A\\). One common technique to reduce the variance of the policy gradient estimator is baselines. Baselines are often state-dependent and of form \\(\phi(s)\\). With a state-dependent baseline, the policy gradient is

$$
\begin{align}
\hat g_s(s,a,\tau)=\left(\hat A(s,a,\tau)-\phi(s)\right)\nabla\log\pi(a|s)\tag 2
\end{align}
$$

We now show that state-dependent baselines effectively reduce the variance. Let \\(f=\hat A(s,a)\nabla\log\pi(a\vert s)\\) and \\(g=\phi(s)\nabla\log\pi(a\vert s)\\). We compute the variance of the gradient estimator after subtracting a state-dependent baseline as follows

$$
\begin{align}
\text{Var}\Big(f-g\Big)
=&\mathbb E\left[((f-g)-\mathbb E[f-g])^2\right]\\\
=&\mathbb E\left[((f-\mathbb E[f])-(g-\mathbb E[g]))^2\right]\\\
=&\mathbb E\left[(f-\mathbb E[f])^2\right]-2\mathbb E\left[(f-\mathbb E[f])(g-\mathbb E[g])\right]+\mathbb E\left[(g-\mathbb E[g])^2\right]\\\
=&\text{Var}(f)-2\text{Cov}(f,g)+\text{Var}(g)\tag 3
\end{align}
$$

Equation \\((3)\\) shows that the introduction of \\(g\\)(the baseline gradient) reduces the variance as long as \\(2\text{Cov}(f,g)>\text {Var}(g)\\). In other words, the variance is reduced if \\(f\\) and \\(g\\) are highly correlated or \\(g\\) has low variance. As a result, we often use the state-value function as a baseline, which appropriately meets these two requirements.

## Action-Dependent Baselines

Several methods have extended the approach to state-action-dependent baselines of form \\(\phi(s,a)\\). With a state-action dependent baseline, the policy gradient is

$$
\begin{align}
\hat g_a(s,a,\tau)=\left(\hat A(s,a,\tau)-\phi(s,a)\right)\nabla\log\pi(a|s)+\nabla \mathbb E_{\pi}[\phi(s,a)]\tag 4
\end{align}
$$

Different from Equation \\((2)\\), the last term is necessary as \\(\nabla\mathbb E_\pi[\phi(s,a)]\ne 0\\).

The variance of Equation \\((4)\\), \\(\Sigma:=\text{Var}_{s, a, \tau}(\hat g_a(s, a, \tau))\\), can be decomposed using the law of total variance

$$
\begin{align}
\Sigma
=&\mathbb E_s\left[\text{Var}_{a, \tau|s}(\hat g_a(s, a, \tau))\right] + \text{Var}_s(\mathbb E_{a, \tau|s}[\hat g(s, a, \tau)])\\\
&\quad\color{red}{\text{apply the law of total variance}}\\\
=&\mathbb E_s\left[\text{Var}_{a, \tau|s}\left(\left(\hat A(s,a,\tau)-\phi(s,a)\right)\nabla\log\pi(a|s)+\underbrace{\nabla \mathbb E_{\pi}[\phi(s,a)]}_{\text{constant for a given }s}\right)\right]\\\
&\quad +\text{Var}_s\left[\mathbb E_{a, \tau|s}\left(\hat A(s,a,\tau)\nabla\log\pi(a|s)\underbrace{-\phi(s,a)\nabla\log\pi(a|s)+\nabla \mathbb E_{\pi}[\phi(s,a)]}_{\text{canceled under the expectation of }a}\right)\right]\\\
&\quad\color{red}{\text{omit constants}}\\\
=&\mathbb E_s\left[\text{Var}_{a, \tau|s}\left(\left(\hat A(s,a,\tau)-\phi(s,a)\right)\nabla\log\pi(a|s)\right)\right]\\\
&\quad +\text{Var}_s\left[\mathbb E_{a, \tau|s}\left(\hat A(s,a,\tau)\nabla\log\pi(a|s)\right)\right]\tag 5\\\
&\quad\color{red}{\text{apply the law of total variance to the first term and let }\hat A(s,a)=\mathbb E_{\tau|s, a}[\hat A(s, a, \tau)]}\\\
=&\underbrace{\mathbb E_{s,a}\left[\text{Var}_{\tau|s, a}\big(\hat A(s, a, \tau)\nabla\log\pi(a|s)\big)\right]}_{\Sigma_\tau}\\\
&\quad+\underbrace{\mathbb E_s\left[\text{Var}_{a|s}\left(\big(\hat A(s,a)-\phi(s,a)\big)\nabla\log\pi(a|s)\right)\right]}_{\Sigma_a}\\\
&\quad +\underbrace{\text{Var}_s\left[\mathbb E_{a, \tau|s}\left(\hat A(s,a,\tau)\nabla\log\pi(a|s)\right)\right]}_{\Sigma_s}\tag 6\\\
\end{align}
$$

Equation \\((6)\\) decomposes the variance in the on-policy gradient estimate into three terms, where \\(\Sigma_\tau\\) describes the variance due to sampling a single \\(\tau\\), \\(\Sigma_a\\) describes the variance due to sampling a single \\(a\\), and \\(\Sigma_s\\) describes the variance coming from visiting a limited number of states.

From Equation \\((6)\\), we can see that \\(\phi\\) only affects \\(\Sigma_a\\) and \\(\Sigma_a\\) diminishes when \\(\phi(s,a)=\hat A(s,a)=\mathbb E_{\tau\vert s,a}[\hat A(s, a,\tau)]\\). Regarding the relative magnitude of these terms, the effectiveness of the optimal state-action-dependent baseline varies. This benefit is further restricted as, in practice, we can only approximate \\(\hat A(s,a)\\) with a function approximator. Several experiments conducted by [Tucker et al. 2018](#ref1) also show that a learned state-action-dependent baseline does not reduce variance over a state-dependent baseline.

## Unveiling the Mirage

### IPG

IPG([Gu et al. 2017](#ref2)) normalizes the learning signal \\(\hat A(s, a,\tau)-\phi(s, a)\\) but not the bias correction term \\(\nabla\mathbb E_a(\phi(s,a))\\), which gives

$$
\begin{align}
\hat g_{IPG}(s, a, \tau)={1\over\hat \sigma}\left(\hat A(s,a,\tau)-\phi(s,a)-\hat\mu\right)\nabla\log\pi(a|s)+\nabla \mathbb E_{\pi}[\phi(s,a)]\tag 7
\end{align}
$$

where \\(\hat\mu\\) and \\(\hat \sigma\\) are batch-based estimates of the mean and standard deviation of \\(\hat A(s, a,\tau)-\phi(s, a)\\). 

Defining \\(\lambda={1\over\hat\sigma}\\), we can rewrite Equation \\((7)\\) in a more general form

$$
\begin{align}
\hat g_{IPG}(s, a, \tau)=\lambda\left(\hat A(s,a,\tau)-\phi(s,a)\right)\nabla\log\pi(a|s)+\nabla \mathbb E_{\pi}[\phi(s,a)]\tag 8
\end{align}
$$

where \\(\hat\mu\\) is omitted as it's a constant and \\(\nabla \mathbb E[\hat\mu]=0\\).

Equation \\((8)\\) introduces bias—when \\(\lambda\ne 1\\)—as the first term weights differently from the bias correction term. We can compute the bias by subtracting Equation \\((7)\\) from Equation \\((4)\\), which gives us

$$
\begin{align}
\text{Bias}(\hat g_{IPG})=(1-\lambda)\left(\hat A(s,a,\tau)-\phi(s,a)\right)\nabla\log\pi(a|s)
\end{align}
$$

On the other hand, Equation \\((8)\\) indeed reduces variance for \\(\lambda< 1\\). Using the law of total variance, we have the variance of the estimator as

$$
\begin{align}
\text{Var}(\hat g_{IPG}(s,a,\tau))=&\lambda^2\mathbb E_s\left[\text{Var}_{a, \tau|s}\left(\left(\hat A(s,a,\tau)-\phi(s,a)\right)\nabla\log\pi(a|s)\right)\right]\\\
&\quad +\text{Var}_s\mathbb E_{a|s}\left[\left(\lambda\mathbb E_{\tau|s, a}\left[\hat A(s,a,\tau)\right]+(1-\lambda)\phi(s,a)\right)\nabla\log\pi(a|s)\right]
\end{align}
$$

When \\(\phi(s,a)\approx\mathbb E_{\tau\vert s, a}[\hat A(s ,a, \tau)]\\), introducing \\(\lambda\\) effectively reduce the variance of the first term by \\(\lambda^2\\).

Therefore, the observed empirical performance gain of IPG is mainly from the bias and variance trade off.

## References

<a name="ref1"></a>Tucker, George, Surya Bhupatiraju, Shixiang Gu, Richard E Turner, and Zoubin Ghahramani. 2018. “The Mirage of Action-Dependent Baselines in Reinforcement Learning.”

<a name="ref2"></a>Gu, Shixiang, Timothy Lillicrap, Zoubin Ghahramani, Richard E. Turner, Bernhard Schölkopf, and Sergey Levine. 2017. “Interpolated Policy Gradient: Merging on-Policy and off-Policy Gradient Estimation for Deep Reinforcement Learning.” *ArXiv*, no. Nips.