---
title: "SL — Statistic Learning: A Connection to Neural Networks"
excerpt: "We expand the topic of latent variable models in a sense that the latent variables model the underlying structure of the observed data, whereby the model is able to do statistical inference over these latent variables. Then we will build a connnection between statistic learning and neural networks."
categories:
  - Mathematics
tags:
  - Mathematics
---

## Introduction

In the previous post, we talk about probabilistic latent variable models that use latent variables to introduce randomness so that we are able to model the marginal likelihood of the observed data. In this post, we will continue our discussion on latent variable models in a sense that the latent variables model the underlying structure of the observed data, wherey the model is able to do statistical inference over these latent variables. Then we will build a connection between these statistic learning and neural networks.

## Statistic Learning

### Bayesian Learning

Bayesian learning calculates the probability of each latent variables(or hypothesis), given the data, and make predictions on that basis. Mathematically, Bayesian learning first computes the probability distribution of the latent variable conditioned on the data

$$
\begin{align}
p(z_i|\mathcal D)={p(\mathcal D|z_i)p(z_i)\over \sum_j p(\mathcal D|z_j)p(z_j)}=\alpha p(\mathcal D|z_i)p(z_i)\tag {1}
\end{align}
$$

where \\( z_i \\) is the latent variable and \\( \mathcal D \\) represents the observed data. Then it makes prediction about a new quantity \\( X \\) as follows

$$
\begin{align}
p(X|\mathcal D)=\sum_ip(X|\mathcal D,z_i)p(z_i|\mathcal D)=\sum_ip(X|z_i)p(z_i|\mathcal D)\tag {2}
\end{align}
$$


### Maximum A Posteriori (MAP)

Bayesian learning is hard and even intractable for large problems, since it requires summation (or integration) over the latent variable \\( z \\). MAP simplifies Bayesian learning by predicting based on a single most probable hypothesis, and therefore transforming a large summation (or integration) problem into an optimization problem. 

More specifically, MAP makes predictions approximately using

$$
\begin{align}
p(X|\mathcal D)&\approx p(X|z_{MAP})\\\
where\quad z_{MAP}&=\arg\max_{z}p(\mathcal D|z)p(z)
\end{align}\tag{3}
$$

Generally, we assume data are independent of each other, and take the negative of the logarithm to avoid hard multiplication and turn the maximization into minimization. Then we have

$$
\begin{align}
z_{MAP}=\arg\min_{z}-\log p(\mathcal D|z)-\log p(z)\tag {4}
\end{align}
$$

In fact, both Bayesian and MAP learning methods use the prior to *penalize complexity*. Typically, more complex latents have a lower prior probability — in part because there are usually many more complex latents than simple latents. This could also be seen in Eq.\\( (4) \\): the last term \\( -\log p(z) \\) actually measures the bits required to specify the latent \\( z \\). By minimizing \\( -\log p(z) \\), we reduce the bits required for \\( z_i \\), which means \\( z_i \\) become more simple, and therefore more uncertain.

### Maximum Likelihood Estimation (MLE)

MLE takes one step further to simplify MAP by assuming that uniform prior over the space of latent variables. As a result, Eq.\\( (4) \\) in MLE becomes

$$
\begin{align}
z_{MLE}=\arg\min_{z}-\log p(\mathcal D|z)\tag {5}
\end{align}
$$

MLE provides a good approximation to Bayesian and MAP learning when the dataset is large, because the data swamps the prior distribution over the latents, but it can be problematic with small datasets, which could be further excerbated when the dataset is small enough that some events have not yet been observed — the ML hypothesis assign zero probability to those events. Various tricks are used to avoid this problem, such as initializing the counts for each event to \\( 1 \\) instead of \\( 0 \\), but these are mostly a hack. 

In general MLE could be solved by a standard process as follows

1. Write the log likelihood of the data as a function of parameter(s): \\( \log p(\mathcal D\vert z) \\)
2. Compute the gradients with respective to each parameter
3. Find the parameter values such that the gradients are zero (do gradient descent, or calculate it directly if the problem is simple)

## A Connection Between Statistic Learning and Neural Networks

To build a connection between statistic learning and neural networks, we first change the name convention, replacing the latent variables \\( z \\) with parameters \\( \theta \\). For better illustration, we take supervised learning as examples throughout the discussion.

In supervised learning, we generally have the objective

$$
\begin{align}
\min_{\theta}-\log p(y|x;\theta)
\end{align}
$$

This is the same objective in MLE, where the data is partitioned into two parts — inputs \\( x \\) and output labels \\( y \\). In fact, if we analyze MAP in this fashion, we can easily see that \\( -\log p(\theta) \\) is L2 regularization when \\( \theta \\) has a Gaussian prior, and L1 regularization when \\( \theta \\) has Laplace prior, where zero mean and fixed variance are assumed:

$$
\begin{align}
\min_\theta-\log\Bigg({1\over\sqrt{2\pi\sigma^2}}\exp\left(-{(\theta-\mu)^2\over2\sigma^2}\right)\Bigg)&=\min\theta^2\\\
\min_\theta-\log\Bigg({1\over 2b}\exp\left(-{|\theta-\mu|\over b}\right)\Bigg)&=\min|\theta|
\end{align}
$$

The bayesian-learning equivalent is a little tricky since the integral over the parameter space is generally intractable. However, the Bayesian network architecture, where each parameter has its own probability distribution, is used in many places, such as Tompson sampling. The simpliest example may be the noisy network we discussed back when we talked about Rainbow, in which parameters are assumed to be of Gaussian distribution with fixed variance.
