---
title: "Exponential Families"
excerpt: "Discussion on Exponential Famlies"
categories:
  - Mathematics
tags:
  - Mathematics
---

## Definition

Given a measure $$\eta$$, an exponential family of probability distributions is defined as 

$$
\begin{align}
p(x|\eta)=h(x)\exp\left\{\eta^TT(x)-A(\eta)\right\}\\\
where\quad A(\eta)=\log\int h(x)\exp\left\{\eta^TT(x)\right\}dx
\end{align}
$$

where $$\eta$$ is a function of the parameter $$\theta$$. $$\eta$$ and $$\eta(\theta)$$ are identical and we only use the latter notation when necessary.

### Notation Explanations

$$T(x)$$ is a *sufficient statistic* of the distribution. For exponential families, the sufficient statistic is a *function of the data* that holds all information of data $$x$$ provides with regard to the unknown parameters values. It encapsulates all the information needed to describe the posterior distribution of the parameters, given the data (i.e., $$p(\theta\vert T(x),x)=p(\theta\vert T(x))$$ or $$p(x\vert T(x),\theta)=p(x\vert T(x))$$). This also means that, for any data sets $$x$$ and $$y$$, if $$T(x)=T(y)$$, the likelihood ratio of the probability distributions w.r.t. any two parameters is the same 

$$
\begin{align}
{p(x;\theta_1)\over p(x;\theta_2)}&=\exp\left\{(\eta(\theta_1)-\eta(\theta_2))^{T}T(x)-(A(\eta_1)-A(\eta_2))\right\}\\\
&=\exp\left\{(\eta(\theta_1)-\eta(\theta_2))^{T}T(y)-(A(\eta_1)-A(\eta_2))\right\}={p(y;\theta_1)\over p(y;\theta_2)}
\end{align}
$$

$$\eta$$ is referred to as the *natual parameter* or *canonical parameter*. The set of parameters $$\eta$$ for which the cumulant function is finite is referred to as the *natural parameter space*:

$$
\begin{align}
\mathcal N=\left\{\eta:A(\eta)<\infty\right\}
\end{align}
$$

Exponential families are referred to as *regular* if the natural parameter space is nonempty open set.

$$A(\eta)$$ is known as the *cumulant function*, or log-partition function as it is the logarithm of a normalization factor. 

An exponential family is referred to as *minimal* if the components of $$\eta(\theta)$$ are linearly independent and so are those of $$T(x)$$. Non-minimal families can always be reduced to minimal families via a suitable transiformation and reparameterization.

## Examples

### The Binomial Distribution

We take $$n$$ independent experiments, each with an boolean-valued outcome. Let $$x$$ be the number of times  successes, $$\theta$$ be the probability of success. We have

$$
\begin{align}
p(x|\theta)&={n!\over x!(n-x)!}\theta^x(1-\theta)^{n-x}\\\
&={n!\over x!(n-x)!}\exp\left\{x\log\theta+(n-x)\log(1-\theta)\right\}\\\
&={n!\over x!(n-x)!}\exp\{x\log{\theta\over 1-\theta}+n\log(1-\theta)\}
\end{align}
$$

so we see that the Bernoulli distribution is an exponential family distribution with

$$
\begin{align}
\eta&=\log{\theta\over 1-\theta}\\\
T(x)&=x\\\
A(\eta)&=-n\log(1-\theta)=n\log (1+e^\eta)\\\
h(x)&={n!\over x!(n-x)!}
\end{align}
$$

Moreover, the relationship between $$\eta$$ and $$\theta$$ is invertible:

$$
\begin{align}
\theta={1\over1+e^{-\eta}}
\end{align}
$$

which is the *logistic function* (or *sigmoid function*). This is commonly used as the last activation function in deep neural networks for binary classification problems, where $$\eta$$ is the output of the neural net, the input of the sigmoid function.

### The Multinomial Distribution

We take $$n$$ independent experiments, of which the outcome has a categorical distribution. Let's say we have $$K$$ categories. Let $$x_i$$ be the total number of times the $$i$$th event occurs, $$\theta_i$$ be the probability of the $$i$$th event occurring in any given trial. We have

$$
\begin{align}
p(x|\theta)&={n!\over x_1!x_2!\dots x_K!}\theta_1^{x_1}\theta_2^{x_2}\dots\theta_K^{x_K}\\\
&={n!\over \sum_{k=1}^Kx_k!}\exp\left\{\sum_{k=1}^Kx_k\log\theta_k\right\}\\\
&={n!\over \sum_{k=1}^Kx_k!}\exp\left\{\sum_{k=1}^{K-1}x_k\log\theta_k+\left(n-\sum_{k=1}^{K-1}x_k\right)\log\left(1-\sum_{k=1}^{K-1}\theta_k\right)\right\}\\\
&={n!\over \sum_{k=1}^Kx_k!}\exp\left\{\sum_{k=1}^{K-1}x_k\log{\theta_k\over 1-\sum_{i=1}^{K-1}\theta_i}+n\log \left(1-\sum_{k=1}^{K-1}\theta_k\right)\right\}
\end{align}
$$

where in step third, we use the facts that $$\sum_{k=1}^Kx_k=n$$ and $$\sum_{k=1}^K\theta_k=1$$.

To align it with the exponential family, we have

$$
\begin{align}
\eta_k&=\log{\theta_k\over 1-\sum_{i=1}^{K-1}\theta_i}=\log{\theta_k\over\theta_K}\\\
T(x)_k&=x_k\\\
A(\eta)&=-n\log \left(1-\sum_{k=1}^{K-1}\theta_k\right)=n\log \left(\sum_{k=1}^Ke^{\eta_k}\right)\\\
h(x)&={n!\over \sum_{k=1}^Kx_k!}
\end{align}
$$

If we further define $$\eta_K=\log{\theta_{K}\over\theta_K}=0$$, we could compute $$\theta_K$$:

$$
\begin{align}
\sum_{k=1}^Ke^{\eta_k}={1\over\theta_K}\\\
\theta_K={1\over\sum_{k=1}^Ke^{\eta_k}}={e^{\eta_K}\over\sum_{k=1}^Ke^{\eta_k}}
\end{align}
$$

then we derive $$\theta_k$$ from $$\eta$$

$$
\begin{align}
\theta_k&={\theta_k\over \theta_K}\theta_K\\\
&={e^{\eta_k}\over\sum_{k=1}^Ke^{\eta_k}}
\end{align}
$$


we can see that adding a constant to all $$\eta_k$$ does not change the value of $$\theta_k$$, which suggest that we do not restrict the above result to $$\eta_K=0$$. This equation is exactly the softmax activation we use in deep neural networks when we do multi-categorical classification.

### The Univariate Gaussian Distribution

The univariate Gaussian density can be written as follows

$$
\begin{align}
p(x|\mu,\sigma^2)&={1\over\sqrt{2\pi\sigma^2}}\exp\left\{-{(x-\mu)^2\over 2\sigma^2}\right\}\\\
&={1\over\sqrt{2\pi}}\exp\left\{-{1\over2\sigma^2}x^2+{\mu\over\sigma^2}x-{\mu^2\over 2\sigma^2}-\log\sigma\right\}
\end{align}
$$

This is in the exponential family form, with

$$
\begin{align}
\eta&=\begin{bmatrix}-1/2\sigma^2\\\\mu/\sigma^2\end{bmatrix}\\\
T(x)&=\begin{bmatrix}x^2\\\x\end{bmatrix}\\\
A(\eta)&={\mu^2\over 2\sigma^2}+\log\sigma\\\
h(x)&={1\over\sqrt{2\pi}}
\end{align}
$$


### The Multivariate Gaussian Distribution

The multivariate Gaussian distribution is

$$
\begin{align}
p(x|\mu,\Sigma)&={1\over\sqrt{|2\pi\Sigma|}}\exp\left\{-{1\over2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right\}\\\
&=\exp\left\{-{1\over 2}\left(x^T\Sigma^{-1} x-2\mu^T\Sigma^{-1}x+\mu\Sigma^{-1}\mu+\log|2\pi\Sigma|\right)\right\}
\end{align}
$$

It is easy to decompose the second term in the exponential. We now analyze the first term in the exponential:

$$
\begin{align}
x^T\Sigma^{-1} x&=\mathrm{Tr}(x^T\Sigma^{-1} x)\\\
&=\mathrm{Tr}(\Sigma^{-1} xx^T)\\\
&=\sum_i\sum_j\Sigma^{-1}_{i,j} (xx^T)_{j,i}\\\
&=\sum_i\sum_j\Sigma^{-1}_{j,i} (xx^T)_{j,i}\\\
&=\mathrm{vec}(\Sigma^{-1})^T\mathrm{vec}(xx^T)
\end{align}
$$

where Trace operator is used in the first step since $$x^T\Sigma^{-1}x$$ is a scalar, the fourth step is obtained because $$\Sigma^{-1}$$ is symmetric, and we apply vectoring operator at the last step. Now we can easily see the exponential family form of the multivariate Gaussian

$$
\begin{align}
\eta&=\begin{bmatrix}-{1\over 2}\mathrm{vec}(\Sigma^{-1})\\\\Sigma^{-1}\mu\end{bmatrix}\\\
T(x)&=\begin{bmatrix}\mathrm{vec}(xx^T)\\\x\end{bmatrix}\\\
A(\eta)&={1\over 2}\mu\Sigma^{-1}\mu+{1\over 2}\log|2\pi\Sigma|\\\
h(x)&=1
\end{align}
$$


### Mean and Variance of The Sufficient Statistic

The first derivative of the cumulant function is the mean of the sufficient statistic

$$
\begin{align}
{\partial A(\eta)\over \partial\eta^T}&=\int h(x)\exp\{\eta^TT(x)-A(\eta)\}T(x)dx\\\
&=\int p(x|\eta)T(x)dx\\\
&=\mathbb E_{x\sim p(x|\eta)}[T(x)]
\end{align}
$$

The second derivative of the cumulant function is the variance of the sufficient statistic

$$
\begin{align}
{\partial^2 A(\eta)\over \partial\eta\partial\eta^T}&=\int h(x)\exp\{\eta^TT(x)-A(\eta)\}T(x)\left(T(x)-{\partial A(\eta)\over \partial \eta}\right)^Tdx\\\
&=\mathbb E[T(x)T(x)]-\mathbb E[T(x)]\mathbb E[T(x)]\\\
&=Var[T(X)]
\end{align}
$$


Especially, when the random variable equals to the sufficient statistic, $$X=T(X)$$, this gives us the mean and variance of that random variable.

### Miscellanea

The natural exponential family has conjugate prior

$$
\begin{align}
p(\theta)=\exp(\eta(\theta)T(\theta)-\log Z(\theta)))
\end{align}
$$




$$
\begin{align}
\sigma^2+\mu^2=\mathbb E[X^2]
\end{align}
$$


This suggest that $$\bar X$$ is Gaussian with mean $$\mu$$ and variance $$\sigma^2/n$$

## References

[The Exponential Family - People @EECS at UC Berkeley](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf)