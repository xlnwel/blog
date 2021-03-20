---
title: "Discrete Probability Distributions"
excerpt: "Discussion on several discrete probability distributions in statistics"
categories:
  - Mathematics
tags:
  - Mathematics
---

## Table of Contents

- [Binary Variables](#bv)
  - [Bernoulli Distribution](#bernoulli)
  - [Binomial Distribution](#binomial)
  - [Beta Distribution](#beta)
- [Multinomial Variables](#mn)
  - [Multinomial Distribution](#multinomial)
  - [Dirichlet Distribution](#dirichlet)
- [Poisson Distribution](#poisson)

## <a name="bv"></a>Binary Variables

For a binary variable, we define the Bernoulli distribution as

$$
\begin{align}
\text{Ber}(x|\mu)=\mu^x(1-\mu)^{1-x}\tag{1}\label{eq:1}
\end{align}
$$

where binary variable $$x\in\{0,1\}$$, and $$\mu\in[0,1]$$ is the probability of $$x=1$$.

The mean and variance of the Bernoulli distribution are

$$
\begin{align}
\mathbb E[x]&=\mu\\\
\text{var}[x]&=\mu(1-\mu)
\end{align}
$$

Suppose we have a dataset $$\mathcal D=\{x_1,\dots,x_N\}$$ of i.i.d. observed values of $$x$$. The likelihood function can be expressed as

$$
\begin{align}
p(\mathcal D|\mu)=\prod_{n=1}^Np(x_n|\mu)=\prod_{n=1}^N\mu^{x_n}(1-\mu)^{1-x_n}\tag{2}\label{eq:2}
\end{align}
$$

We can estimate a value for $$\mu$$ by maximizing the logarithm of the likelihood function

$$
\begin{align}
\ln p(\mathcal D|\mu)=\sum_{n=1}^N\big(x_n\ln\mu+(1-x_n)\log(1-\mu)\big)
\end{align}
$$

If we set its derivative to zero, we obtain the maximum likelihood estimator for $$\mu$$

$$
\begin{align}
\mu_{ML}={1\over N}\sum_{n=1}^Nx_n \tag{3}\label{eq:3}
\end{align}
$$


### <a name="binomial"></a>Binomial Distribution

We can extend the Bernoulli distribution to the binomial distribution, in which we have $$N$$ variables of a Bernoulli distribution and consider the probability of the number of successes. The binomial distribution can be written as

$$
\begin{align}
\text{Bin}(m|N,\mu)=\begin{pmatrix}
N\\\
m
\end{pmatrix}
\mu^m(1-\mu)^{N-m}\tag{4}\label{eq:4}
\end{align}
$$

The mean and variance of the binomial distribution are given by

$$
\begin{align}
\mathbb E[m]&=N\mathbb E[\text{Ber}(\mu)]=N\mu\\\
\text{var}[m]&=N\text{var}[m]=N\mu(1-\mu)
\end{align}
$$


### <a name="beta"></a>Beta Distribution

The maximum likelihood estimation given by Equation $$\eqref{eq:3}$$ can easily overfit for small dataset, especially when we have all observations of $$1$$ or $$0$$. We can mitigate such overfitting by introducing a prior distribution on the parameters. For binary variables, we choose the beta distribution for $$\mu$$, which is defined as

$$
\begin{align}
\text{Beta}(\mu|a,b)&={\Gamma(a+b)\over\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}\propto\mu^{a-1}(1-\mu)^{b-1}\tag{5}\label{eq:5}\\\
\Gamma(x)&=\int_0^\infty u^{x-1}e^{-u}du
\end{align}
$$

Where $$\Gamma(x)$$ is the gamma function(see [Appendix 1](#app1) for further discussion), the coefficient ensures that the beta distribution is normalized.

The mean and variance of the beta distribution are given by

$$
\begin{align}
\mathbb E[\mu]&={a\over a+b}\\\
\text{var}[\mu]&={ab\over(a+b)^2(a+b+1)}
\end{align}
$$

The parameters $$a$$ and $$b$$ are often called *hyperparameters* because they control the distribution of the parameter $$\mu$$.

The posterior distribution of $$\mu$$ is obtained following the Bayes' theorem

$$
\begin{align}
p(\mu|m,l,a,b)&\propto\mu^{m+a-1}(1-\mu)^{l+b-1}\\\
p(\mu|m,l,a,b)&={\Gamma(a+l+b+m)\over\Gamma(a+m)\Gamma(b+l)}\mu^{m+a-1}(1-\mu)^{l+b-1}\tag{6}\label{eq:6}
\end{align}
$$

Where $$m$$ and $$l$$ are the numbers of successes and failures observed, respectively. Equation $$\eqref{eq:6}$$ allows us incrementally update the distribution of $$\mu$$ as new data is observed. We can see that the beta distribution is a conjugate distribution of the binomial distribution. 

If our goal is to predict, we can follow the sum and product rules of probability

$$
\begin{align}
p(x=1|\mathcal D)=\int_0^1p(x=1|\mu)p(\mu|\mathcal D)d\mu=\int_0^1\mu p(\mu|\mathcal D)d\mu=\mathbb E[\mu|\mathcal D]={a+m\over a+m+b+l}
\end{align}
$$


## <a name="mn"></a>Multinomial Variables

We generally represent a variable that can take on one of $$K$$ values by a one-hot $$K$$-dimensional vector $$\pmb x$$. 

If we denote the probability of $$x_k=1$$ by the parameter $$\mu_k$$, then the distribution of $$\pmb x$$ is given

$$
\begin{align}
p(\pmb x|\pmb \mu)=\prod_{k=1}^K\mu_k^{x_k}\tag{7}\label{eq:7}\\\
s.t.\quad\sum_{k=1}^K\mu_k=1,\ \mu_k>0
\end{align}
$$

where $$\pmb \mu=(\mu_1,\dots,\mu_K)^T$$.

The expected value is

$$
\begin{align}
\mathbb E[\pmb x|\pmb \mu]=\sum_{\pmb x}p(\pmb x|\pmb\mu)\pmb x=\pmb\mu
\end{align}
$$

Suppose we have a dataset $$\mathcal D=\{\pmb x_1,\dots,\pmb x_N\}$$ of i.i.d. observed values of $$\pmb x$$. The likelihood function can be expressed as

$$
\begin{align}
p(\mathcal D|\pmb\mu)=\prod_{n=1}^N p(\pmb x_n|\mu)=&\prod_{n=1}^N\prod_{k=1}^K\mu_k^{x_{nk}}=\prod_{k=1}^K\mu_k^{\sum_n x_{nk}}=\prod_{k=1}^K\mu_k^{m_k}\tag{8}\label{eq:8}\\\
where\quad m_k=&\sum_nx_{nk}\\\
s.t.\quad\sum_{k=1}^K\mu_k=&1,\ \mu_k>0
\end{align}
$$

We can maximize the likelihood of $$\pmb \mu$$ using a Lagrangian multiplier $$\lambda$$ to take account for the constraint

$$
\begin{align}
\ln p(\mathcal D|\mu)=\sum_{k=1}^Km_k\ln\mu_k +\lambda\left(1-\sum_{k=1}^K\mu_k\right)
\end{align}
$$

If we set its derivative to zero, we obtain the maximum likelihood estimator for $$\mu_k$$

$$
\begin{align}
\mu_k={m_k\over\lambda}\tag{9}\label{eq:9}
\end{align}
$$

We can obtain the Lagrangian multiplier by substituting Equation $$\eqref{eq:9}$$ into the constraint, which gives us $$\lambda=N$$. Therefore, we have

$$
\begin{align}
\mu_k^{ML}={m_k\over N}
\end{align}
$$


### <a name="multinomial"></a>Multinomial Distribution

We can consider the joint distribution of the quantities $$m_1,\dots, m_K$$ conditioned on the parameters $$\pmb\mu$$ and on the total number $$N$$ of observations. This gives us

$$
\begin{align}
\text{Mult}(m_1,m_2,\dots,m_K)=\begin{pmatrix}N\\\
m_1m_2\dots m_K\end{pmatrix}\prod_{k=1}^K\mu_k^{m_k}
\end{align}
$$


### <a name="dirichlet"></a>Dirichlet Distribution

Similar to the beta distribution for the binomial distribution, the Dirichlet distribution is a conjugate distribution for the multinomial distribution. It takes the form

$$
\begin{align}
\text{Dir}(\pmb\mu|\pmb\alpha)=&{\Gamma(\alpha_0)\over\Gamma(\alpha_1)\dots\Gamma(\alpha_K)}\prod_{k=1}^K\mu_k^{\alpha_k-1}\propto\prod_{k=1}^K\mu_k^{\alpha_k-1}\\\
where\quad\alpha_0=&\sum_{k=1}^K\alpha_k
\end{align}
$$


Where $$\pmb \alpha$$ is a $$K$$-dimension vector of positive numbers. Sampling from a Dirichlet distribution gives us a categorical distribution. According to [this answer](https://qr.ae/pNyGKz), if we factorize $$\pmb \alpha$$ into a probability distribution and a normalization constant, then the probability distribution happens to be the mean of the Dirichlet distribution. Furthermore, the constant is related to the variance: the variance is high for a larger constant, low for a small constant.

## Poisson Probability

Poisson probability is a discrete probability distribution defined below.

$$
\begin{align}
p(X=x)={\lambda^xe^{-\lambda}\over x!},\quad x\ge0
\end{align}
$$

Both the mean and variance of a Poisson are $$\lambda$$. TBC.

### Relationship with Binomial Distribution

The binomial distribution tends to be the Poisson distribution as $$n\rightarrow\infty,u\rightarrow0$$ and $$nu$$ stays constant, in which we have $$\lambda=np$$.

## Appendix

### <a name="app1"></a>1. Properties of Gamma Function

The gamma function can be seen as the solution to the following interpolation problem:

> Find a smooth curve that connects the points $$(x,y)$$ given by $$y=(x-1)!$$

$$\Gamma(x)=(x-1)\Gamma(x-1)$$:

$$
\begin{align}
\Gamma(x)&=\int_0^\infty u^{x-1}e^{-u}du\\\
&=[-u^{x-1}e^{-u}]_0^\infty + \int_0^\infty ({x-1})u^{x-2}e^{-u}du\\\
&=\int_0^\infty ({x-1})u^{x-2}e^{-u}du\\\
&=({x-1})\Gamma(x-1)
\end{align}
$$

where the second step is held because of integration by parts

$$
\begin{align}
\int_a^bu(x)v'(x)dx=[u(x)v(x)]_a^b-\int_a^bu'(x)v(x)dx
\end{align}
$$

Also notice that $$\Gamma(1)=\int_0^\infty e^{-u}du=1$$. Therefore, for positive integer $$n$$, $$\Gamma(n)=(n-1)!$$ 

