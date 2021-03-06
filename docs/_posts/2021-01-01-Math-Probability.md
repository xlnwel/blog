---
˜title: "Mathematics for Machine Learning — Probability"
excerpt: "in which we discuss probabilities used in machine learning/deep learning"
categories:
  - Mathematics
---

# Probability

## Basics

**Frequentist probability** refers to the rates at which events occur, while **Bayesian probability** is related to qualitative levels of certainty.

**Sample space** $$\Omega$$ is defined as a set of possible outcomes of a random experiment.	

A **Event** is the probability of a subset of $$\Omega$$. The set of events is denoted $$\mathcal F$$. The **complement** of the event $$A$$ is another event, $$A^c=\Omega\text\ A$$

A **probability measure** is a real value function $$\mathbb P(A):\mathcal F\rightarrow [0,1]$$ that must satisfies

- $$\mathbb P(\Omega)=1$$

- **Countable additivity:** for any countable collection of disjoint sets $$\{A_i\}\subseteq \mathcal F$$
  
$$
  \mathbb P(\bigcup_iA_i)=\sum_i\mathbb P(A_i)
  $$


The triple $$(\Omega,\mathcal F,\mathbb P)$$ is called a **Probability space**

If $$\mathbb P(A)=1$$, we say $$A$$ occurs **almost surely**, and conversely $$A$$ occurs **almost never** if $$\mathbb P(A)=0$$

The **Bayes's rule** can be written as

$$
\begin{align}
\mathbb P(A|B)\propto\mathbb P(A)\mathbb P(B|A)
\end{align}
$$

Under this formulation, $$\mathbb P(A)$$ is often referred to as the **prior**, $$\mathbb P(A\vert B)$$ as the **posterior**, and $$\mathbb P(B\vert A)$$ as the **likelihood**

In machine learning, we can use the Bayes' rule to update our beliefs (e.g., values of the model parameters) given some data that we've observed

## Expectation and variance

Expectation has some useful properties

- Linearity: $$\mathbb E\left[\sum_{i=1}^n\alpha X_i+\beta\right]=\alpha\sum_{i=1}^n\mathbb E[X_i]+\beta$$
- If all $$X_i$$ are independent, the product rule holds $$\mathbb E[\prod_{i=1}^nX_i]=\prod_{i=1}^n\mathbb E[X_i]$$

Variance also has some useful properties

- $$\text{Var}(\alpha X+\beta)=\alpha^2 \text{Var}(X)$$
- If all $$X_i$$ are uncorrelated, then $$\text{Var}(\sum_{i=1}^nX_i)=\sum_{i=1}^n\text{Var}(X_i)$$

Because the units of variance are not the same as the units of the random variable, we introduce the **standard deviation** as the square root of the variance.

## Covariance

Covariance is a measure of the linear relationship between two random variables. We define it as

$$
\begin{align}
\text{Cov}(X,Y)=\mathbb E[(X-\mathbb E[X])(Y-\mathbb E[Y])]
\end{align}
$$

The linearity of expectation allows us to rewrite it as

$$
\begin{align}
\text{Cov}(X,Y)=\mathbb E[XY]-\mathbb E[X]\mathbb E[Y]
\end{align}
$$

A useful property of covariance is that of **bilinearity**:

$$
\begin{align}
\text{Cov}(\alpha X+\beta Y,Z)=&\alpha\text{Cov}(X,Z)+\beta\text{Cov}(Y,Z)\\\
\text{Cov}(X,\alpha Y+\beta Z)=&\alpha\text{Cov}(X,Y)+\beta\text{Cov}(X,Z)\\\
\text{Cov}(\pmb A\pmb X+\pmb B)=&\pmb A\text{Cov}(\pmb X)\pmb A^\top
\end{align}
$$

**Correlation** is defined as the normalized

$$
\begin{align}
\rho(X,Y)={\text{Cov}(X,Y)\over\sqrt{\text{Var}(X)\text{Var}(Y)}}
\end{align}
$$

Correlation always lies between $$-1$$ and $$1$$.

Two variables are **uncorrelated** if $$\rho(X,Y)=\text{Cov}(X,Y)=0$$.

## Gaussian distribution

In the absence of prior knowledge about what form a distribution over the real numbers should take, the **Gaussian distribution** (a.k.a. the **normal distribution**) is a good default choice for two major reasons

- Many distributions we wish to model are truly close to being normal distributions. The **central limit theorem** shows that the sum/mean of many independent random variables is approximate normally distributed.
- Out of all possible probability distributions with the same variance, the normal distribution encodes the maximum amount of uncertainty(in other words, it has the maximum entropy) over the real numbers. 

The multivariate normal distribution is expressed as

$$
\begin{align}
\mathcal N(\pmb x;\pmb u,\pmb \Sigma)=\sqrt{ { 1\over\det(2\pi\pmb \Sigma) } } \exp((\pmb x-\pmb \mu)^\top\pmb \Sigma^{-1}(\pmb x-\pmb \mu))
\end{align}
$$

Since we need to invert the covariance matrix $$\pmb \Sigma$$ to evaluate the PDF, we can instead use a **precision matrix** $$\pmb \beta$$:

$$
\begin{align}
\mathcal N(\pmb x;\pmb u,\pmb \beta^{-1})=\sqrt{ {\det(\pmb \beta)\over(2\pi)^n} }\exp((\pmb x-\pmb \mu)^\top\pmb \beta(\pmb x-\pmb \mu))
\end{align}
$$

where $$n$$ is the dimension of the vector space. We often fix the covariance matrix to be a diagonal matrix.

## Exponential and Laplace distributions

If we want to have a probability distribution with a sharp point at $$x=0$$, we can use the exponential distribution:

$$
\begin{align}
p(x;\lambda)=\lambda\pmb 1_{x\ge0}\exp(-\lambda x)
\end{align}
$$

The **Laplace distribution** allows us to place a sharp peak of probability mass at an arbitrary point $$\mu$$

$$
\begin{align}
\text{Laplace}(x;\mu,\gamma)={1\over 2\gamma}\exp(-{|x-\mu|\over\gamma})
\end{align}
$$


## The Dirac distribution and empirical distribution

The Dirac delta distribution defines a PDF that puts all of the mass in around a single point:

$$
\begin{align}
p(x)=\delta(x-\mu)
\end{align}
$$

where $$\delta$$ is the Dirac delta function, which is zero-valued everywhere except $$0$$, yet integrates to $$1$$. The Dirac delta function is not an ordinary function that associates each value $$x$$ with a real-valued output, instead it is a different kind of mathematical object called a **generalized function** that is defined in terms of its properties when integrated.

A common use of the Dirac delta distribution is as a component of an **empirical distribution**,

$$
\begin{align}
\hat p(\pmb x)={1\over m}\sum_{i=1}^m \delta(\pmb x-\pmb x^{(i)})
\end{align}
$$

which puts probability mass $$1\over m$$ on each of the $$m$$ points $$\pmb x^{(1)},\dots,\pmb x^{(m)}$$ forming a given dataset. The empirical distribution is only necessary for continuous variables. For discrete variables, one can use the Multinoulli distribution directly. 

## Technique details of continuous variables

**Measure zero.** A set of points negligibly small is said to have measure zero. Intuitively, a set of measure zero occupies no *volume* in the space we are measuring. For example, any subset of $$\mathbb R^n$$ whose dimension is smaller than $$n$$ has measure zero. Moreover, any union of countably many sets that each has measure zero also has measure zero.

A property that holds **almost everywhere** holds throughout all of space except for on a set of measure zero.

For two random variables, $$\pmb x$$ and $$\pmb y$$, such that $$\pmb y=g(\pmb x)$$, where $$g$$ is an invertible, continuous, differentiable transformation. We don't have $$p_y(\pmb y)=p_x(g^{-1}(\pmb y))$$. Instead, we derive $$p_y(\pmb y)$$ from the property $$\vert p_y(\pmb y)d\pmb y\vert =\vert p_x(\pmb x)d\pmb x\vert $$:

$$
\begin{align}
|p_y(\pmb y)d\pmb y|=&|p_x(\pmb x)d\pmb x|\\\
p_y(\pmb y)=&p_x(\pmb x)\left |{d\pmb x\over d\pmb y}\right |\\\
=&p_x(\pmb x)\left |{d\pmb y\over d\pmb x}\right |^{-1}\\\
=&p_x(\pmb x)\left |\det Dg(\pmb x)\right |^{-1}\\\
\end{align}
$$

where $$\det Dg(\pmb y)$$ is the determinant of the partial derivative of $$g$$ at $$\pmb x$$, a.k.a., the [Jacobian determinant](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant#Jacobian_determinant). Intuitively, the absolute value of the Jacobian determinant gives the factor by which the function $$g$$ expands or shrinks volumes near $$\pmb x$$.

## Mixtures of distributions

A mixture of distributions is made up of several component distributions

$$
\begin{align}
P(x)=\sum_iP(c=i)P(x|c=i)
\end{align}
$$


## Structured probabilistic models

When we represent the factorization of a probability distribution with a graph, we call it a **structured probabilistic model** or **graphical model**. There are two main kinds of structured probabilistic models: directed and undirected. Note that they are only different in the description of probability distributions; any probability distribution may be described in both ways.

**Direct models** use graphs with directed edges, and they represent factorizations into conditional probability distributions. A direct model contains one factor for every random variable $$x_i$$ in the distribution, and that factor consists of the conditional distribution over $$x_i$$ given the parent of $$x_i$$ denoted $$Pa_{\mathcal G}(x_i)$$

$$
\begin{align}
P(\pmb x)=\prod_{i}p(x_i|Pa_{\mathcal G}(x_i))
\end{align}
$$

**Undirected models** use graph with undirected edges, and they represent factorizations into a set of functions; unlike in the directed case, these functions are usually *not probability distributions of any kind*. A **clique** is a subset of nodes, such that every two distinct nodes are adjacent. Each clip $$\mathcal C^{(i)}$$ is associated with a factor $$\phi^{(i)}(\mathcal C^{(i)})$$. The output of each factor must be non-negative, but there is no constraint that the factor must sum or integrate to $$1$$ like a probability distribution. 

## References

Thomas, Garrett. 2018. “Mathematics for Machine Learning” 56 (5): 1–47.

Ian, Goodfellow, Bengio Yoshua, and Courville Aaron. 2016. *Deep Learning*. MIT Press.