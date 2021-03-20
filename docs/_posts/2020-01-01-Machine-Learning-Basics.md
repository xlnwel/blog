---
title: "Mathematics for Machine Learning — Calculus and Optimization"
excerpt: "Discussion on the basic principles of machine learning"
categories:
  - Mathematics
---

# Machine Learning Basics

## Capacity, overfitting and underfitting

The **representational capacity** of a model refers to the family of functions the model can learn in order to reduce the training error. In practice, a learning algorithm's **effective capacity** may be less than the representational capacity of the model family because, for example, the imperfection of the optimization algorithm.

Even an oracle that knowns the true probability distribution of the data will incur some error because there may be some noise in the distribution. In the case of supervised learning, for example, the mapping from $$\pmb x$$ to $$y$$ may be inherently stochastic, or $$y$$ may depend on some other variables besides those included in $$\pmb x$$. We term such a error as the **Bayesian error**

#### The no free lunch theorem

The **no free lunch theorem** for machine learning states that, *averaged over all* possible data generating distributions, every classification algorithm has the same error rate when classifying previously unobserved data. This means that we cannot produce a universal machine learning algorithm. Fortunately, in practice, we can still build a good learning algorithm based on the understanding of what kind of distributions are relevant to our task.

## Estimators, bias and variance

**Point estimation** is the attempt to provide the single "best" prediction of some quantity of interest. In general, the quantity of interest can be the parameters of a model, or alternatively, the whole function a model represents. In order to distinguish the point estimation from the true quantity, we denote the estimate of a quantity $$\pmb \theta$$ by $$\hat {\pmb\theta}$$.

Let $$\{\pmb x^{(1)}, \dots,\pmb x^{(m)} \}$$ be a set of $$m$$ i.i.d. data points. A **point estimator** or **statistic** is any *function* of the data

$$
\begin{align}
\hat{\pmb\theta}_m=g(\pmb x^{(1)}, \dots,\pmb x^{(m)})
\end{align}
$$


### Bias

The bias of an estimator is defined as

$$
\begin{align}
\text{bias}(\hat{\pmb\theta}_m)=\mathbb E[\hat{\pmb\theta}_m] - \pmb \theta
\end{align}
$$

An estimator $$\hat{\pmb\theta}_m$$ is unbiased if $$\text{bias}(\hat{\pmb\theta}_m)=\pmb 0$$. An estimator $$\hat{\pmb\theta}_m$$ is asymptotically unbiased if $$\lim_{m \rightarrow\infty}\text{bias}(\hat{\pmb\theta}_m)=\pmb 0$$.

#### Estimators of sample variance of a Gaussian distribution with unknown mean

We show that the sample variance defined below is biased

$$
\begin{align}
\hat\sigma_m^2={1\over m}\sum_{i=1}^m\left(x^{(i)}-\hat\mu_m\right)^2
\end{align}
$$

We evaluate the term $$\mathbb E[\hat\sigma_m^2]$$:


$$
\begin{align}
\mathbb E[\hat\sigma_m^2]=&\mathbb E\left[{1\over m}\sum_{i=1}^m\left(x^{(i)}-\hat\mu_m\right)^2\right]\\\
=&\mathbb E\left[{1\over m}\sum_{i=1}^m\left(x^{(i)}-\mu+\mu-\hat\mu_m\right)^2\right]\\\
=&\mathbb E\left[{1\over m}\sum_{i=1}^m\left[\left(x^{(i)}-\mu\right)^2+2(x^{(i)}-\mu)(\mu-\hat\mu_m)+\left(\mu-\hat\mu_m\right)^2\right]\right]\\\
=&\mathbb E\left[\underbrace{ {1\over m}\sum_{i=1}^m\left(x^{(i)}-\mu\right)^2}_{=\sigma^2}
+2\underbrace{ {1\over m}\sum_{i=1}^m(x^{(i)}-\mu)}_{\hat\mu_m-\mu}(\mu-\hat\mu_m)+{1\over m}\sum_{i=1}^m\left(\mu-\hat\mu_m\right)^2\right]\\\
=&\sigma^2-\underbrace{\mathbb E\left[(\mu-\hat\mu_m)^2\right]}_{=\sigma_{\hat \mu_m}^2={1\over m}\sigma^2}\\\
=&{m-1\over m}\sigma^2
\end{align}
$$

Because $$\text{bias}(\hat\sigma_m^2)=\mathbb E[\hat\sigma_m^2]-\sigma^2=-{1\over m}\sigma^2$$, $$\hat\sigma_m^2$$ is biased. We can easily obtain an **unbiased sample variance estimator** from the above evaluation 

$$
\begin{align}
\tilde\sigma_m^2={1\over m-1}\sum_{i=1}^m\left(x^{(i)}-\hat\mu_m\right)^2
\end{align}
$$


### Variance

The variance or the standard error of an estimator measures how the estimate vary as we sample the dataset from the underlying data generating process.

We often estimate the *generalization error* by computing the sample mean of the error on the test set. Taking advantage of the central limit theorem, which tells us that the sample mean will be approximately distributed with a normal distribution, we can use the standard error to compute the probability that the true expectation falls in any chosen interval. Specifically, we can derive the standard normal distribution $$Z=\lim_{m \rightarrow \infty}{\hat\mu_m-\mu\over{1\over \sqrt{m}}\sigma}$$ from the sample mean. For example, the $$95\%$$ confidence interval centered on the mean $$\hat\mu_m$$ is

$$
\begin{align}
(\hat\mu_m-1.96{\sigma\over\sqrt m},\hat\mu_m+1.96{\sigma\over\sqrt m})
\end{align}
$$


### Trading off bias and variance to minimize mean squared error

We show that the mean squared error trade off the bias and variance:

$$
\begin{align}
\text{MSE}=&\mathbb E[(\hat\theta_m-\theta)^2]\\\
=&\mathbb E[\hat\theta_m^2]-2\mathbb E[\hat\theta_m]\theta+\theta^2\\\
=&\underbrace{\mathbb E[\hat\theta_m^2]-\mathbb E[\hat \theta_m]^2}_{\text{Var}(\hat\theta_m)}+\underbrace{\mathbb E[\hat \theta_m]^2-2\mathbb E[\hat\theta_m]\theta+\theta^2}_{\text{Bias}(\hat\theta_m)^2}\\\
=&\text{Var}(\hat\theta_m)+\text{Bias}(\hat\theta_m)^2
\end{align}
$$


### Consistency

An estimator $$\hat\theta_m$$ of parameter $$\theta$$ is said to be **(weak) consistency** if it **converges in probability** to the true value of the parameter

$$
\begin{align}
\text{plim}_{m \rightarrow \infty}\hat\theta_m=\theta
\end{align}
$$

where $$\text{plim}$$ indicates converges in probability, meaning that for any $$\epsilon>0$$, $$P(\vert \hat\theta_m-\theta\vert >\epsilon)\rightarrow0$$ as $$m \rightarrow \infty$$.

The **strong consistency**, which is often referred to as **almost sure convergence**, says that the sequence $$\hat\theta_m$$ converges almost sure to $$\theta$$, meaning 

$$
\begin{align}
P(\lim_{m \rightarrow\infty}\hat\theta_m=\theta)=1
\end{align}
$$

The difference between the weak consistency and almost sure convergence lies in that the latter guarantees, after some point, $$\hat\theta_m$$ is a perfect accurate estimate of $$\theta$$, and there is no tolerate of failure afterwards, while the former only requires that $$\hat\theta_m$$ becomes less and less likely to be inaccurate as $$m$$ grows.

#### Bias versus consistency

Convergency ensures that the bias introduced by the estimator diminishes as the number of data points grows—in other word, convergency guarantees asymptotic unbiasedness. However, there is no direct implication between convergency and bias. 

**Unbiased but not consistent.** Consider i.i.d. samples $$x^{(1)}, x^{(2)}, \dots$$ from the standard normal distribution. One can use $$\hat\theta_m=x^{(m)}$$  as an unbiased estimator of the mean. It's unbiased because the sampling distribution is the same as the underlying distribution, i.e., $$\mathbb E[\hat\theta_m]=0$$. However, it does not converge to any value.

**Biased but consistent.** Consider the sample variance $$\hat\sigma_m^2={1\over m}\sum_{i=1}^m\left(x^{(i)}-\hat\mu_m\right)^2$$. It's an biased estimator as we discussed before, but it converges to the true variance as $$m$$ approaches infinity. 

### Maximum likelihood

Maximum likelihood(ML) is often considered the preferred estimator to use for machine learning because 1) it's consistent 2) when we use a parametric model in the regression case, the mean squared error decreases as $$m$$ increases, and for large $$m$$, the Cramér-Rao lower bound shows that no consistent estimator has a lower means squared error than the maximum likelihood estimator.

### Bayesian statistics

Relative to maximum likelihood estimation, Bayesian estimation offers two important difference:

1. The maximum likelihood approach makes predictions using a point estimate of $$\pmb \theta$$ while the Bayesian approach makes predictions using a full distribution over $$\pmb \theta$$.
2. The Bayesian approach introduces the Bayesian prior distribution, which shifts probability mass density towards regions of the parameter space that are preferred *a priori*.

Bayesian methods typically generalize much better when limited data is available, but typically suffer from high computational cost when the number of training examples is large.

Maximum a posteriori(MAP) simplifies Bayesian statistics by choosing the point of maximal posterior probability as an estimate of $$\pmb \theta$$.

Our [previous post]({{ site.baseurl }}{% post_url 2019-01-07-statistic-learning %}) provides a more detailed discussion on ML, MAP, and Bayesian learning.

## Challenges motivating deep learning

We list challenges that motivate deep learning over machine learning

- The **curse of dimensionality**. The number of possible configurations of our input $$\pmb x$$ increases exponentially as the dimensionality of $$\pmb x$$ increases. A statistical challenges arises because the number of possible configurations of $$\pmb x$$ is much larger than the number of training samples. In that case, there is not data for some configurations of $$\pmb x$$, making it hard to say something meaningful about these unseen configurations. Many traditional machine learning algorithms simply assume that the output at a new point should be simply approximately the same as the output at the nearest training points.
- **Smoothness prior** or **local constancy prior**. Machine learning algorithms need to be guided by prior beliefs about what kind of function they should learn in order generalize well. Among the most widely used of these implicit "prior" is the smoothness prior or local constancy prior, which states that the function we learn should not change very much within a small region. Mathematically, the local constancy prior encourages the learning process to learn a function $$f^*$$ that satisfies the condition $$f^*(\pmb x)\approx f^*(\pmb x+\epsilon)$$ for most $$\pmb x$$ and small change $$\epsilon$$. However, such a local prior alone is generally insufficient because it takes $$O(n)$$ data to learn $$O(n)$$ distinct regions. A more favorable prior, which is also the core idea in deep learning, is assuming the data was generated by the *composition of factors or features*. This prior presents a hierarchical view of the input features, allowing an exponential gain in the relationship with the number of examples and the number of regions that can be distinguished, helping counter the exponential challenges posed by the curse of dimensionality
- **Manifold learning.** A manifold is a connected region. Mathematically, it is a set of points, associated with a neighborhood around each point. Manifold learning tackles high-dimensional data by assuming that most of the data consists of invalid input and that interesting inputs occur only along a collection of manifolds containing a small subset of points, with interesting variations in the output of the learned function occurs only along directions that lie on that manifold, or with interesting variations happening only when we move from one manifold to another. In other words, manifold learning assumes we can extract from the input data a high-level representation of lower dimensions that captures the most interesting variations in the input. In that case, each feature/a subset of features in the representation associates with a manifold.
  - There are two observations that support the manifold hypothesis: 1) the probability distributions over images, text strings, and sounds that occur in real life is highly concentrated; 2) we can imagine such neighborhoods and transformations, at least informally. For example, we can trace out a manifold in image space associated with the light, color, etc.

## References

Ian, Goodfellow, Bengio Yoshua, and Courville Aaron. 2016. *Deep Learning*. MIT Press.