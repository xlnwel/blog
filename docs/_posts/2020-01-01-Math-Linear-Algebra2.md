---
title: "Mathematics for Machine Learning — Linear Algebra Part 2"
excerpt: "Discussion on linear algebra used in machine learning/deep learning"
categories:
  - Mathematics
---

# Linear Algebra Part 2

## Eigenthings

**Proposition.** The eigenvalues of $$\pmb A$$ can be computed by solving $$\vert \pmb A-\lambda \pmb I\vert =0$$.

**Proof.** By the definition of eigenvalues we have

$$
\begin{align}
\pmb A\pmb x=&\lambda\pmb x\\\
(\pmb A-\lambda\pmb I)\pmb x=&\pmb 0
\end{align}
$$

In order to have nontrivial solution $$\pmb x$$, the determinant must be zero, i.e., $$\vert \pmb A-\lambda\pmb I\vert =0$$; otherwise, there will be an invert matrix such that $$\pmb x=(\pmb A-\lambda\pmb I)^{-1}\pmb 0=\pmb 0$$.

**Proposition.** Let $$\pmb x$$ be an eigenvector of $$\pmb A$$ with corresponding eigenvalue $$\lambda$$. Then

1. For any $$\gamma\in\mathbb R$$, $$\pmb x$$ is an eigenvector of $$\pmb A+\gamma\pmb I$$ with eigenvalue $$\lambda+\gamma$$
2. If $$\pmb A$$ is invertible, then $$\pmb x$$ is an eigenvector of $$\pmb A^{-1}$$ with eigenvalue $$\lambda^{-1}$$
3. $$\pmb A^k \pmb x=\lambda^k\pmb x$$ for any $$k\in\mathbb Z$$

**Eigendecomposition.** Let $$\pmb A$$ be an $$n\times n$$ matrix with *linearly independent* eigenvectors $$\pmb q_1,\dots,\pmb q_n$$ and their eigenvalues $$\lambda_1,\dots,\lambda _n$$. we have

$$
\begin{align}
\pmb A=\pmb Q\pmb \Lambda\pmb Q^{-1}
\end{align}
$$

where $$\pmb Q$$ is the square matrix with $$\pmb q_1,\dots,\pmb q_n$$ as its columns, and $$\pmb \Lambda=\text{diag}(\lambda_1,\dots,\lambda _n)$$.

**Proof.** By the definition of eigenvalues and eigenvectors, we have

$$
\begin{align}
\pmb A\pmb Q=&\pmb Q\pmb \Lambda\\\
\pmb A=&\pmb Q\pmb\Lambda\pmb Q^{-1}
\end{align}
$$

A matrix $$\pmb A$$ with eigendecomposition is always [**diagonalizable**](https://en.wikipedia.org/wiki/Diagonalizable_matrix), and vice versa. $$\pmb A$$ is diagonalizable if there exists a diagonal matrix $$\pmb D$$ and an invertible matrix $$\pmb P$$ such that $$\pmb A=\pmb P\pmb D\pmb P^{-1}$$. We call such matrices **normal** matrices. Note that there is no connection between 1)$$\pmb A$$ is diagonalizable and 2)$$\pmb A$$ is invertible. See examples below

|            | Diagonalizable                         | Non-diagonalizable                     |
| ---------- | -------------------------------------- | -------------------------------------- |
| Invertible | $$\begin{bmatrix}1&0\\\0&1\end{bmatrix}$$ \vert  $$\begin{bmatrix}1&1\\\0&1\end{bmatrix}$$ |
| Singular   | $$\begin{bmatrix}0&0\\\0&0\end{bmatrix}$$ \vert  $$\begin{bmatrix}0&1\\\0&0\end{bmatrix}$$ |

One way to remember eigendecomposition is to regard $$\pmb A\pmb x$$ as a three-step transformation to $$\pmb x$$:

1. $$\pmb Q^{-1}$$ translates $$\pmb x$$ to the space denoted by the eigenvectors of $$\pmb A$$ (preserving the length if $$\pmb q_1,\dots\pmb q_n$$ are orthonormal)
2. $$\pmb \Lambda$$ then scales the result along the basis by the corresponding eigenvalues 
3. $$\pmb Q$$ translates the result back to the canonical space

**Proposition.** $$\pmb A$$ is an $$n\times n$$ matrix with $$n$$ linearly independent eigenvectors $$\pmb q_1,\dots,\pmb q_n$$ and their eigenvalues $$\lambda_1,\dots,\lambda _n$$, then we have $$\pmb A^m=\pmb Q\pmb \Lambda^m\pmb Q^{-1}$$, where $$\pmb Q$$ is a matrix with $$\pmb q_1,\dots,\pmb q_n$$ as its columns, and $$\pmb \Lambda=\text{diag}(\lambda_1,\dots,\lambda _n)$$. 

**Proof.** Because $$\pmb A=\pmb Q\pmb\Lambda\pmb Q^{-1}$$, we have

$$
\begin{align}
\pmb A^m=(\pmb Q\pmb\Lambda\pmb Q^{-1})^m=\pmb Q\pmb\Lambda^m\pmb Q^{-1}
\end{align}
$$


## Square Matrices

### Trace

The trace of a *square* matrix is the sum of its diagonal entries

$$
\begin{align}
tr(\pmb A)=\sum_{i=1}^n A_{ii}
\end{align}
$$

The trace has several nice algebraic properties

1. $$tr(\pmb A+\pmb B)=tr(\pmb A)+tr(\pmb B)$$
2. $$tr(\alpha\pmb A)=\alpha tr(\pmb A)$$
3. $$tr(\pmb A)=tr(\pmb A^\top)$$
4. Invariance under cyclic permutations. $$tr(\pmb A\pmb B\pmb C\pmb D)=tr(\pmb B\pmb C\pmb D\pmb A)=tr(\pmb C\pmb D\pmb A\pmb B)=tr(\pmb D\pmb A\pmb B\pmb C)$$
5. $$tr(\pmb A)=\sum_i\lambda_i(\pmb A)$$

### Determinant

The determinant of a *square* matrix $$\pmb A\in\mathbb R^{n\times n}$$ has several properties

1. $$\det(\pmb I)=1$$
2. $$\det(\pmb A)=\det(\pmb A^\top)$$
3. $$\det(\pmb A\pmb B)=\det(\pmb A)\det(\pmb B)$$
4. $$\det(\pmb A^{-1})=\det(\pmb A)^{-1}$$
5. $$\det(\alpha\pmb A)=\alpha^n\det(\pmb A)$$
6. $$\det(\pmb A)=\prod_i\lambda_i(\pmb A)$$

From 6, we have $$\det(\pmb A)=0$$ if $$\exist_i\lambda_i(\pmb A)=0$$. Because $$\det(\pmb A)=0$$, $$\det(\pmb A)^{-1}=0^{-1}$$ is not valid and there is no $$\det(\pmb A^{-1})$$ and $$\pmb A^{-1}$$.

### Orthogonal matrices

A matrix $$\pmb A\in\mathbb R^{n\times n}$$ is orthogonal if its *columns and rows* are pairwise *orthonormal*. This implies

$$
\begin{align}
\pmb A\pmb A^{\top}=\pmb A^\top \pmb A\pmb =\pmb I\Rightarrow \pmb A^\top=\pmb A^{-1}
\end{align}
$$

Orthogonal matrices preserve inner product and $$\ell_2$$ norm

$$
\begin{align}
\langle\pmb A\pmb x,\pmb A\pmb y\rangle=\pmb x^\top\pmb y\Rightarrow\Vert \pmb A\pmb x\Vert_2=\Vert\pmb x\Vert_2
\end{align}
$$

Therefore, multiplication by an orthogonal matrix can be considered as a transformation that preserve length, but may rotate or reflect the vector about the origin. We call such matrices **unitary matrices** and the corresponding linear map an **isometry**.

The determinant of an orthogonal matrix is either $$1$$ or $$-1$$. This can be derived as follows

$$
\begin{align}
1=\det(\pmb I)=\det(\pmb A\pmb A^\top)=\det(\pmb A)\det(\pmb A^\top)=\det(\pmb A)^2
\end{align}
$$


### Symmetric matrices

**Theorem.** (Spectral Theorem) If $$\pmb A\in\mathbb R^{n\times n}$$ is symmetric, then there exists an orthonormal basis for $$\mathbb R^n$$ consisting of eigenvectors of $$\pmb A$$

The spectral theorem implies that if $$\pmb A\in\mathbb R^{n\times n}$$ is symmetric, there exists an orthogonal matrix $$\pmb Q$$, whose columns&rows are orthonormal eigenvectors of $$\pmb A$$. This gives us

$$
\begin{align}
\pmb Q\pmb A=\pmb \Lambda\pmb Q\Rightarrow\pmb A=\pmb Q\pmb \Lambda\pmb Q^\top
\end{align}
$$

since $$\pmb Q^\top=\pmb Q^{-1}$$ for an orthogonal matrix $$\pmb Q$$. 

#### Rayleigh quotients

Let $$\pmb A\in\mathbb R^{n\times n}$$ be a *symmetric* matrix. The expression $$\pmb x^\top\pmb A\pmb x$$ is called the *quadratic form*.

The **Rayleigh quotient** is defined as

$$
\begin{align}
R_{\pmb A}(\pmb x)={\pmb x^\top\pmb A\pmb x\over\pmb x^\top\pmb x}
\end{align}
$$

It has a couple of important properties

1. **Scale invariance.** For any vector $$\pmb x\ne\pmb 0$$ and any scalar $$\alpha\ne 0$$, $$R_{\pmb A}(\pmb x)=R_{\pmb A}(\alpha\pmb x)$$
2. If $$\pmb x$$ is an eigenvector of $$\pmb A$$ with eigenvalue $$\lambda$$, then $$R_{\pmb A(\pmb x)}=\lambda$$
3. **Min-max theorem**. $$\lambda_\min\le R_{\pmb A(\pmb x)}\le\lambda_{\max}$$ with equality if and only if $$\pmb x$$ is a corresponding eigenvector.

**Proof of min-max theorem.** By the scale invariance of Rayleigh quotient, we can assume $$\Vert\pmb x\Vert_2=1$$ without loss of generality. Therefore, min-max theorem boils down to

$$
\begin{align}
\lambda_\min\le\pmb x^\top\pmb A\pmb x\le \lambda_\max
\end{align}
$$

for all unit $$\pmb x$$. Because $$\pmb A$$ is symmetric, we have

$$
\begin{align}
\pmb x^\top\pmb A\pmb x=&\pmb x^\top \pmb Q\pmb \Lambda\pmb Q^\top\pmb x\\\
&\qquad\color{red}{\text{let }\pmb y=\pmb Q^\top\pmb x}\\\
=&\pmb y^\top\pmb\Lambda\pmb y\\\
=&\sum_{i}\lambda_i y_i^2
\end{align}
$$

where $$\pmb Q$$ is an orthogonal matrix with orthonormal eigenvectors of $$\pmb A$$ as its columns. Because $$\Vert\pmb x\Vert_2=1$$ and $$\pmb Q$$ is an orthogonal matrix, we have $$\Vert\pmb y\Vert_2=1$$. Therefore the maximum of $$\sum_{i}\lambda_i y_i^2$$ is obtained when

$$
\begin{align}
y_i=\begin{cases}
1&\text{if }\lambda_i=\max_j\lambda_j\\\
0&\text{otherwise}
\end{cases}
\end{align}
$$

Since $$\pmb x=\pmb Q\pmb Q^\top\pmb x=\pmb Q\pmb y=\sum_iy_i\pmb q_i$$, we have 

$$
\begin{align}
\arg\max_{\pmb x}\pmb x^\top\pmb A\pmb x=\pmb q_i,\quad where\ i=\arg\max_j\lambda_j
\end{align}
$$

We can derive the minimum in an analogous way.

### Positive (semi-)definite matrices

A *symmetric matrix* $$\pmb A$$ is a positive semi-definite matrix if for all $$\pmb x\in \mathbb R^n$$, $$\pmb x^\top\pmb A\pmb x\ge 0$$

A *symmetric matrix* $$\pmb A$$ is a positive definite matrix if for all *nonzero* $$\pmb x\in \mathbb R^n$$, $$\pmb x^\top\pmb A\pmb x> 0$$

**Proposition.** A symmetric matrix is positive semi-definite if and only if its eigenvalues are nonnegative, and positive definite if and only if its eigenvalues are positive.

**Proposition.** Suppose $$\pmb A\in\mathbb R^{m \times n}$$. Then $$\pmb A^\top\pmb A$$ is positive semi-definite. If $$\text{null}(A)=\{\pmb 0\}$$, then $$\pmb A^\top\pmb A$$ is positive definite.

Positive definite matrices are always invertible, but positive semi-definite may not. The following proposition says that we can always make a positive semi-definite invertible by add a small number to its diagonal.

**Proposition.** If $$\pmb A$$ is positive semi-definite and $$\epsilon>0$$, then $$\pmb A+\epsilon \pmb I$$ is positive definite.

#### The geometry of positive definite quadratic forms

A **level set** or **isocontour** of a function is the set of all inputs such that the function applied to these inputs yields a given output. Mathematically the $$c$$-isocontour of $$f$$ is $$\{\pmb x\in\text{dom} f:f(x)=c\}$$.

Consider the special case $$f=\pmb x^\top\pmb A\pmb x$$ where $$\pmb A$$ is *positive definite matrix*. Because $$\pmb A$$ is positive definite, it has unique matrix square root $$\pmb A^{1/2}=\pmb Q\pmb \Lambda^{1/2}\pmb Q^\top$$, where $$\pmb Q\pmb \Lambda\pmb Q^\top$$ is the eigendecomposition of $$\pmb A$$ and $$\pmb \Lambda=\text{diag}(\sqrt{\lambda_1},\dots,\sqrt{\lambda_n})$$. For a $$c\ge 0$$, the $$c$$-isocontour of $$f$$ is the set of $$\pmb x\in\mathbb R^n$$ such that

$$
\begin{align}
c=\pmb x^\top\pmb A\pmb x=\pmb x^\top\pmb A^{1/2}\pmb A^{1/2}\pmb x=\Vert\pmb A^{1/2}\pmb x\Vert_2^2
\end{align}
$$

Making the change of variable $$\pmb z=\pmb A^{1/2}\pmb x$$, we have the condition $$\Vert \pmb z\Vert_2=\sqrt c$$. That is, the values $$\pmb z$$ lie on a sphere of radius $$\sqrt c$$. Then we parameterize $$\pmb z=\sqrt c\hat{\pmb z}$$ where $$\hat{\pmb z}$$ has $$\Vert\hat{\pmb z}\Vert_2=1$$ and have

$$
\begin{align}
\pmb x=\pmb A^{-1/2}\pmb z=\pmb Q\pmb \Lambda^{-1/2}\pmb Q^\top\sqrt c\hat{\pmb z}=\sqrt c\pmb Q\pmb \Lambda^{-1/2}\tilde{\pmb z}
\end{align}
$$

where $$\tilde{\pmb z}=\pmb Q^\top\hat {\pmb z}$$ also satisfies $$\Vert\tilde{\pmb z}\Vert_2=1$$ since $$\pmb Q^\top$$ is orthogonal. Using this parameterization, we see that the solution set $$\{\pmb x\in\mathbb R^n:f(\pmb x)=c\}$$ is the image of the unit space $$\{\tilde {\pmb z}\in\mathbb R^n:\Vert\tilde {\pmb z}\Vert_2=1\}$$ under the invertible linear map $$\pmb x=\sqrt c\pmb Q\pmb \Lambda^{-1/2}\tilde{\pmb z}$$. This gives us a clear algebraic understanding of the $$c$$-isocontour of $$f$$ in terms of a sequence of linear transformations applied to a unit sphere: 

- We starts with the unit sphere $$\tilde {\pmb z}$$, and scale every axis $$i$$ by $$\lambda_i^{-1/2}$$, resulting in an axis-aligned ellipsoid. Note that the axis lengths of the ellipsoid are proportional to the inverse square roots of the eigenvalues of $$\pmb A$$.
- Then this ellipsoid undergoes a rigid transformation(i.e., one that preserves length and angles, such as rotation and reflection) given by $$\pmb Q$$, changing the basis to eigenvectors of $$\pmb A$$.

## Singular value decomposition

Every matrix $$\pmb A\in\mathbb R^{m \times n}$$ has an **singular value decomposition** as follows

$$
\begin{align}
\pmb A=\pmb U\pmb \Sigma\pmb V^\top
\end{align}
$$

where 

- The columns of $$\pmb U$$ are the **left-singular eigenvectors** of $$\pmb A$$, i.e., *orthonormal eigenvectors* of $$\pmb A\pmb A^\top$$, 
- $$\pmb\Sigma$$ is a diagonal matrix whose diagonal entries are the **singular values** of $$\pmb A$$. i.e., square roots of the eigenvalues of both $$\pmb A\pmb A^\top$$ and $$\pmb A^\top\pmb A$$
- The columns of $$\pmb V$$ are the **right-singular eigenvectors** of $$\pmb A$$, i.e., *orthonormal eigenvectors* of $$\pmb A^\top\pmb A$$ .

### Comparison between singular value decomposition and eigenvalue decomposition

- The vectors in the eigendecomposition matrix $$\pmb Q$$ are not necessarily orthogonal, so the change of basis isn't a simple rotation. On the other hand, the vectors in the matrices $$\pmb U$$ and $$\pmb V$$ in the SVD are orthonormal, so they do represent rotations (and possibly flips).
- In the SVD, the matrices $$\pmb U$$ and $$\pmb V$$ are usually not related to each other at all. In the eigendecomposition the non-diagonal matrices $$\pmb Q$$ and $$\pmb Q^{-1}$$ are inverses of each other.
- In the SVD the entries in the diagonal matrix $$\pmb \Sigma$$ are all real and nonnegative. In the eigendecomposition, the entries of $$\pmb \Lambda$$ can be any complex number—negative, positive, imaginary, whatever.
- The SVD always exists for any sort of rectangular or square matrix, whereas the eigendecomposition can only exists for square matrices, and even among square matrices sometimes it doesn't exist.

## References

Thomas, Garrett. 2018. “Mathematics for Machine Learning” 56 (5): 1–47.

https://math.stackexchange.com/a/320232/401382