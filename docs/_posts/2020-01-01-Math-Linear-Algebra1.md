---
title: "Mathematics for Machine Learning — Linear Algebra Part 1"
excerpt: "Discussion on linear algebra used in machine learning/deep learning"
categories:
  - Mathematics
---

# Linear Algebra Part 1

## Linear Algebra

There are two interpretations for $$\pmb A\pmb x=\pmb b$$:

1. As a linear map, $$\pmb A$$ projects $$\pmb x$$ in the space $$\mathbb R^n$$ onto a (sub)space of $$\mathbb R^m$$ denoted by the columns of $$\pmb A$$, which gives us the result vector $$\pmb b$$ in the column space of $$\pmb A$$.
2. We take the columns of $$\pmb A$$ as axes, then each element $$x_i$$ of $$\pmb x$$ describes the coordinate along the respective columns $$\pmb A_{:,i}$$. On the other hand, each element $$b_i$$ of $$\pmb b$$ describes the coordinate along the associated canonical axis $$\pmb e_i$$, which is zero everywhere except for one at the $$i^{th}$$ position.

## Vector spaces

A vector space $$V$$ must satisfy

- There exists an additive identity $$\pmb 0$$ in $$\pmb V$$ such that $$\pmb x+\pmb 0=\pmb x$$ for all $$\pmb x\in V$$
- For each $$\pmb x\in V$$, there exists an additive inverse $$-\pmb x$$ such that $$\pmb x +(-\pmb x) = 0$$
- There exists a multiplicative identity $$1$$ in $$\mathbb R$$ such that $$1\pmb x=\pmb x$$
- Commutative: $$\pmb x+\pmb y=\pmb y+\pmb x$$
- Associative: $$(\pmb x + \pmb y)+\pmb z=\pmb x + (\pmb y + \pmb z)$$ and $$\alpha(\beta\pmb x)=\alpha\beta\pmb x$$
- Distributivity: $$\alpha(\pmb x+\pmb y)=\alpha\pmb x +\alpha\pmb y$$ and $$(\alpha+\beta)\pmb x=\alpha\pmb x+\beta\pmb y$$

The **span** of $$\pmb v_1,\dots,\pmb v_n\in V$$ is the set of all vectors that can be expressed as a linear combination of them, i.e., $$\text{span}(\pmb v_1,\dots,\pmb v_n)=\{\pmb v\in V:\exist\alpha_1,\dots,\alpha_n\text{ such that }\alpha_1\pmb v_1+\alpha_n\pmb v_n=\pmb v\}$$. These vectors are said to be a **basis** for $$V$$ if they are linearly independent.

#### Subspace

$$S\subseteq V$$ is a **subspace** of $$V$$ if

- $$\pmb 0\in S$$
- $$S$$ is closed under addition: $$\pmb x,\pmb y\in S\implies \pmb x+\pmb y\in S$$
- $$S$$ is closed under scalar multiplication: $$\pmb x\in S,\alpha\in\mathbb R\implies \alpha\pmb x\in S$$

If $$U$$ and $$V$$ are subspaces of $$V$$, the sum $$U+V$$ is also a subspace of $$V$$. If $$U\cap W=\{\pmb 0\}$$, the sum is said to be a direct sum and written $$U\oplus W$$. Every vector in $$U\oplus W$$ can be written uniquely as $$\pmb u+\pmb w$$ for some $$\pmb u\in U$$ and $$\pmb w\in W$$. (This is both a necessary and sufficient condition for a direct sum.)

The dimensions of sums of subspaces obey a friendly relationship

$$
\begin{align}
\text{dim}(U+W)=\text{dim}U+\text{dim}W-\text{dim}(U\cap W)
\end{align}
$$


## Linear maps

A **linear map** is a function $$T:V\rightarrow W$$, where $$V$$ and $$W$$ are vector spaces, that satisfies

- $$T(\pmb x+\pmb y)=T(\pmb x)+T(\pmb y)$$
- $$T(\alpha\pmb x)=\alpha T(\pmb x)$$ 

It's common to drop unnecessary parentheses, writing $$T\pmb x$$ rather than $$T(\pmb x)$$ if there is no risk of ambiguity, and denote denote composition $$ST$$ rather than $$S\circ T$$.

A linear map is called a **homomorphism **of *vector space* as it's structure-preserving, i.e., it preserves addition and scalar multiplication. An invertible homomorphism is called an **isomorphism**. If $$V$$ and $$W$$ are **isomorphic**, we write $$V\cong W$$. It is an interesting fact that finite-dimensional vector spaces of the same dimension are always isomorphic; if $$V, W$$ are real vector spaces with $$\text{dim} W=\text{dim}V=n$$, then we have the natural isomorphism $$\varphi:V\rightarrow W$$.

### Nullspace, range

If $$T:V\rightarrow W$$ is a linear map, we define the **nullspace** of $$T$$ as

$$
\begin{align}
\text{null}(T)=\{\pmb v\in V|T\pmb v=\pmb 0\}
\end{align}
$$

and the **range** of $$T$$ as

$$
\begin{align}
\text{range}(T)=\{\pmb w\in W|\pmb v\in V\text{ such that }T\pmb v=\pmb w\}
\end{align}
$$

Define $$T\pmb v= \pmb A\pmb v$$, where $$\pmb A\in \mathbb R^{m \times n}$$. Because $$T\pmb v=v_1\pmb A_{:,1}+\dots+v_m\pmb A_{:,m}$$, the **column space** of $$\pmb A$$ is also the **range** of $$T$$, which we denote by $$\text{range}(\pmb A)$$. Similarly, the **row space** of $$\pmb A$$ is denoted by $$\text{range}(A^\top)$$

Note the nullspace of $$\pmb A$$ is orthogonal to the range of the row space of $$\pmb A$$ not the column space.

It is a remarkable fact that the dimension of the column space of $$\pmb A$$ is the same as the dimension of the row space of $$\pmb A$$. This quantity is called the **rank** of $$\pmb A$$, and defined as

$$
\begin{align}
\text{rank}(\pmb A)=\dim\text{range}(\pmb A)=\dim\text{range}(\pmb A^T)
\end{align}
$$


## Metric spaces

Metrics generalize the notion of distance from Euclidean space.

A **metric** on a set $$S$$ is a function $$d:S\times S\rightarrow R$$ that satisfies

- Non-negativity: $$d(x,y)\ge 0$$, with equality if and only if $$x=y$$
- Commutativity: $$d(x,y)=d(y,x)$$
- Triangle inequality: $$d(x,y)\le d(x,z)+d(z,y)$$

### Normed spaces

Norms generalize the notion of length from Euclidean space

A **norm** on a real vector space $$V$$ is a function $$\Vert\cdot\Vert:V\rightarrow \mathbb R$$ that satisfies

- Non-negativity: $$\Vert \pmb x\Vert\ge 0$$, with equality if and only if $$\pmb x=\pmb 0$$
- Absolute homogeneous: $$\Vert\alpha\pmb x\Vert=\vert \alpha\vert \Vert\pmb x\Vert$$
- Triangle inequality: $$\Vert\pmb x+\pmb y\Vert\le \Vert\pmb x\Vert+\Vert\pmb y\Vert$$

A vector space endowed with a norm is called a **normed vector space**, or simply **norm space**.

Any norm on $$V$$ induces a distance metric on $$V$$

$$
\begin{align}
d(\pmb x,\pmb y)=\Vert\pmb x,\pmb y\Vert
\end{align}
$$

Therefore, any normed space is also a metric space.

We are typically concerned with a few specific norms on $$\mathbb R^n$$

$$
\begin{align}
\Vert \pmb x\Vert_p=&\left(\sum_{i=1}^nx_i^p\right)^{1\over p},\quad where\ p\ge 1\\\
\Vert \pmb x\Vert_\infty=&\max_{1\le i\le n} |x_i|
\end{align}
$$


### Inner product spaces

An **inner product** on a *real* vector space $$V$$ is a function $$\langle\cdot,\cdot\rangle:V\times V\rightarrow \mathbb  R$$ satisfying

- Non-negativity: $$\langle\pmb x,\pmb x\rangle\ge 0$$, with equality if and only if $$\pmb x=\pmb 0$$
- Linearity in the first slot: $$\langle\pmb x+\pmb y,\pmb z\rangle=\langle\pmb x,\pmb z\rangle+\langle\pmb y,\pmb z\rangle$$ and $$\langle\alpha\pmb x,\pmb y\rangle=\alpha\langle\pmb x,\pmb y\rangle$$
- Conjugate symmetry: $$\langle\pmb x,\pmb y\rangle=\langle\pmb y,\pmb x\rangle$$

The standard inner product on $$\mathbb R^n$$(aka., dot product) is given by

$$
\begin{align}
\langle\pmb x,\pmb y\rangle=\sum_{i=1}^nx_iy_i=\pmb x^\top\pmb y=\pmb x\cdot\pmb y
\end{align}
$$

Every inner product gives rise to a norm, called the canonical or induced norm, where the norm of a vector $$\pmb x$$ is denoted by

$$
\begin{align}
\Vert\pmb x\Vert=\sqrt{\langle\pmb x,\pmb x\rangle}
\end{align}
$$

**Pythagorean Theorem.** If $$\pmb x\perp\pmb y$$, i.e., $$\langle \pmb x,\pmb y\rangle=0$$, then

$$
\begin{align}
\Vert\pmb x+\pmb y\Vert^2=\Vert\pmb x\Vert^2+\Vert\pmb y\Vert^2
\end{align}
$$

**Cauchy-Schwarz inequality.** For all $$\pmb x, \pmb y\in V$$

$$
\begin{align}
|\langle\pmb x,\pmb y\rangle|\le \Vert\pmb x\Vert\Vert\pmb y\Vert
\end{align}
$$

where the equality holds when $$\pmb x$$ and $$\pmb y$$ are linear dependent.

**Proof.** Let $$\pmb z$$ be the projection of $$\pmb y$$ on to the plane orthogonal to $$\pmb x$$, i.e., $$\pmb z=\pmb y-{\langle\pmb y,\pmb x\rangle\over\langle\pmb x,\pmb x\rangle}\pmb x$$. By Pythagorean theorem, we have

$$
\begin{align}
\Vert\pmb y\Vert^2=&\left\Vert {\langle\pmb y,\pmb x\rangle\over\langle\pmb x,\pmb x\rangle}\pmb x\right\Vert^2+\Vert\pmb z\Vert^2\\\
=&\langle{\langle\pmb y,\pmb x\rangle\over\langle\pmb x,\pmb x\rangle}\pmb x, {\langle\pmb y,\pmb x\rangle\over\langle\pmb x,\pmb x\rangle}\pmb x\rangle+\Vert\pmb z\Vert^2\\\
=&{|\langle\pmb y,\pmb x\rangle|^2\over \Vert\pmb x\Vert^2}\Vert\pmb x\Vert+\Vert\pmb z\Vert^2\\\
\ge&{|\langle\pmb y,\pmb x\rangle|^2\over \Vert\pmb x\Vert}\\\
\Longleftrightarrow\quad \Vert\pmb y\Vert^2\Vert\pmb x\Vert^2=&|\langle\pmb y,\pmb x\rangle|^2\\\
\Vert\pmb y\Vert\Vert\pmb x\Vert=&|\langle\pmb y,\pmb x\rangle|
\end{align}
$$

**Orthogonal complements.** If $$S\subseteq V$$, where $$V$$ is an inner product space, then the **orthogonal complement** of $$S$$, denote $$S^\perp$$, is the set of all vectors in $$V$$ that are orthogonal to every element of $$S$$

$$
\begin{align}
S^\perp=\{\pmb v\in V|\pmb v\perp\pmb s\text{ for all }\pmb s\in S\}
\end{align}
$$

It is easy to verify that $$S^\perp$$ is a subspace of $$V$$ for any $$S\subseteq V$$ and furthermore $$\pmb v=\pmb v_S+\pmb v_\perp$$ and $$V=S\oplus S^\perp$$.

**Orthogonal projection.** The project of $$V$$ on $$S$$ is

$$
\begin{align}
P_S:V\rightarrow S
\end{align}
$$

**Proposition.** Let $$S$$ be a finite dimensional subspace of $$V$$. Then

1. For any $$\pmb v\in V$$ and orthogonal basis $$\pmb u_1,\dots,\pmb u_m$$ of $$S$$, the projection of $$\pmb v$$ onto $$S$$ is defined as
   
$$
   P_S\pmb v=\langle\pmb v,\pmb u_1\rangle\pmb u_1+\cdots+\langle\pmb v,\pmb u_m\rangle\pmb u_m
   $$


2. For any $$\pmb v$$, $$\pmb v-P_S\pmb v\perp S$$

3. $$P_S$$ is a linear map

4. $$P_S$$ is the identity when restricted to $$S$$ (i.e., $$P_S\pmb s=\pmb s$$ for all $$\pmb s\in S$$)

5. $$\text{range}(P_S)=S$$ and $$\text{null}(P_S)=S^\perp$$

6. $$P_S^2=P_S$$

7. $$P_S^\top=P_S$$

8. For any $$\pmb v\in V$$, $$\Vert P_S\pmb v\Vert\le \Vert \pmb v\Vert$$

9. For any $$\pmb v\in V$$ and $$\pmb s\in S$$
   
$$
   \Vert\pmb v- P_S\pmb v\Vert\le\Vert\pmb v-\pmb s\Vert
   $$

   With equality if and only if $$\pmb s=P_S\pmb v$$. That is
   
$$
   P_S\pmb v=\min_{\pmb s\in S}\Vert \pmb v-\pmb s\Vert
   $$


Consider a special case where $$S$$ is a subspace of $$\mathbb R^n$$ with orthonormal basis $$\pmb u_1,\dots\pmb u_m$$, then

$$
\begin{align}
P_S\pmb v=\sum_{i=1}^n\pmb u_i^\top\pmb v\pmb u_i=\sum_{i=1}^n\pmb u_i\pmb u_i^\top\pmb v=\left(\sum_{i=1}^n\pmb u_i\pmb u_i^\top\right)\pmb v
\end{align}
$$

Therefore the operator $$P_S$$ can be expressed as a matrix

$$
\begin{align}
P_S=\sum_{i=1}^n\pmb u_i\pmb u_i^\top=\pmb U\pmb U^\top
\end{align}
$$

where $$\pmb U$$ has $$\pmb u_1,\dots,\pmb u_n$$ as its columns. 

### Gram-Schmidt process

**Objective.** construct an orthonormal basis from linearly independent vectors $$\pmb v=\{\pmb v_1,\dots,\pmb v_n\}$$.

$$
\begin{align}
&\pmb e_1={\pmb v_1\over\Vert \pmb v_1\Vert_2}\\\
&\quad\mathbf{For}\ i=2,\dots,n\ \mathbf{do}\\\
&\qquad \pmb u_i=\pmb v_i-\sum_{j=1}^{i-1}\langle \pmb v_i,\pmb e_j\rangle\pmb v_i\\\
&\qquad \pmb e_i={\pmb u_i\over\Vert\pmb u_i\Vert_2}\\\
&\pmb E=\{\pmb e_1,\dots,\pmb e_n\}\text{ is an orthonormal basis}
\end{align}
$$


## References

Thomas, Garrett. 2018. “Mathematics for Machine Learning” 56 (5): 1–47.