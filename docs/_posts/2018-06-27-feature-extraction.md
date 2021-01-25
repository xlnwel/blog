---
title: "Feature Extraction"
excerpt: "An introduction to Canny edge detection, a technique widely used in image processing to extract edges from images."
categories:
  - Machine Learning
tags:
  - Machine Learning
---

## Introduction

This post is talking about how to detect lines in an image, 

## <a name='dir'></a>Table of Contents

- [Canny Edge Detector](#ced)
- [Harris Corner Detector](#hcd)
- [Histogram of Oriented Gradients](#hog)

## Basic Operators

Before diving into complex feature extraction algorithms, it is helpful to know some basic operators commonly used in the preprocessing step of more complex algorithms. All filters introduced next are used to convolve with images in the same way filters in ConvNet do.

#### Low-pass Filter

Low-pass filters are used to preprocess image to reduce noise and unwanted traits in an image. The simplest example is


$$
  \eta
  \begin{vmatrix}
  1 & 1 & 1\\\
  1 & 1 & 1\\\
  1 & 1 & 1
  \end{vmatrix} 
  $$


where $$ \eta $$ is the normalization factor defined as the sum of filter. A more wildly used low-pass filter is the Gaussian filter, which is defined according to the Gaussian distribution


$$
  G(x,y)={1\over 2\pi\sigma^2}e^{-{(x-\mu_x)^2 + (y-\mu_y)^2 \over 2\sigma^2}}
$$


An example of $$ 3\times3 $$ Gaussian filter with $$ \sigma=1 $$ is 

$$
  \eta
  \begin{vmatrix}
  1 & e^{1\over 2} & 1\\\
  e^{1\over 2} & e & e^{1\over 2}\\\
  1 & e^{1\over 2} & 1
  \end{vmatrix} 
  $$


Heads up:
  - It is important to understand that the selection of the *size of the Gaussian kernel* will affect the performance of the detector. The larger the size is, the lower the detector’s sensitive to noise. Additionally, the localization error to detect the edge will slightly increase with the increase of the Gaussian filter kernel size. A \$$ 5\times 5 \$$ is a good size for most cases, but this will also vary depending on specific situations. 
- The same discussion works for the *standard deviation in the Gaussian kernel*, \$$ \sigma \$$: large \$$ \sigma \$$ suggests the weights of the surrounding pixels are relatively large and thus weaken the influence of the center pixel. (Also recall that, in Gaussian kernel of SVM, large \$$ \sigma \$$ means high bias. Both indicate that large \$$ \sigma \$$ causes insensitivity to noise, in a slightly different sense though)

#### High-pass Filter

High-pass filters detect big changes in intensity or color in an image and produce an output that shows these edges. Common choices include the Sobel operator and the Prewitt Operator.

The Sobel operator uses following two kernels to calculate approximations of the horizontal and vertical gradient

$$
 Sobel_x=\begin{vmatrix}-1 & 0 & 1\\\ -2 &0 &2 \\\ -1& 0&1\end{vmatrix}\ Sobel_y=\begin{vmatrix}-1 & -2 & -1\\\ 0 &0 &0\\\ 1& 2&1\end{vmatrix} 
$$

whereas the Prewitt Operator uses

$$
 Prewitt_x=\begin{vmatrix}-1 & 0 & 1\\\ -1 &0 &1\\\ -1& 0&1\end{vmatrix}\ Prewitt_y=\begin{vmatrix}-1 & -1 & -1\\\ 0 &0 &0\\\ 1& 1&1\end{vmatrix} 
$$


## <a name='ced'></a>Canny Edge Detector 

Canny edge detection is a technique widely used in image processing to extract *well-defined edges* from images.

### Algorithm

<figure>
  <img src="{{ '/images/cv/Canny.png' | absolute_url }}" alt="" width="1000">
  <figcaption></figcaption>
  <style>
    figure figcaption {
    text-align: center;
    }
  </style>
</figure>

Now we'll go through the process of Canny edge detection algorithm

#### i. Apply Gaussian filter to smooth the image in order to reduce the noise


#### ii. Find the intensity gradients of the image

The intensity gradients are calculated from the horizontal and vertical gradients $$ G_x $$ and $$ G_y $$, which could be approximated by high-pass filters introduced at the beginning. Given $$ G_x $$ and $$ G_y $$, the gradient magnitude is calculated by

$$
 G=\sqrt{G_x^2+G_y^2} 
$$

and the gradient's direction

$$
 \Theta=atan2(G_y, G_x) 
$$

What, you wanna know what $$ atan2 $$ is? Sorry, I'm gonna take a rain check :-)

#### iii. Apply non-maximum suppression to get rid of spurious response to the edge detection

*Non-maximum suppression*, an edge thinning technique. For each pixel in the gradient image, it works as follows:
1. Compare the edge strength (gradient magnitude) of the current pixel with the edge strength of the pixels in the positive and negative *gradient direction*
2. If the edge strength of the current pixel is the largest compared to the other two, the value will be preserved. Otherwise, the value will be suppressed (by setting it to \$$ 0 \$$)

#### iv. Apply double threshold to determine potential edges

1. Select high and low threshold values.
2. Mark pixels whose gradient value is higher than the high threshold value as strong edge pixels
3. Mark pixels whose gradient value is smaller than the high threshold value but larger than the low threshold value as weak edge pixels
4. Suppress pixels whose gradient value is smaller than the low threshold value (by setting it to \$$ 0 \$$)

#### v. Track edge by [hysteresis](https://en.wikipedia.org/wiki/Hysteresis)

Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges — that is, no strong edge pixel is around the weak edge pixel.

[Elevator back to directory](#dir)

## <a name='hcd'></a>Harris Coner Detector

### Theoretical basis

Considering taking a window (i.e. filter), $$ W $$, and shifting it by $$ (\Delta x, \Delta y) $$, the *weighted sum of squared differences* between the original and shfited window is given by

$$
S(x,y) = \sum_{x,y \in W}w(x,y)(I(x+\Delta x, y+\Delta y)-I(x,y))^2
 $$

where $$ w $$ is the windown function, which gives weights to pixels underneath and $$ I $$ is the intensity function. 

The window function is either a rectangle window, which assigns value $$ 1 $$ to all pixels in the window, otherwise $$ 0 $$, or a Gaussian window. For simplicity, we only consider the window function is a rectangle window for time being.

For a corner, shifting the window in any direction should yield a big variation in the direction and magnitude of the gradient. Therefore, we have to maximize $$ S(x,y) $$ to find a corner.

Note that $$ I(x+\Delta x, y+\Delta y) $$ can be approximated by Tayler expansion

$$
I(x+\Delta x, y+\Delta y)\approx I(x,y)+\Delta xI_x(x,y)+\Delta yI_y(x,y)
 $$

$$ S(x,y) $$ appriximates to

$$
S(x,y)\approx \sum_{x,y\in W}w(x,y)(\Delta xI_x+\Delta yI_y)^2
 $$

It can be shown that the above equation can be expressed in a matrix form as

$$
S(x,y)\approx \begin{bmatrix}\Delta x&\Delta y\end{bmatrix}M\begin{bmatrix} \Delta x\\\ \Delta y\end{bmatrix}
 $$

Where $$ M $$ is the *structure tensor*

$$
M=\sum_{x,y \in W}w(x,y)\begin{bmatrix}I_x^2 & I_xI_y\\\I_xI_y&I_y^2\end{bmatrix}
 $$

$$ M $$ has two eigenvalues: $$ \lambda_1, \lambda_2 $$. According to the magnitude of the eigenvalues, there are three cases

1. The region is flat: both are small 
2. The region contains an edge: one is small and the other is large
3. The region contains a corner: both are large

Mathematically, we ususally define a matrics function, *Harris response* $$ R $$ to determine if the region contains a corner

$$
\begin{align}
\det(M)&=\lambda_1 \lambda_2\\\
\mathrm{trace}(M)&=\lambda_1+\lambda_2\\\
R&=\det(M)-k(\mathrm {trace}(M))^2
\end{align}
 $$

where $$ k $$ is a tunable sensitivity parameter, usually in range of $$ [0.04, 0.06] $$. Now the cases become

1. Flat: $$ \vert R\vert  $$ is small
2. Edge: $$ R $$ is negative with large magnitude
3. Corner: $$ R $$ is large

There is a fact that I have little idea why the above cases hold. And I've done some researches, exception some intuition no rigorous explanation found. If someone can help me get it straight, please let me know, I'll be very grateful for that :-)

### Algorithm

#### i. Convet color to grayscale

#### ii. Calculate spatial derivative

The same step as the second step of Canny Edge detector

#### iii. Calculate the structure tensor $$ M $$ 

#### iv. Calculate Harris response $$ R $$

#### v. Thresholding

Suppress those regions with $$ R $$ smaller than a threshold and mark those with $$ R $$ greater as corners

[Elevator back to directory](#dir)

## <a name="hog"></a>Histogram of Oriented Gradients

### Algorithm

<figure>
  <img src="{{ '/images/cv/HOG.png' | absolute_url }}" alt="" width="1000">
  <figcaption></figcaption>
  <style>
    figure figcaption {
    text-align: center;
    }
  </style>
</figure>

#### i. Compute Gradient

Dalal and Triggs showed that 1-D centered, point discrete derivative mask, $$ [-1,0,1] $$ and $$ [-1,0,1]^T $$, outperforms other more complex kernels in face detection

#### ii. Create Histograms for Each Cell

This step groups pixels into cells, and then create an oriented-based histogram for each cell. The cells are rectangle, and the histogram bins are evenly spread over $$ 0 $$ to $$ 180 $$ for unsigned gradients or $$ 0 $$ to $$ 360 $$ for signed gradients. Dalal and Triggs found that unsigned gradients with $$ 9 $$ bins performed best in their human detection experiments. Each pixel interpolates weighted votes (gradient magnitude) linearly (or with Gaussian) into its neighboring bin centers based on its gradient angle.

#### iii. Group Cells into Blocks 

To account for local changes in illumination and contrast, the gradient magnitude must be locally normalized, which requires grouping the cells together into larger blocks. We do so by shifting a block window around the image just like the convolutional operation in CNN. Dalal and Triggs found in their experiment the optimal parameters are $$ 8\times8 $$ pixels per cell and $$ 2\times 2 $$ cells per block 

#### iv. Perform Block Normalization 

For each block, we concatenate histogram vectors from cells in it, and then perform normalization. There are four normalizations

1. L2-norm: $$ v = {v\over\sqrt{\left\vert v\right\vert _2^2+\epsilon}} $$
2. L2-hys: L2-norm followed by clipping (limiting the maximum value of $$ v $$ to $$ 0.2 $$) and renormalizing
3. L1-norm: $$ v = {v\over\left\vert v\right\vert _1+\epsilon} $$
4. L1-sqrt: $$ v = {\sqrt{v\over\left\vert v\right\vert _1+\epsilon}} $$

Dalal and Triggs found *1, 2, 4* provide similar performance, while *3* provides slightly less reliable performance. However, all four methods showed very significant improvement over the non-normalized data

#### v. Concatenate All Feature Vectors Forming HOG descriptor

HOG descriptors then could be used as features to some machine learning algorithm, such as SVMs and neural networks

[Elevator back to directory](#dir)