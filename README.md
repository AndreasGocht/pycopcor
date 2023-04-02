# pyCopCor

pyCopCor is a framework which implements different copula-based correlation approaches. The framework is a result of my dissertation. If you use any of these methods, you are welcome to cite my [Dissertation](#dissertation-cite). While I create the empirical approach, the other approaches are based on other works, which you'll find in the [References](#references) section. However, I did spend some time parallelising and optimising some methods to modern CPUs supporting AVX2.

### Table of contents


* [Marginals](#marginals)
* [Variable Selection based on the Work of Schweizer and Wolff](#variable-selection-based-on-the-work-of-schweizer-and-wolff)
* [Copula Entropy](#copula-entropy)
  + [Embedding the copula entropy into mutual information theory](#embedding-the-copula-entropy-into-mutual-information-theory)
  + [Usage](#usage)
    - [Normalisation](#normalisation)
  + [Questions](#questions)
* [Dissertation (Cite)](#dissertation-cite)
* [References](#references)

## Marginals

To work with the Copula functions, you first need the marginals of your data. 

These can be computed for 1-D numpy arrays X, Y or Z using:

```python
import pycopcor.marginal as pcm
fx_0 = pcm.density(X[:])
fx_1 = pcm.density(Y[:])
fx_2 = pcm.density(Z[:])
```

## Variable Selection based on the Work of Schweizer and Wolff

One promising approach for dependence or correlation analysis is created by Schweizer and Wolff [[1]](#references) based on a dissertation of Wolff [[2]](#references). Using copulas, they described a set of measures for dependence:

* Spearman's $\rho = 12 \int_0^1{\int_0^1 {(C(u,v)-uv)du dv}}$
* $\gamma = (90 \int_0^1{\int_0^1 {(C(u,v)-uv)^2du dv}})^{\frac{1}{2}}$
* $\sigma = 12 \int_0^1{\int_0^1 {|C(u,v)-uv|du dv}}$

where $C(u,v)$ is the Copula of the marginals $u=F(x)$ and $v=G(y)$

Seth and Príncipe [[3]](#references) proposed $\gamma$ for variable selection, using _maximum relevance minimum redundancy_

$\rho$, $\sigma$ and $\gamma$ can be computed using:
```python
import pycopcor.copula.wolff as pcw
pcw.spearman(fx_0,fx_1)
pcw.sigma(fx_0,fx_1)
pcw.gamma(fx_0,fx_1)
```

## Copula Entropy

More recent work from Blumentritt and Schmid [[4]](#references) or from Ma and Sun [[5]](#references) showed that copula entropy is mutual information for continuous-valued variables. Two ways to calculate the copula entropy exist: Using histograms, as done in [[4,5]](#references) or using the empirical copula, as shown in my dissertation [[0, p. 41]](#dissertation-cite). Both approaches have benefits and drawbacks. Some of them are shown in the `notebooks` folder.

As the theory around the histogram approach is well covered, I focus on the empirical approach.
The copula entropy is based on the copula density, the derivate of the copula:

$c(u,v) = \frac{\partial^2 C(u,v)}{\partial u \partial v} $

However, the empirical copula $C$ is usually defined using the step function, which is not differentiable. The workaround I choose is to use the sigmoid function:

* $\sigma(x) = \frac{1}{1+e^{-x}}$
* $\frac{\partial \sigma(x)}{\partial (x)} = \sigma(x)\sigma(-x)$

So the empirical copula can be estimated with:

$C_{x,y}(u,v) = \frac{1}{n} \sum\limits_{i=1}^{n} \left( \sigma(-o \cdot (F_x(x_i) - u)) \sigma(-o \cdot (F_y(y_i) - v) \right)$,

which allows the computation of the derivate and the calculation of the copula entropy:

$I_{CD}(x,y) = \int_{[0,1]^2} \left( c_{x,y}(u,v) \log(c_{x,y}(u,v) \right) du dv$

### Embedding the copula entropy into mutual information theory

As also shown in my dissertation [[0, p. 39 ff]](#dissertation-cite), the copula entropy $I_{CD}(x,y,z)$ relates to the total correlation as defined in [[6]](#references). Therefore it can be associated with the joint mutual information [[7]](#references) and the historical joint mutual information [[8]](#references):

$I_{CD}(x,y,z) = TC(x,y,z) = I(x,y) + I(xy,z)$

$J_{HJMI,CD}(x_k,S) = J_{H,CD} + \frac{1}{|S|} \sum_{x_j \in S} \left(I_{CD}(x_j,x_k,y) - I_{CD}(x_j, x_k) - I_{CD}(x_j, y) \right)$

Please be aware that normalisation plays its role in this scenario. The normalisation to the numerical maximum seems to be most similar to the traditional mutual information, as shown in the notebook section. The results of my dissertation indicate the same. 

$I_{CD,NormMax}(x,y,z) = \frac{I_{CD}(x,y,z)}{I_{CD}(x,x,x)}$

$I_{CD,NormMax}(x,y) = \frac{I_{CD}(x,y)}{I_{CD}(x,x)}$

Please be aware that Blumentritt and Schmid [[4]](#references) suggested a different normalisation related to the gaussian copula, which has its own benefits:

$I_{CD,NormGauss}(x,y) = \sqrt{1 - e^{-2 \cdot I_{CD}(x,y)}}$ 

The more dimensional normalisation is shown in [[4]](#references).

### Usage

Compute the copula entropy:

```python
import pycopcor.copula.entropy as pce

# histogram based approaches
pce.histogram_2d(fx_0,fx_1)
pce.histogram_3d(fx_0,fx_1,fx_2)

# empirical approaches
pce.empirical_2d(fx_0,fx_1)
pce.empirical_3d(fx_0,fx_1,fx_2)
```

#### Normalisation
```python
import pycopcor.copula.entropy as pce
import pycopcor.marginal as pcm



# for 2d data:

## based on the numerical maximum, as in [0]
rng = numpy.random.default_rng()
ref = rng.normal(0,1,n)
f_ref = pcm.density(ref)

i_cd_ref_e = pce.empirical_2d(f_ref,f_ref)
norm_max_e = lambda x: x/i_cd_ref_e

i_cd_ref_h = pce.histogram_2d(f_ref,f_ref)
norm_max_h = lambda x: x/i_cd_ref_h

## based on gaussian copula, as in [4]
norm_gauss = lambda x: numpy.sqrt(1 - numpy.exp(-2*x))



# for 3d data

## based on the numerical maximum, as in [0]
rng = numpy.random.default_rng()
ref = rng.normal(0,1,n)
f_ref = pcm.density(ref)

i_cd_ref_e = pce.empirical_3d(f_ref,f_ref,f_ref)
norm_max_e = lambda x: x/i_cd_ref_e

i_cd_ref_h = pce.histogram_3d(f_ref,f_ref,f_ref)
norm_max_h = lambda x: x/i_cd_ref_h

## based on gaussian copula, as in [4]
def norm_gauss_3d(val):
    d_1 = 3 - 1
    def f(p): return - 1/2 * numpy.log((1-p)**d_1 * (1+d_1*p)) - val
    ret = scipy.optimize.root_scalar(f, bracket=(0, 1))
    if not ret.converged:
        raise RuntimeError("Not Converged")
    return ret.root
```

### Questions

Feel free to open a ticket in GitHub or email me if you have any questions.

## Dissertation (Cite)

[0] Gocht-Zech, Andreas   
Ein Framework zur Optimierung derEnergieeffizienz von HPC-Anwendungen auf der Basisvon Machine-Learning-Methoden (2022)  
Technische Universität Dresden, Technische Universität Dresden  
[`https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa2-819405`](https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa2-819405)  

## References

[1] Schweizer, B. ans Wolff, Edward. F.   
On Nonparametric Measures of Dependence for Random Variables (1981)  
The Annals of Statistics , Vol. 9, No. 4   
DOI: [10.1214/aos/1176345528](https://dx.doi.org/10.1214/aos/1176345528)  


[2] Wolff, Edward F. 
Measures of dependence derived from copulas (1977)  
Dissertation  
[https://search.proquest.com/docview/302846303](https://search.proquest.com/docview/302846303)  


[3] Seth, Sohan and Príncipe, José Carlos  
Variable Selection: A Statistical Dependence Perspective  
2010 Ninth International Conference on Machine Learning and Applications  
DOI: [10.1109/ICMLA.2010.148](https://dx.doi.org/10.1109/ICMLA.2010.148)  

[4] Blumentritt, Thomas and Schmid, Friedrich   
Mutual information as a measure of multivariate association: analytical properties and statistical estimation (2012)  
Journal of Statistical Computation and Simulation , Vol. 82, No. 9   
DOI: [10.1080/00949655.2011.575782](https://dx.doi.org/10.1080/00949655.2011.575782)  

[5] Ma, J. and Sun, Z.   
Mutual information is copula entropy (2011)  
Tsinghua Science and Technology , Vol. 16, No. 1   
DOI: [10.1016/S1007-0214(11)70008-6](https://dx.doi.org/10.1016/S1007-0214(11)70008-6)  

[6] Timme, Nicholas / Alford, Wesley / Flecker, Benjamin / Beggs, John M.   
Synergy, redundancy, and multivariate information measures: an experimentalist's perspective (2014)  
Journal of Computational Neuroscience , Vol. 36, No. 2   
DOI: [10.1007/s10827-013-0458-4](https://dx.doi.org/10.1007/s10827-013-0458-4)  

[7] Brown, Gavin and Pocock, Adam and Zhao, Ming-Jie and Luján, Mikel   
Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection (2012)  
Journal of Machine Learning Research , Vol. 13   
[https://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf](https://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf)

[8] Gocht, A. and Lehmann, C. and Schöne, R.   
A New Approach for Automated Feature Selection (2018)  
2018 IEEE International Conference on Big Data (Big Data)   
DOI: [10.1109/BigData.2018.8622548](https://dx.doi.org/10.1109/BigData.2018.8622548)  
