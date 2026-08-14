# Accuracy Bounds


## Installation

This project has been tested under **Python 3.7** on a Unix development environment.  

### 1. Clone the repository and create a virtual environment and change into the repository folder.
```
git clone https://github.com/nm19000/AccuracyBounds.git
cd AccuracyBounds
```

### 2. Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate
```
### 3. Install the project
#### 3.1 Lightweight Version (without PyTorch)
```
pip install -e .
```
#### 3.2 Full version (with PyTorch)
```
pip install -e .[torch]
```

## Main Theorem

The framework provides model-independent performance bounds for both regression and classification. For regression, the **average kernel size** measures the typical spread of target values associated with inputs that are indistinguishable under the chosen input metric. For any predictor, the empirical error is bounded below by half of this quantity, and the optimal error over continuous regressors lies within the interval
\[
\frac{1}{2}\operatorname{Kersize}(\mathcal{D}_{M'},p)_K
\;\leq\;
\inf_{\phi}\mathcal{L}(\mathcal{D}_{M'},\phi,p)
\;\leq\;
\operatorname{Kersize}(\mathcal{D}_{M'},p)_K.
\]
Thus, a nonzero average kernel size identifies a component of prediction error that cannot be removed solely by changing the model architecture or optimization procedure. For classification, using the normalized Hamming distance, the same quantity gives an upper bound on achievable accuracy:
\[
\operatorname{Acc}(\mathcal{D}_{M'},\phi)
\;\leq\;
1-\frac{1}{2}\operatorname{Kersize}(\mathcal{D}_{M'},1)_K.
\]
For continuous classifiers, the optimal accuracy is bounded between \(1-\operatorname{Kersize}\) and \(1-\tfrac{1}{2}\operatorname{Kersize}\).

## Feasible-Set Sampling

The bounds are estimated directly from data using feasible-set sampling. Given an anchor pair \((y_k,x_k)\), the algorithm collects targets \(y_{k,n}\) from samples whose inputs satisfy
\[
d_{\mathcal X}(x_{k,n},x_k)\leq \tau_{\mathcal X}(x_k)
\]
or whose targets satisfy
\[
d_{\mathcal Y}(y_{k,n},y_k)\leq \tau_{\mathcal Y}(y_k).
\]
These targets form an approximate feasible set \(F_{x_k}^{N(k)}\), representing plausible outputs for locally indistinguishable inputs. The average kernel size is then computed from pairwise target distances within the sampled feasible sets. Its value, and therefore the resulting accuracy bounds, depends on the selected input and target metrics, thresholds, number of anchors, and maximum feasible-set size. Smaller thresholds produce more local estimates, while larger thresholds provide more samples but may combine less comparable observations.
        

# Average Kernel Sizes - Computable Sharp Accuracy Bounds for Inverse Problems

The reconstruction of an unknown quantity from noisy measurements is a mathematical problem relevant in most applied sciences, for example, in medical imaging, radar inverse scattering, or astronomy. This underlying mathematical problem is often an ill-posed (non-linear) reconstruction problem, referred to as an ill-posed inverse problem. To tackle such problems, there exist a myriad of methods to design approximate inverse maps, ranging from optimization-based approaches, such as compressed sensing, over Bayesian approaches, to data-driven techniques such as deep learning. For all stable approximate inverse maps, there are accuracy limits that are strictly larger than zero for ill-posed inverse problems, due to the accuracy-stability tradeoff [Gottschling et al., SIAM Review, 67.1 (2025)] and [Colbrook et al., Proceedings of the National Academy of Sciences, 119.12 (2022)]. The variety of methods that aim to solve such problems begs for a unifying approach to help scientists choose the approximate inverse map that obtains this theoretical optimum. Up to now there do not exist computable accuracy bounds to this optimum that are applicable to all inverse problems. We provide computable sharp accuracy bounds to the reconstruction error of solution methods to inverse problems. The bounds are method-independent and purely depend on the dataset of signals, the forward model of the inverse problem, and the noise model. To facilitate the use in scientific applications, we provide an algorithmic framework and an accompanying software library to compute these accuracy bounds.

## Accuracy Bounds for Inverse Problems

Computation of worst-case and average kernel size for an inverse problem with noise of the form: 

$$
\text{recover } x \in \mathcal{M}_1 \subset \mathbb{C}^{d_1} \text{ given noisy measurements } y = F(x,e)\in \mathbb{C}^{d_2} \text{ of } x  \text{ and }  e \in \\mathcal{E}\subset \mathbb{C}^{d_3}.
$$

The lower bound to the average error of any approximate inverse map is the average kernel size. The definition of and algorithms for computing the average kernel size are to be found in the following [preprint](https://arxiv.org/abs/2510.10229).

The lower bound to the worst-case error of any approximate inverse map is the worst-case kernel size. The definition of the worst-case kernel size can be found in the following [preprint](https://arxiv.org/abs/2311.16898).

