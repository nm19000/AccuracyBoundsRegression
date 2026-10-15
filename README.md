# Deterministic Bounds on Accuracy and Stability of Metric-Space Regression and Classification
This repository contains code and example notebooks for the *Deterministic Bounds on Accuracy and Stability of Metric-Space Regression and Classification* as introduced in [this preprint](add_link_here_when_online) .

<img width="800" alt="Performance Bounds Overview" src="https://github.com/user-attachments/assets/c2e9258d-faf5-4360-a971-236c5fcd1821" />


The repository further contains all computations needed to reproduce the results of this paper. 
Additionally, a specific framework is currently under preparation. 

If you use the code and find it useful, please cite the paper. 

```bibtex
@article{gottschling2026deterministic,
  author    = {Gottschling, Nina Maria and Gawlikowski, Jakob and Caprio, Michele},
  title     = {Deterministic Bounds on Accuracy and Stability of Metric-Space Regression and Classification},
  year      = {2026},
  journal   = {arXiv preprint arXiv:TBA}
}
```

## Installation

This project has been tested under **Python 3.7** on a Unix development environment.  

### 1. Clone the repository and create a virtual environment and change into the repository folder.
```
git clone https://github.com/nm19000/AccuracyBoundsRegression.git
cd AccuracyBoundsRegression
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

## Example Applications
The repository comes with various [example notebooks](https://github.com/nm19000/AccuracyBoundsRegression/tree/main/examples) for regression and classification tasks. For MNIST and EuroSAT the folders also contain training scripts for training with label noise. 

**Regression Examples**

<img width="693" alt="regression examples" src="https://github.com/user-attachments/assets/19a19e53-1fac-4014-8c4a-97abeec92aef" />

**Classification Examples**

<img width="673" alt="classification examples" src="https://github.com/user-attachments/assets/8d50fc2d-1ac5-44ca-8eef-667c4de3004c" />

## Contributions 
In the following the main contributions of these implementations and the underlying paper are summarized. 

## Main Theorem: Accuracy Bounds based on Average Kernel Size
The framework provides model-independent accuracy bounds for both regression and classification. Please refer to the paper for a comprehensive introduction of the framework and definitions. For regression, the **average kernel size** measures the typical spread of target values associated with inputs that are indistinguishable under the chosen input metric. For any predictor, the empirical error is bounded below by half of this quantity, and the optimal error over continuous regressors lies within the interval

```math
\frac{1}{2}\mathop{\mathrm{Kersize}}(\mathcal{D}_{M'},p)_K\;\leq\;\inf_{\phi}\mathcal{L}(\mathcal{D}_{M'},\phi,p)\;\leq\;\mathop{\mathrm{Kersize}}(\mathcal{D}_{M'},p)_K.
```

Thus, a nonzero average kernel size identifies a component of prediction error that cannot be removed solely by changing the model architecture or optimization procedure. For classification, using the normalized Hamming distance, the same quantity gives an upper bound on achievable accuracy:

```math
\mathop{\mathrm{Acc}}(\mathcal{D}_{M'},\phi)\;\leq\;1-\frac{1}{2}\mathop{\mathrm{Kersize}}(\mathcal{D}_{M'},1)_K.
```

For continuous classifiers, the optimal accuracy is bounded between $1-\mathop{\mathrm{Kersize}}$ and $1-\tfrac{1}{2}\mathop{\mathrm{Kersize}}$.

## Feasible-Set Sampling

The bounds are estimated directly from data using feasible-set sampling. Given an anchor pair $(y_k,x_k)$, the algorithm collects targets $y_{k,n}$ from samples whose inputs satisfy

```math
d_{\mathcal X}(x_{k,n},x_k)\leq \tau_{\mathcal X}(x_k)
```

or whose targets satisfy

```math
d_{\mathcal Y}(y_{k,n},y_k)\leq \tau_{\mathcal Y}(y_k).
```

These targets form an approximate feasible set $F_{x_k}^{N(k)}$, representing plausible outputs for locally indistinguishable inputs. The average kernel size is then computed from pairwise target distances within the sampled feasible sets. Its value, and therefore the resulting accuracy bounds, depends on the selected input and target metrics, thresholds, number of anchors, and maximum feasible-set size. Smaller thresholds produce more local estimates, while larger thresholds provide more samples but may combine less comparable observations.

## Accuracy Bounds for Inverse Problems

Computation of worst-case and average kernel size for an inverse problem with noise of the form: 

```math
\text{recover } x \in \mathcal{M}_1 \subset \mathbb{C}^{d_1} \text{ given noisy measurements } y = F(x,e)\in \mathbb{C}^{d_2} \text{ of } x  \text{ and }  e \in \\mathcal{E}\subset \mathbb{C}^{d_3}.
```

The lower bound to the average error of any approximate inverse map is the average kernel size. The definition of and algorithms for computing the average kernel size are to be found in the following [preprint](https://arxiv.org/abs/2510.10229).

The lower bound to the worst-case error of any approximate inverse map is the worst-case kernel size. The definition of the worst-case kernel size can be found in the following [preprint](https://arxiv.org/abs/2311.16898).

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{gottschling2026deterministic,
  author    = {Gottschling, Nina Maria and Gawlikowski, Jakob and Caprio, Michele},
  title     = {Deterministic Bounds on Accuracy and Stability of Metric-Space Regression and Classification},
  year      = {2026},
  journal   = {arXiv preprint arXiv:TBA}
}
```
