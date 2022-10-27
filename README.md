# A Multifractal Theory of Loss Landscape-Facilitated Learning in Deep Neural Networks

The `multifractal-learning` module provides the code for the Honours thesis, "A Multifractal Theory of Loss Landscape-Facilitated Learning in Deep Neural Networks". This includes the simulation of gradient descent on a multifractal-like landscape (or any 2D landscape), the characterisation of the emergent chaotic dynamics and anomalous diffusive dynamics, and all associated visualisations in the thesis.

![MBS](doc/images/MBS.pdf)

![trajectory](doc/images/trajectory.pdf)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Simulation and Plots

All code for simulations and plots to reproduce the figures of the thesis are included in Jupyter notebooks. Text is included within the notebooks to guide the user. The notebooks are ordered according to the chapters of the thesis: `review_figures.ipynb`, `model_figures.ipynb`, `theory_figures.ipynb` and `comparison_figures.ipynb`.

## References

The [FracLab toolbox on MATLAB](https://project.inria.fr/fraclab/) was used for the generation of fractal- and multifractal-like 2D landscapes, specifically fractional and multifractional Brownian surfaces:

[1] Véhel, J. L., & Legrand, P. (2004). Signal and Image processing with FracLab. Thinking in Patterns, 321-322.

The [loss-landscape module](https://github.com/tomgoldstein/loss-landscape) was used for visualising the loss landscape:

[2] Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Visualizing the loss landscape of neural nets. Advances in neural information processing systems, 31.

The [PLBMF toolbox on MATLAB](https://www.irit.fr/~Herwig.Wendt/software.html) was used for multifractal analysis:

[3] Leonarduzzi, R., Wendt, H., Abry, P., Jaffard, S., Melot, C., Roux, S. G., & Torres, M. E. (2016). p-exponent and p-leaders, Part II: Multifractal analysis. Relations to detrended fluctuation analysis. Physica A: Statistical Mechanics and its Applications, 448, 319-339.

[4] Wendt, H., Roux, S. G., Jaffard, S., & Abry, P. (2009). Wavelet leaders and bootstrap for multifractal analysis of images. Signal Processing, 89(6), 1100-1114.

[5] Wendt, H., Abry, P., & Jaffard, S. (2007). Bootstrap for empirical multifractal analysis. IEEE signal processing magazine, 24(4), 38-48.

The [Anomalous-diffusion-dynamics-of-SGD module](https://github.com/ifgovh/Anomalous-diffusion-dynamics-of-SGD) was used for tracking the training process of a DNN:

[6] Chen, G., Qu, C. K., & Gong, P. (2022). Anomalous diffusion dynamics of learning in deep neural networks. Neural Networks, 149, 18-28.

The [pytorch-hessian-eigenthings module](https://github.com/noahgolmant/pytorch-hessian-eigenthings) was used for calculating the Hessian eigenvalues at each epoch of training.

[7] Golmant, N., Yao, Z., Gholami, A., Mahoney, M., Gonzalez, J. (2018). pytorch-hessian-eigenthings: efficient PyTorch Hessian eigendecomposition. 