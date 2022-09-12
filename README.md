
Conditioning Normalizing Flows for Rare Event Sampling
===========

Code for the paper on conditioning normalizing flows for the generation of configurations in a parallel path sampling scheme.
The publication can be found here:

Falkner, S., Coretti, A., Romano, S., Geissler, P., & Dellago, C. (2022). Conditioning Normalizing Flows for Rare Event Sampling (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2207.14530 

Further Reading:
===================

* Noé, F., Olsson, S., Köhler, J., & Wu, H. (2019). Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. Science, 365(6457), eaaw1147. https://doi.org/10.1126/science.aaw1147

* Ardizzone, L., Lüth, C., Kruse, J., Rother, C., & Köthe, U. (2019). Guided Image Generation with Conditional Invertible Neural Networks. ArXiv. http://arxiv.org/abs/1907.02392

* Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using Real NVP. 5th International Conference on Learning Representations, ICLR 2017 - Conference Track Proceedings. http://arxiv.org/abs/1605.08803


Python Distribution
===================

Installation is recommended using a scientific Python distribution 
such as anaconda (https://www.anaconda.com).

Prerequisites
=============

1) Python >= 3.6
https://www.python.org

2) PyTorch
https://pytorch.org/

3) Numba
http://numba.pydata.org/

4) NumPy
https://www.numpy.org/

5) SciPy
https://www.scipy.org/

6) PyYAML
https://pyyaml.org/

7) matplotlib
https://matplotlib.org/

For the examples you will also need:

8) Jupyter
https://jupyter.org/

For free energy calculations PyEMMA is required:
8) PyEMMA
http://emma-project.org/latest/


Installation
============

Head to the root folder of the project and type ``pip install .``.
The installation script checks automatically for the packages mentioned above except for pytorch and the example dependencies.

PyTorch needs to be installed manually. Instructions can be found at https://pytorch.org/.

