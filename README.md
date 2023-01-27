
### Fatigue Repository
## Neural networks for fatigue crack propagation predictions in real-time under uncertainty  
V. Giannella , F. Bardozzo , A. Postiglione , R. Tagliaferri , R. Sepe and R. Citarella


> Crack propagation analyses are fundamental for all mechanical structures for
which safety must be guaranteed, e.g. as for the aviation and aerospace fields. The estimation
of life for structures in presence of defects is a process inevitably affected by numerous and
unavoidable uncertainty and variability sources, whose effects need to be quantified to avoid
unexpected failures.
In this work, residual fatigue life prediction models have been created through neural
networks for the purpose of performing probabilistic life predictions of damaged structures in
real-time and under stochastically varying input parameters. In detail, five different neural
network architectures have been compared in terms of accuracy, computational runtimes and
minimum number of samples needed for training, so to determine the ideal architecture with
the strongest generalization power. The networks have been trained, validated and tested by
using the fatigue life predictions computed by means of simulations developed with finite
element and Monte Carlo methods. A real-world case study has been presented to show how
the proposed approach can deliver accurate life predictions even when input data are
uncertain and highly variable. 
Results demonstrated that the 'H1-L1' neural network has been the best model, achieving an
accuracy (Mean Square Error) of 4.8e-7 on the test dataset, and the best and the most stable
results when decreasing the amount of data., while using the lowest number of parameters,
thus highlighting its potential applicability for structural health monitoring purposes.




<p align="center">
  <img width="800" height="450" src="https://github.com/lodeguns/Fatigue/blob/main/fig2.png?raw=true">
</p>




This repository contains the manuscript mentioned at this [link] and associated code and data sets used for benchmarking
our predictive methodology based on neural networks.

<p align="center">
  <img width="800" height="450" src="https://github.com/lodeguns/Fatigue/blob/main/fig5.png?raw=true">
</p>


In this repository are provided the code and the data to train, test and validate our neural networks models. 
The code is developed in Python with Keras backend Tensorflow 2.10. 

The model wrapper for the optimizators should be installed from [this repository](https://github.com/fabiodimarco/tf-levenberg-marquardt). 

Should you need help running our code, please contact us. If you use this code in your research work you must cite us.


**How to cite this paper**

```
@article{Giannella2023neural,
title = {Neural networks for fatigue crack propagation predictions in real-time under uncertainty.},
journal = {-},
pages = {-},
year = {2023},
issn = {-},
doi = {-},
url = {-},
author = {V. Giannella, F. Bardozzo , A. Postiglione, R. Tagliaferri, R. Sepe and R. Citarella}
}
```

**Licence**
The same of the Journal.

**Corresponding author: ** vgiannella at unisa dot it

This work is supported by the Departements of Industrial Engineering and DISA-MIS - NeuRoNe Lab of the University of Salerno - Via Giovanni Paolo II, 132, Fisciano (SA), Italy 
