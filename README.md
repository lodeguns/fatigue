
# Fatigue
## Neural networks for fatigue crack propagation predictions in real-time under uncertainty  
V. Giannella 1* , F. Bardozzo 2 , A. Postiglione 2 , R. Tagliaferri 2 , R. Sepe 1 , R. Citarella


> Abstract: Crack propagation analyses are fundamental for all mechanical structures for
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
Results demonstrated that the “H1-L1” neural network has been the best model, achieving an
accuracy (Mean Square Error) of 4.8e-7 on the test dataset, and the best and the most stable
results when decreasing the amount of data., while using the lowest number of parameters,
thus highlighting its potential applicability for structural health monitoring purposes.


This repository contains the manuscript mentioned at this [link]
and associated code and data sets used for benchmarking
our predictive methodology based on neural networks.

Here the code to train, test and validation our neural networks is developed in 
Python with Keras backend Tensorflow 2.10. 
The model wrapper for the optimizators should be installed from this repository. 
Should you need help running our code, please contact us. 



<p align="center">
  <img width="600" height="700" src="https://github.com/lodeguns/StaSiS-Net/blob/main/imgs/gh_example.png?raw=true">
</p>


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
author = {V. Giannella, F. Bardozzo , A. Postiglione, R. Tagliaferri, R. Sepe and R. Citarellai}
}
```


**Licence**
The same of the Journal.
If you use this code you must cite this paper.































## Fatigue repository
1 Department of Industrial Engineering, University of Salerno, via Giovanni Paolo II, 132, Fisciano (SA), Italy 
2 Department of Management and Innovation Systems, University of Salerno, 
Via Giovanni Paolo II, 132, Fisciano (SA), Italy
* Correspondence: email: vgiannella@unisa.it;
