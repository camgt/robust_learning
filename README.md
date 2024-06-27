# README

This repository contains code for robustification in training via regularization and by using a system of particles with Wasserstein ascend-descend dynamics (WAD) as described in the papers "On the regularized risk of distributionally robust learning over deep neural networks" [DOI:10.1007/s40687-022-00349-9](https://doi.org/10.1007/s40687-022-00349-9) and "On adversarial robustness and the use of Wasserstein ascent-descent dynamics to enforce it" [arXiv:2301.03662](https://arxiv.org/abs/2301.03662).


- Robustification with regularization
    + code: /Robust_nn/ord1L.py
    + tests: Tests_o1o2.ipynb
- Robustification WAD:
    + code: Robust_nn/WAD.py
    + tests: Tests_WAD_script.py

Note: As stated in https://pytorch.org/docs/stable/notes/randomness.html reproducibility of results is not guaranteed; However, results should be relatively close in between runs.


### Citations:
Please cite the corresponding paper.

### Contact details
Camilo Garcia Trillos (UCL)

[email](camilo.garcia@ucl.ac.uk) | [website](https://profiles.ucl.ac.uk/45384-camilo-garcia-trillos/about)



Improvements and comments are welcome!
