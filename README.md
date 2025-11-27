# Constrained Consensus-Based Optimization and Numerical Heuristics for the Low Particle Regime: Example code

This repository contains the experimental source code to reproduce the numerical experiments in:

* J. Beddrich, E. Chenchene, M. Fornasier, H. Huang, B. Wohlmuth. Constrained Consensus-Based Optimization and Numerical Heuristics for the Low Particle Regime. 2024. [ArXiv preprint](https://arxiv.org/abs/2410.10361)

To reproduce the results of the numerical experiments in Sections 5.1 and 5.2, run:
```bash
python3 rastrigin/main.py
```

To reproduce the results of the numerical experiments in Section 5.3, run the following commands. Be aware that these simulations might require significant time and memory (5.6GB). 
```bash
pip install cbx 
mkdir p-allen-cahn/data 
mkdir p-allen-cahn/images 
python3 p-allen-cahn/main.py
```

If you find this code useful, please cite the above-mentioned paper:
```BibTeX
@article{bcfhw24,
  author = {Beddrich, Jonas and Chenchene, Enis and Fornasier, Massimo and Huang, Hui and Wohlmuth, Barbara},
  title = {Constrained Consensus-Based Optimization and Numerical Heuristics for the Low Particle Regime},
  pages = {2410.10361},
  journal = {ArXiv},
  year = {2024}
}
```

## Requirements

Please make sure to have the following Python modules installed, most of which should be standard.

* [numpy>=1.20.1](https://pypi.org/project/numpy/)
* [matplotlib>=3.3.4](https://pypi.org/project/matplotlib/)
* [tqdm>=4.66.1](https://pypi.org/project/tqdm/)
* [cbx>=Commit7b29285](https://github.com/PdIPS/CBXpy/tree/main)

## Acknowledgments  


* The Department of Mathematics and Scientific Computing at the University of Graz, with which H.H. is affiliated, is a member of NAWI Graz (https://nawigraz.at/en).   
## License  
This project is licensed under the GPLv3 license - see [LICENSE](LICENSE) for details.
