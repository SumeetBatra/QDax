[![Documentation Status](https://readthedocs.org/projects/qdax/badge/?version=latest)](https://qdax.readthedocs.io/en/latest/?badge=latest)
![pytest](https://github.com/adaptive-intelligent-robotics/QDax/actions/workflows/ci.yaml/badge.svg?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/adaptive-intelligent-robotics/QDax/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/adaptive-intelligent-robotics/QDax/branch/feat/add-codecov/graph/badge.svg)](https://codecov.io/gh/adaptive-intelligent-robotics/QDax)


# QDax: Accelerated Quality-Diversity
[Fork](https://github.com/adaptive-intelligent-robotics/QDax) of the QDax Library.
- QDax [paper](https://arxiv.org/abs/2202.01258)
- QDax [documentation](https://qdax.readthedocs.io/en/latest/)


## Installation
```bash
conda env create -f environment.yaml
```
Tested on Ubuntu 16.04 with Cuda 11.8 runtime and nvidia Driver Version: 520.56.06

If you have trouble creating the conda environment b/c of GPU or driver differences, see the [jax documentation](https://github.com/google/jax#installation)
under "pip installation: GPU (CUDA)" 

To install without GPU, create a conda environment from scratch and 
```python
pip install requirements.txt
```
To get the basic python packages installed. Then install nvidia-cuda-toolkit, cuda, and drivers appropriate to your system. 


## Running PGA-ME
### Basic Usage 
```python
python3 scripts/train_pga_me.py --env_name=walker2d_uni --seed=42 --num_iterations=100
```

A full list of configurable command line args can be found under ```scripts/train_pga_me.py``` or by calling
```python
python3 scripts/train_pga_me.py --help
```


## QDax core algorithms
QDax currently supports the following algorithms:

| Algorithm  | Example |
| --- | --- |
| [MAP-Elites](https://arxiv.org/abs/1504.04909) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/notebooks/mapelites_example.ipynb) |
| [CVT MAP-Elites](https://arxiv.org/abs/1610.05729) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/notebooks/mapelites_example.ipynb) |
| [Policy Gradient Assisted MAP-Elites (PGA-ME)](https://hal.archives-ouvertes.fr/hal-03135723v2/file/PGA_MAP_Elites_GECCO.pdf) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/notebooks/pgame_example.ipynb) |
| [OMG-MEGA](https://arxiv.org/abs/2106.03894) |  [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/notebooks/omgmega_example.ipynb) |
| [CMA-MEGA](https://arxiv.org/abs/2106.03894) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/notebooks/cmamega_example.ipynb) |
| [Multi-Objective Quality-Diversity (MOME)](https://arxiv.org/abs/2202.03057) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/notebooks/mome_example.ipynb) |


## QDax baseline algorithms
The QDax library also provides implementations for some useful baseline algorithms:

| Algorithm  | Example |
| --- | --- |
| [DIAYN](https://arxiv.org/abs/1802.06070) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/notebooks/diayn_example.ipynb) |
| [DADS](https://arxiv.org/abs/1907.01657) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/notebooks/dads_example.ipynb) |
| [SMERL](https://arxiv.org/abs/2010.14484) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/notebooks/smerl_example.ipynb) |
| [NSGA2](https://ieeexplore.ieee.org/document/996017) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/notebooks/nsga2_spea2_example.ipynb) |
| [SPEA2](https://www.semanticscholar.org/paper/SPEA2%3A-Improving-the-strength-pareto-evolutionary-Zitzler-Laumanns/b13724cb54ae4171916f3f969d304b9e9752a57f) | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adaptive-intelligent-robotics/QDax/blob/main/notebooks/nsga2_spea2_example.ipynb) |

## Contributing
Issues and contributions are welcome. Please refer to the [contribution guide](https://qdax.readthedocs.io/en/latest/guides/CONTRIBUTING/) in the documentation for more details.

## Related Projects
- [EvoJAX: Hardware-Accelerated Neuroevolution](https://github.com/google/evojax). EvoJAX is a scalable, general purpose, hardware-accelerated neuroevolution toolkit. [Paper](https://arxiv.org/abs/2202.05008)
- [evosax: JAX-Based Evolution Strategies](https://github.com/RobertTLange/evosax)

## Citing QDax
If you use QDax in your research and want to cite it in your work, please use:
```
@article{lim2022accelerated,
  title={Accelerated Quality-Diversity for Robotics through Massive Parallelism},
  author={Lim, Bryan and Allard, Maxime and Grillotti, Luca and Cully, Antoine},
  journal={arXiv preprint arXiv:2202.01258},
  year={2022}
}
```

## Contributors

QDax was developed and is maintained by the [Adaptive & Intelligent Robotics Lab (AIRL)](https://www.imperial.ac.uk/adaptive-intelligent-robotics/) and [InstaDeep](https://www.instadeep.com/).

<img align="center" src="docs/images/AIRL_logo.png" alt="AIRL_Logo" width="220"/> <img align="center" src="docs/images/instadeep_logo.png" alt="InstaDeep_Logo" width="220"/>

<a href="https://github.com/limbryan" title="Bryan Lim"><img src="https://github.com/limbryan.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/maxiallard" title="Maxime Allard"><img src="https://github.com/maxiallard.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/Lookatator" title="Luca Grilloti"><img src="https://github.com/Lookatator.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/manon-but-yes" title="Manon Flageat"><img src="https://github.com/manon-but-yes.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/Aneoshun" title="Antoine Cully"><img src="https://github.com/Aneoshun.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/felixchalumeau" title="Felix Chalumeau"><img src="https://github.com/felixchalumeau.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/ranzenTom" title="Thomas Pierrot"><img src="https://github.com/ranzenTom.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/Egiob" title="Raphael Boige"><img src="https://github.com/Egiob.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/valentinmace" title="Valentin Mace"><img src="https://github.com/valentinmace.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/GRichard513" title="Guillaume Richard"><img src="https://github.com/GRichard513.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/flajolet" title="Arthur Flajolet"><img src="https://github.com/flajolet.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/remidebette" title="Rémi Debette"><img src="https://github.com/remidebette.png" height="auto" width="50" style="border-radius:50%"></a>
