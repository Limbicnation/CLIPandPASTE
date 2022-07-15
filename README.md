# **CLIP and PASTE: Using AI to Create Photo Collages from Text Prompts**
## How to use ML models to extract objects from photographs and rearrange them to create modern art

![CLIPandPadte Cover Image](https://raw.githubusercontent.com/robgon-art/CLIPandPASTE/main/cover%20shot%20mid.jpg)

**By Robert. A Gonsalves**</br>

You can see my article on [Medium](https://towardsdatascience.com/clip-and-paste-using-ai-to-create-modern-collages-from-text-prompts-38de46652827).

The source code and generated images are released under the [CC BY-SA license](https://creativecommons.org/licenses/by-sa/4.0/).</br>
![CC BYC-SA](https://licensebuttons.net/l/by-sa/3.0/88x31.png)

## Google Colabs
* [CLIP and PASTE](https://colab.research.google.com/github/robgon-art/CLIPandPASTE/blob/main/CLIP_and_PASTE.ipynb)

## Acknowledgements
- CLIP by A. Radford et al., Learning Transferable Visual Models From Natural Language Supervision (2021)
- F. Boudin, PKE: An Open-Source Python-based Keyphrase Extraction Toolkit (2016), Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: System Demonstrations
- Wikimedia Commons (2004-present)
- OpenImages (2020)
- GRoIE by L. Rossi, A. Karimi, and A. Prati, A Novel Region of Interest Extraction Layer for Instance Segmentation (2020)
- D. P. Kingma and J. Lei Ba, Adam: A Method for Stochastic Optimization (2015), The International Conference on Learning Representations 2015
- M. Grootendorst, KeyBERT: Minimal keyword extraction with BERT (2020)
- E. Riba, D. Mishkin, D. Ponsa, E. Rublee, and G. Bradski, Kornia: an Open Source Differentiable Computer Vision Library for PyTorch (2020), Winter Conference on Applications of Computer Vision

# Installation for local runs

Prerequirements:
make sure you have [CUDA](https://developer.nvidia.com/cuda-downloads) and [Aanaconda](https://www.anaconda.com/products/distribution) installed. This code runs on a Ubuntu 20.04 system with an NVIDIA Geforce graphics card.

Download:
```
https://github.com/Limbicnation/CLIPandPASTE.git
cd CLIPandPASTE
```

## Conda Virtual Environment
```
conda create --name CLIPandPASTE python=3.9
conda activate CLIPandPASTE
```
Pip install torch and torchvision
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
```
pip install torch
pip install torchvision
```

Install notebook

```
conda install -c conda-forge notebook
```
Install Jupyter

```
pip install jupyter
```

I don't remember what this command was for, but I got an error when I didn't use it.

```
pip install --upgrade --no-cache-dir gdown
```

## Citation
To cite this repository:

```bibtex
@software{CLIP and PASTE,
  author  = {Gonsalves, Robert A.},
  title   = {CLIP and PASTE: Using AI to Create Photo Collages from Text Prompts},
  url     = {https://github.com/robgon-art/CLIPandPAST},
  year    = 2022,
  month   = June
}
```
