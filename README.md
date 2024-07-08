## [Block Coordinate Plug-and-Play Methods for Blind Inverse Problems (NeurIPS 2023)](https://openreview.net/forum?id=IyWpP2e0bF&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2023%2FConference%2FAuthors%23your-submissions))

[[openreview](https://openreview.net/forum?id=IyWpP2e0bF&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2023%2FConference%2FAuthors%23your-submissions))] [[project page](https://wustl-cig.github.io/bcpnpwww/)] [[arXiv](https://arxiv.org/abs/2305.12672)]

Plug-and-play (PnP) prior is a well-known class of methods for solving imaging inverse problems by computing fixed-points of operators combining physical measurement models and learned image denoisers. While PnP methods have been extensively used for image recovery with known measurement operators, there is little work on PnP for solving blind inverse problems. We address this gap by presenting a new block-coordinate PnP (BC-PnP) method that efficiently solves this joint estimation problem by introducing learned denoisers as priors on both the unknown image and the unknown measurement operator. We present a new convergence theory for BC-PnP compatible with blind inverse problems by considering nonconvex data-fidelity terms and expansive denoisers. Our theory analyzes the convergence of BC-PnP to a stationary point of an implicit function associated with an approximate minimum mean-squared error (MMSE) denoiser. We numerically validate our method on two blind inverse problems: automatic coil sensitivity estimation in magnetic resonance imaging (MRI) and blind image deblurring. Our results show that BC-PnP provides an efficient and principled framework for using denoisers as PnP priors for jointly estimating measurement operators and images.

## Setup the environment
We use Conda to set up a new environment by running 

```
conda env create --file "requirement.yml" -n bcpnp
```

and activate this new environment

```
conda activate bcpnp
```

## How to run the testing

### (1) Download the dataset and pre-trained denoiser from the following link and put them into the folders `./data` and `./network`

[https://bit.ly/41PPf6j](https://bit.ly/41PPf6j)

### (2) Run the BCPnP testing on MRI reconstruction

```
chmod +x run.sh
./run.sh GPU_INDEX
```

The results can be found at the folder `./result`.

## How to train the denoiser

Training the denoiser is not our main focus in this study. We will prepare the code later. On the meantime, check [https://github.com/cszn/DPIR](https://github.com/cszn/DPIR), which is the main structure/reference of our denoiser. They provide the training script in the repo.

## Citation
```
@article{gan2023block,
  title={Block coordinate plug-and-play methods for blind inverse problems},
  author={Gan, Weijie and Hu, Yuyang and Liu, Jiaming and An, Hongyu and Kamilov, Ulugbek and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```