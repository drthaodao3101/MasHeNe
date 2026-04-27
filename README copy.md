<<<<<<< HEAD
# VM-UNet
This is the official code repository for "VM-UNet: Vision Mamba UNet for Medical
Image Segmentation". {[Arxiv Paper](https://arxiv.org/abs/2402.02491)}

## Abstract
In the realm of medical image segmentation, both CNN-based and Transformer-based models have been extensively explored. However, CNNs exhibit limitations in long-range modeling capabilities, whereas Transformers are hampered by their quadratic computational complexity. Recently, State Space Models (SSMs), exemplified by Mamba, have emerged as a promising approach. They not only excel in modeling long-range interactions but also maintain a linear computational complexity. In this paper, leveraging state space models, we propose a U-shape architecture model for medical image segmentation, named Vision Mamba UNet (VM-UNet). Specifically, the Visual State Space (VSS) block is introduced as the foundation block to capture extensive contextual information, and an asymmetrical encoder-decoder structure is constructed. We conduct comprehensive experiments on the ISIC17, ISIC18, and Synapse datasets, and the results indicate that VM-UNet performs competitively in medical image segmentation tasks. To our best knowledge, this is the first medical image segmentation model constructed based on the pure SSM-based model. We aim to establish a baseline and provide valuable insights for the future development of more efficient and effective SSM-based segmentation systems.

## 0. Main Environments
```bash
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```
The .whl files of causal_conv1d and mamba_ssm could be found here. {[Baidu](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k)}

## 1. Prepare the dataset

### ISIC datasets
- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm) or [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)}. 

- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

### Synapse datasets

- For the Synapse dataset, you could follow [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) to download the dataset, or you could download them from {[Baidu](https://pan.baidu.com/s/1JCXBfRL9y1cjfJUKtbEhiQ?pwd=9jti)}.

- After downloading the datasets, you are supposed to put them into './data/Synapse/', and the file format reference is as follows.

- './data/Synapse/'
  - lists
    - list_Synapse
      - all.lst
      - test_vol.txt
      - train.txt
  - test_vol_h5
    - casexxxx.npy.h5
  - train_npz
    - casexxxx_slicexxx.npz

## 2. Prepare the pre_trained weights

- The weights of the pre-trained VMamba could be downloaded [here](https://github.com/MzeroMiko/VMamba) or [Baidu](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy). After that, the pre-trained weights should be stored in './pretrained_weights/'.



## 3. Train the VM-UNet
```bash
cd VM-UNet
python train.py  # Train and test VM-UNet on the ISIC17 or ISIC18 dataset.
python train_synapse.py  # Train and test VM-UNet on the Synapse dataset.
```

## 4. Obtain the outputs
- After trianing, you could obtain the results in './results/'

## 5. Acknowledgments

- We thank the authors of [VMamba](https://github.com/MzeroMiko/VMamba) and [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) for their open-source codes.
=======
<div align="center">
<h1>3M-Unet</h1>
<h3>Mixture of Expert with Multi-scale Feature Extraction on Mamba for Laryngoscopy Image Segmentation</h3>


### News
* **` Feb, 2025`:** We introduce our model for laryngoscopy image segmentation


## Abstract
Segmentation in laryngoscopy images plays a vital role in evaluating lesion edges and their invasive level to related anatomical structures. However, there has been less research about such a topic until now. In this study, we introduce a novel laryngoscopy image dataset for vocal fold and lesion segmentation. Moreover, we propose a new deep learning-based approach for segmenting glottic anatomical landmarks and lesions in flexible laryngoscopy images. Our method is based on integrating mixture of expert and multi-view edge extraction into state space architecture. We trained and evaluated our model on a dataset of 1724 flexible laryngoscopy images, and achieved an overall Dice coefficient and HD95 of \highlight{Xxx} and \highlight{Xxx}, respectively. We also conducted additional experiments to evaluate the robustness of our method with each subset. The results indicate that our proposed approach not only effectively segments glottic anatomical landmarks and lesions in flexible laryngoscopy images but also obtains efficient computational cost, providing a valuable tool for clinical applications. 

## Overview
<div align="center">
<img src="fig/3munet.png" />
</div>

## Install Repo and Env
- Clone our repository:
  ```bash
  git clone https://github.com/your-username/3M-Unet.git 3munet
  cd 3munet

- Requirements: requirements.txt
   ```bash
  conda create -n 3munet python=3.8
  conda activate 3munet`
  pip install -r requirements.txt

## Train: 
`python train_vofo.py --mode train  # Train and val 3M-UNet on our laryngoscopy image dataset.`

## Evaluation:
`python train_vofo.py --mode test`


**Note**
If you want to visualize:
`python train_vofo.py --mode test --visualize`
It will save the output images in the .\result_vofo\visualize folder, with the colors specified in the config_setting_vofo.py file.


## Acknowledgement :heart:
This project is based on Mamba ([paper](https://arxiv.org/abs/2402.02491), [code](https://github.com/JCruan519/VM-UNet)). Thanks for their wonderful work.

## Citation
If you find our model that is useful in your research or applications, please consider giving us a star 🌟 and citing it in the following BibTeX entry.

```bibtex
 @article{3munet,
  title={3M-Unet: Mixture of Expert with Multi-scale Feature Extraction on Mamba for Laryngoscopy Image Segmentation},
  author={},
  journal={},
  year={2025}
}
```
>>>>>>> 2325e147a143b968f0b4f782dc595967cb467700
