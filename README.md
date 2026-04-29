# MasHeNe
### MasHeNe: A Benchmark for Head and Neck CT Mass Segmentation using Window-Enhanced Mamba with Frequency-Domain Integration


## Abstract
Head and neck masses are space-occupying lesions that can compress the airway and esophagus and may affect nerves and blood vessels. Available public datasets primarily focus on malignant lesions and often overlook other space-occupying conditions in this region. To address this gap, we introduce MasHeNe, an initial dataset of 3,779 contrast-enhanced CT slices that includes both tumors and cysts with pixel-level annotations. We also establish a benchmark using standard segmentation baselines and report common metrics to enable fair comparison. In addition, we propose the Windowing-Enhanced Mamba with Frequency integration (WEMF) model. WEMF applies tri-window enhancement to enrich the input appearance before feature extraction. It further uses multi-frequency attention to fuse information across skip connections within a U-shaped Mamba backbone. On MasHeNe, WEMF attains the best performance among evaluated methods, with a Dice of 70.45%, IoU of 66.89%, NSD of 72.33%, and HD95 of 5.12 mm. This model indicates stable and strong results on this challenging task. MasHeNe provides a benchmark for head-and-neck mass segmentation beyond malignancy-only datasets. The observed error patterns also suggest that this task remains challenging and requires further research. Our dataset and code are available at https://github.com/drthaodao3101/MasHeNe.git. 


## Download Dataset:
Link to download our dataset at [Drive](https://drive.google.com/drive/folders/1-Ow93acZSoD31f98QV6CXmnvKFwdBqeQ?usp=sharing)


## Install Repo and Env
- Clone our repository:
  ```bash
  git clone https://github.com/your-username/MasHeNe.git mashene
  cd mashene

- Requirements: requirements.txt
   ```bash
  conda create -n mashene python=3.8
  conda activate mashene`
  pip install -r requirements.txt

## Create "pre_trained_weights" folder
You could download the weights (vmamba_small_e238_ema.pth and vmamba_tiny_e292.pth) used by VM-UNet from [Baidu Drive](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy) or [Google Drive](https://drive.google.com/drive/folders/1Fr7zM1wq7106d0P7_3oeU5UZqUvk2KaP?usp=sharing) 

## Train: 
`python train.py --mode train`

## Evaluation:
`python train.py --mode test`


## Acknowledgement :heart:
This project is based on Mamba, VM-UNet ([paper](https://arxiv.org/abs/2402.02491), [code](https://github.com/JCruan519/VM-UNet)). Thanks for their wonderful work.


```bibtex
@article{dao2025mashene,
  title={MasHeNe: A Benchmark for Head and Neck CT Mass Segmentation using Window-Enhanced Mamba with Frequency-Domain Integration},
  author={Dao, Thao Thi Phuong and Nguyen, Tan-Cong and Thanh, Nguyen Chi and Viet, Truong Hoang and Do, Trong-Le and Tran, Mai-Khiem and Pham, Minh-Khoi and Le, Trung-Nghia and Tran, Minh-Triet and Le, Thanh Dinh},
  journal={arXiv preprint arXiv:2512.01563
        
        },
  year={2025}
}
```

