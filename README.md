# Implementation of Scalable Wasserstein Gradient Flow for Generative Modeling through Unbalanced Optimal Transport (ICML, 2024)#
Link to paper: https://arxiv.org/pdf/2402.05443

## Training Wasserstein-Gradient-Flow UOTM ##

#### Toy ####
Command for OTM/fixed-source-UOTM/UOTM and S-JKO on toy datasets.

Command for both-relaxed-UOTM
```
python train.py --exp {exp_name} --phi1 {kl,chi,softplus} --phi2 {kl,chi,softplus} --dataset {25gaussian,twocircles} --image_size 2 --model_name toy --nz 2 --batch_size 400 --lr_g 1e-4 --lr_d 2e-5 --reg_name none --num_phase 1 --init_h 1 --h 1 --init_num_iterations 50000 --save_image_every 5000
```

Command for fixed-source-UOTM:
```
python train.py --exp {exp_name} --phi2 {kl,chi,softplus} --dataset {25gaussian,twocircles} --image_size 2 --model_name toy --nz 2 --batch_size 400 --lr_g 1e-4 --lr_d 2e-5 --reg_name none --num_phase 1 --init_h 1 --h 1 --init_num_iterations 50000 --save_image_every 5000
```

Command for S-JKO:
```
python train.py --exp {exp_name} --phi2 {kl,chi,softplus} --dataset {25gaussian,twocircles} --image_size 2 --model_name toy --nz 2 --batch_size 400 --lr_g 1e-4 --lr_d 2e-5 --reg_name none --num_phase 10 --init_h 1 --h 1 --init_num_iterations 5000 --num_iterations 5000 --save_image_every 5000
```


#### CIFAR-10 ####
The main comparison model is original UOTM. Here, $1/\tau = 2dh$ where $d$ is an image dimension and $h$ is a step size (init_h). For example, if we set $h=0.1$, then $\tau\approx 0.0016$. All models on CIFAR10 are trained on 4 32-GB V100 GPU. 

Command for JKO-UOTM is as the follows:
```
python train.py --exp {exp_name} --phi2 {kl,chi,softplus} --num_phase 100 --init_h 0.1 --h 0.1 --init_num_iterations 10000 --num_iterations 2000 --use_ema --ema_decay 0.9999 --lr_scheduler
```

Command for JKO-UOTM with JSD is as the follows:
```
python train.py --exp {exp_name} --num_phase 100 --init_h 0.1 --h 0.1 --init_num_iterations 10000 --num_iterations 2000 --use_ema --ema_decay 0.9999 --lr_scheduler
```

Command for fixed-source-UOTM is as the follows:
```
python train.py --exp {exp_name} --phi2 {kl,chi,softplus} --num_phase 1 --init_h 0.1 --h 0.1 --init_num_iterations 120000 --use_ema --ema_decay 0.9999 --lr_scheduler
```

## Bibtex ##
Cite our paper using the following BibTeX item:
```
@article{choi2024scalable,
  title={Scalable Wasserstein Gradient Flow for Generative Modeling through Unbalanced Optimal Transport},
  author={Choi, Jaemoo and Choi, Jaewoong and Kang, Myungjoo},
  booktitle={International Conference on Machine Learning},
  year={2024},
  organization={PMLR}
}
```