# [ICLR 2025] DoF: A Diffusion Factorization Framework for Offline Multi-Agent Decision Making

![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)
![MIT](https://img.shields.io/badge/license-MIT-blue)


This is the official implementation of **"DoF: A Diffusion Factorization Framework for Offline Multi-Agent Decision Making"** published in **ICLR 2025**.

## 📌 Project Overview

**DoF** is a novel **Diffusion Factorization Framework** for tackling the challenges of **offline Multi-Agent Decision Making (MADM)**. While diffusion models have shown success in image and language generation, their application in cooperative multi-agent decision-making remains limited due to issues with **scalability** and **cooperation**. To address this, we extend the **Individual-Global-Max (IGM)** principle to the **Individual-Global-Identically Distributed (IGD)** principle, ensuring that outcomes from a centralized diffusion model are identically distributed with those from multiple decentralized models, enhancing both scalability and cooperative efficiency.

### 🚀 Key Innovations

1. 🔍 **Introduction of the IGD Principle**  
   A novel extension of the traditional IGM principle that ensures multi-agent diffusion outcomes are identically distributed across centralized and decentralized models.

2. 🏗️ **DoF Framework**  
   A powerful factorization framework that decomposes centralized diffusion models into multiple agent-specific models through:  
   - **Noise Factorization Function:** Factorizes the centralized diffusion noise into decentralized noise components while adhering to the IGD principle.  
   - **Data Factorization Function:** Models complex inter-agent data relationships, improving coordination and learning efficiency.

3. 📈 **Theoretical Validation**  
   We provide formal proof demonstrating that our noise factorization approach satisfies the **IGD principle**, ensuring both scalability and improved cooperative learning.

4. 🧪 **Extensive Empirical Evaluation**  
   Our experiments across various MADM benchmarks (including **SMAC**, **MPE**, and **MA-Mujoco**) demonstrate DoF's effectiveness and superior scalability compared to existing methods.



## 🎯 DoF-Trajectory 

### ⚙️ Environment Setup
#### Installation

```bash
sudo apt-get update
sudo apt-get install libssl-dev libcurl4-openssl-dev swig
conda create -n dof python=3.8
conda activate dof
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

### 📂 Offline Dataset


Datasets for SMAC, sourced from  [off-the-grid MARL](https://github.com/instadeepai/og-marl), and for MPE, provided by[OMAR](https://github.com/ling-pan/OMAR),  are available at the following links:
- [SMAC](https://huggingface.co/datasets/InstaDeepAI/og-marl/tree/main/core/smac_v1)
- [SMAC_V2](https://huggingface.co/datasets/InstaDeepAI/og-marl/tree/main/core/smac_v2)
- [MPE](https://github.com/ling-pan/OMAR)


### 🔄 Reproduction

**Run SMAC Environment**

For the 3m scenario in SMAC, use the following command:
```bash
python pyrun.py -c cfg/smac/mad_smac_3m_ctde_history_good.yaml
```

The evaluation results will be presented in `logs/smac`.


**Run MPE Environment**

For the Spread scenario in MPE, use the following command:
```bash
python pyrun.py -c cfg/mpe/mad_mpe_spread_ctde_exp.yaml 
```

**SMACv2 Environment**
Run Terran 5v5 Maps:
```bash
python pyrun.py -c cfg/smacv2/mad_smac_terran_5_vs_5_ctde_replay_history_DoF.yaml 
```
Run Terran 10v10 Maps:
```bash
python pyrun.py -c cfg/smacv2/mad_smac_terran_10_vs_10_ctde_replay_history_DoF_weight.yaml
```

Run Zerg Maps
```bash
python pyrun.py -c cfg/smacv2/mad_smac_zerg_ctde_replay_history_DoF_weight.yaml 
```


**Evaluate**
To evaluate the trained model:

`python pyrun.py -c cfg/eval_inv.yaml`




## 🎯 DoF-Policy

### ⚙️  Environment Setup
- MPE Environment:
```bash
pip install -e third_party/multiagent-particle-envs
pip install -e third_party/ddpg-agent
```

- MA-Mujoco Environment:
```bash
pip install -e third_party/multiagent_mujoco
```

### 📂 Offline Dataset

Datasets for MPE different tasks can be found at the following link [OMAR](https://github.com/ling-pan/OMAR)

Datasets for MA-Mujoco from from [off-the-grid MARL](https://sites.google.com/view/og-marl) can be found at the following links:
- [2halfcheetah](https://1drv.ms/u/c/1108e60a979b6a27/ESdqm5cK5ggggBGMnAEAAAABScHDktPYWCk-vwcq6C_bGw?e=rbolgT)
- [2ant](https://1drv.ms/u/c/1108e60a979b6a27/ESdqm5cK5ggggBGNnAEAAAABi_kmLd7Fboa8MLY7SBgHiA?e=ywGNLW)
- [4ant](https://1drv.ms/u/c/1108e60a979b6a27/ESdqm5cK5ggggBGOnAEAAAABR_Efk6YjTa-W8D_PxQ0M1Q?e=sWCLA4)

### 🔄  To replicate the results, run:

Please follow the below commands to replicate the results in the paper.

```bash
python train_policy_main.py --config <DoF-P_YAML_DIR>/<ENVIRONMENT>_<SUB_ENVIRONMENT>_<DATASET_TYPE>_dof_<DATA_MODE>_<NOISE_MODE>_<SEED>.yaml
```

Modify the YAML configuration for data and noise factorization:

```yaml
data_factorization_mode: "concat" # "concat" "w-concat" "default"
noise_factorization_mode: "concat" # "concat" "w-concat"
```


## 📊 Reproducing Baselines 

### MADIFF
Refer to the official [MADIFF](https://github.com/zbzhu99/madiff) repository for installation and configuration.

To train:

```
# For Multi-Agent Particle Environment
python run_experiment.py -e exp_specs/mpe/<task>/mad_mpe_<task>_attn_<dataset>.yaml
```


### OG-MARL

Follow the official [OG-MARL](https://github.com/instadeepai/og-marl) repository for setup and experiments.

Clone and install:
```
git clone https://github.com/instadeepai/og-marl.git
pip install -e .[tf2_baselines]
```

## Citation
@inproceedings{dof2025,
  title={DoF: A Diffusion Factorization Framework for Offline Multi-Agent Decision Making},
  author={Li, Chao and Deng, Ziwei and Lin, Chenxing and Chen, Wenqi and Fu, Yongquan and Liu, Weiquan and Wen, Chenglu and Wang, Cheng and Shen, Siqi},
  booktitle={International Conference on Learning Representations},
  year={2025}
}

## Acknowledgements
Our codebase is inspired by the following repositories:
- [Decision Diffuser](https://github.com/anuragajay/decision-diffuser)
- [MADIFF](https://github.com/zbzhu99/madiff)
- [MADDPG-PyTorch](https://github.com/shariqiqbal2810/maddpg-pytorch)

