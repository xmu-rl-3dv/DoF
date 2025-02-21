# DoF: A Diffusion Factorization Framework for Offline Multi-Agent Decision Making

![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)
![MIT](https://img.shields.io/badge/license-MIT-blue)

## Acknowledgements
The development of our codebase is inspired by  [decision-diffuser repo](https://github.com/anuragajay/decision-diffuser),  [madiff](https://github.com/zbzhu99/madiff) and [maddpg-pytorch](https://github.com/shariqiqbal2810/maddpg-pytorch)

## DoF-Trajectory 

### Environment Setup

#### Installation

```bash
sudo apt-get update
sudo apt-get install libssl-dev libcurl4-openssl-dev swig
conda create -n dof python=3.8
conda activate dof
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```


#### Setup SMAC

1. Run `scripts/smac.sh` to install *StarCraftII*.

2. Install SMAC:

    ```bash
    pip install git+https://github.com/oxwhirl/smac.git
    ```

### Offline Dataset


Datasets for SMAC, sourced from  [off-the-grid MARL](https://github.com/instadeepai/og-marl), and for MPE, provided by[OMAR](https://github.com/ling-pan/OMAR),  are available at the following links:
- [SMAC](https://huggingface.co/datasets/InstaDeepAI/og-marl/tree/main/core/smac_v1)
- [SMAC_V2](https://huggingface.co/datasets/InstaDeepAI/og-marl/tree/main/core/smac_v2)
- [MPE](https://github.com/ling-pan/OMAR)


### Reproduction

**Run SMAC Environment**

For the 3m scenario in SMAC, use the following command:
```bash
python pyrun.py -c cfg/iclr/mad_smac_3m_ctde_history_good.yaml
```

The evaluation results will be presented in `logs/smac`.


**Run MPE Environment**

For the Spread scenario in MPE, use the following command:
```bash
python pyrun.py -c cfg/iclr/mad_mpe_spread_ctde_exp.yaml 
```

**SMACv2 Environment**
Run Terran 5v5 Maps:
```bash
python pyrun.py -c cfg/iclr/mad_smac_terran_5_vs_5_ctde_replay_history_DoF.yaml 
```
Run Terran 10v10 Maps:
```bash
python pyrun.py -c cfg/iclr/mad_smac_terran_10_vs_10_ctde_replay_history_DoF_weight.yaml
```

Run Zerg Maps
```bash
python pyrun.py -c cfg/iclr/mad_smac_zerg_ctde_replay_history_DoF_weight.yaml 
```


**Evaluate**
To evaluate the trained model:

`python pyrun.py -c cfg/eval_inv.yaml`




## DoF-Policy

### Environment Setup
- MPE Environment:
```bash
pip install -e third_party/multiagent-particle-envs
pip install -e third_party/ddpg-agent
```

- MA-Mujoco Environment:
```bash
pip install -e third_party/multiagent_mujoco
```

### Offline Dataset

Datasets for MPE different tasks can be found at the following link [OMAR](https://github.com/ling-pan/OMAR)

Datasets for MA-Mujoco from from [off-the-grid MARL](https://sites.google.com/view/og-marl) can be found at the following links:
- [2halfcheetah](https://1drv.ms/u/c/1108e60a979b6a27/ESdqm5cK5ggggBGMnAEAAAABScHDktPYWCk-vwcq6C_bGw?e=rbolgT)
- [2ant](https://1drv.ms/u/c/1108e60a979b6a27/ESdqm5cK5ggggBGNnAEAAAABi_kmLd7Fboa8MLY7SBgHiA?e=ywGNLW)
- [4ant](https://1drv.ms/u/c/1108e60a979b6a27/ESdqm5cK5ggggBGOnAEAAAABR_Efk6YjTa-W8D_PxQ0M1Q?e=sWCLA4)

### Reproduction

Please follow the below commands to replicate the results in the paper.

```bash
python train_policy_main.py --config <DoF-P_YAML_DIR>/<ENVIRONMENT>_<SUB_ENVIRONMENT>_<DATASET_TYPE>_dof_<DATA_MODE>_<NOISE_MODE>_<SEED>.yaml
```

The methods for data factorization <DATA_MODE> and noise factorization <NOISE_MODE> in paper can be selected and modified in the YAML file:

```yaml
data_factorization_mode: "concat" # "concat" "w-concat" "default"
noise_factorization_mode: "concat" # "concat" "w-concat"
```

The evaluation results will be presented in `results/`


## Reproducing Baselines 

### MADIFF
For installation and configuration, please refer to the official repository:

https://github.com/zbzhu99/madiff

To start training, run the following commands

```
# multi-agent particle environment
python run_experiment.py -e exp_specs/mpe/<task>/mad_mpe_<task>_attn_<dataset>.yaml  # CTCE
python run_experiment.py -e exp_specs/mpe/<task>/mad_mpe_<task>_ctde_<dataset>.yaml  # CTDE
# ma-mujoco
python run_experiment.py -e exp_specs/mamujoco/<task>/mad_mamujoco_<task>_attn_<dataset>_history.yaml  # CTCE
python run_experiment.py -e exp_specs/mamujoco/<task>/mad_mamujoco_<task>_ctde_<dataset>_history.yaml  # CTDE
# smac
python run_experiment.py -e exp_specs/smac/<map>/mad_smac_<map>_attn_<dataset>_history.yaml  # CTCE
python run_experiment.py -e exp_specs/smac/<map>/mad_smac_<map>_ctde_<dataset>_history.yaml  # CTDE
```
To evaluate the trained model, first replace the log_dir with those need to be evaluated in exp_specs/eval_inv.yaml and run

`python run_experiment.py -e exp_specs/eval_inv.yaml`


### OG-MARL

For imore details, please refer to the official repository:

https://github.com/instadeepai/og-marl

Clone this repository.

`git clone https://github.com/instadeepai/og-marl.git`

Install og-marl and its requirements. We tested og-marl with Python 3.10 and Ubuntu 20.04. Consider using a conda virtual environment.

`pip install -e .[tf2_baselines]`

Download environment files. We will use SMACv1 in this example. MAMuJoCo installation instructions are included near the bottom of the README.

`bash install_environments/smacv1.sh`

Download environment requirements.

`pip install -r install_environments/requirements/smacv1.txt`

Train an offline system. In this example we will run Independent Q-Learning with Conservative Q-Learning (iql+cql). The script will automatically download the neccessary dataset if it is not found locally.

`python og_marl/tf2_systems/offline/iql_cql.py task.source=og_marl task.env=smac_v1 task.scenario=3m task.dataset=Good`

You can find all offline systems at `og_marl/tf2_systems/offline/` and they can be run similarly. Be careful, some systems only work on discrete action space environments and vice versa for continuous action space environments. The config files for systems are found at `og_marl/tf2_systems/offline/configs/`. 
