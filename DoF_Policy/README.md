# DoF-Policy
## Acknowledgements
The development of our codebase is inspired by  [decision-diffuser repo](https://github.com/anuragajay/decision-diffuser),  [madiff](https://github.com/zbzhu99/madiff) and [maddpg-pytorch](https://github.com/shariqiqbal2810/maddpg-pytorch)

## Environment Setup
- MPE Environment:
```bash
pip install -e third_party/multiagent-particle-envs
pip install -e third_party/ddpg-agent
```

- MA-Mujoco Environment:
```bash
pip install -e third_party/multiagent_mujoco
```

## Offline Dataset

Datasets for MPE different tasks can be found at the following link [OMAR](https://github.com/ling-pan/OMAR)

Datasets for MA-Mujoco from from [off-the-grid MARL](https://sites.google.com/view/og-marl) can be found at the following links:
- [2halfcheetah](https://1drv.ms/u/c/1108e60a979b6a27/ESdqm5cK5ggggBGMnAEAAAABScHDktPYWCk-vwcq6C_bGw?e=rbolgT)
- [2ant](https://1drv.ms/u/c/1108e60a979b6a27/ESdqm5cK5ggggBGNnAEAAAABi_kmLd7Fboa8MLY7SBgHiA?e=ywGNLW)
- [4ant](https://1drv.ms/u/c/1108e60a979b6a27/ESdqm5cK5ggggBGOnAEAAAABR_Efk6YjTa-W8D_PxQ0M1Q?e=sWCLA4)

## Reproduction

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

