
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

