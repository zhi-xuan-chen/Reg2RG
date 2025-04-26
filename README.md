# Reg2RG 🚀
*Official repository for the IEEE TMI paper **“Large Language Model with Region‑Guided Referring and Grounding for CT Report Generation”***

## 📦 Installation
```bash
git clone https://github.com/zhi-xuan-chen/Reg2RG.git
cd Reg2RG
conda create -n reg2rg python=3.9
conda activate reg2rg
pip install -r requirements.txt
```

## 📁 Data Preparation
The **RadGenome‑ChestCT** dataset used in this work can be downloaded from the Hugging Face hub:
<https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT>

## 🗄️ Model Checkpoints
- **Base LLM:** [Llama‑2‑7b‑chat‑hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)  
- **Pre‑trained & auxiliary checkpoints:** [Reg2RG](https://huggingface.co/Trusure/Reg2RG/tree/main)

## 🤖 Inference
After the data and checkpoints are ready, you can inference the model on **RadGenome‑ChestCT** with our pre‑trained weights.

1. Create a custom configuration file `[your_config_name].sh` in `configs/test_radgenome/` (you can copy and modify `configs/test_radgenome/jhcpu7.sh`).
2. Run:
   ```bash
   cd scripts
   bash test_radgenome.sh [your_config_name]
   ```

## 📊 Evaluation
After inference, evaluate the generated reports with the scripts in the `evaluation/` directory.
> **All scripts read file paths from variables *inside* the script.  
> Please open each file and set the path (e.g. `results_path`) to your own CSV results before running.**

### Whole‑Report Evaluation

1. **Strip region prefixes**  
   ```bash
   python evaluation/rm_region_text.py
   ```
2. **Compute Natural Language Generation (NLG) metrics**  
   ```bash
   python evaluation/hf_nlg_evaluation.py
   ```
3. **Compute Clinical Efficacy (CE) metrics**  
   ```bash
   python evaluation/ce_evaluator_ct2rep/ce_evaluation.py
   ```

### Region‑Level Evaluation

1. **Region prediction accuracy**  
   ```bash
   python evaluation/region_pred_acc.py
   ```
2. **Split into region‑specific reports**  
   ```bash
   python evaluation/parse_report_to_region.py
   ```
3. **Compute region‑level NLG metrics**  
   ```bash
   python evaluation/hf_nlg_evaluation_region.py
   ```
4. **Compute region‑level CE metrics**  
   ```bash
   python evaluation/ce_evaluator_ct2rep/ce_evaluation_region.py
   ```

## 🏋️‍♂️ Training
You can fine‑tune the model on your own dataset using our pre‑trained checkpoint.

1. Add your dataset class to `src/Dataset/`, following the style of the existing datasets.  
2. Create a config `[your_config_name].sh` in `configs/train_radgenome/` (copy and adjust `configs/train_radgenome/jhcpu7.sh`).  
3. Run the training script:
   ```bash
   cd scripts
   bash train_radgenome.sh [your_config_name]
   ```

## 📄 Citation
If you find our work useful, please consider citing it:

```bibtex
@article{chen2025reg2rg,
  title   = {Large Language Model with Region-Guided Referring and Grounding for CT Report Generation},
  author  = {Chen, Zhixuan and Bie, Yequan and Jin, Haibo and Chen, Hao},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2025},
  publisher = {IEEE}
}
```

## 🙏 Acknowledgements
This project builds on the [RadFM](https://github.com/chaoyi-wu/RadFM) repository and the [LLaMA 2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model.  
We sincerely thank the original authors for their invaluable contributions to the community.
