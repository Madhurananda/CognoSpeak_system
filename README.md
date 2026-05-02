# CognoSpeak System — Acoustic & Linguistic Modelling Pipeline

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Conda](https://img.shields.io/badge/conda-reproducible-green)
![Machine Learning](https://img.shields.io/badge/methods-ML%20%7C%20LLM%20%7C%20Speech-orange)
![Open Science](https://img.shields.io/badge/open--science-reproducible-success)
![IEEE CIHM 2025](https://img.shields.io/badge/publication-IEEE%20CIHM%202025-blueviolet)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)

---

## Overview

**CognoSpeak System** is a reproducible research pipeline for modelling **speech biomarkers of cognitive decline** using both **acoustic** and **linguistic representations**.

The repository implements experiments conducted on speech recordings from **126 participants**, including:

- Healthy Controls  
- Mild Cognitive Impairment (MCI)  
- Dementia  

The system supports both:

- classical machine learning models (CPU)
- foundation / large language model experiments (GPU)

and forms the experimental backbone of the CognoSpeak clinical speech assessment framework.

---

## Associated Publication

This repository accompanies the paper:

> **CognoSpeak: An Automatic, Remote Assessment of Early Cognitive Decline in Real-World Conversational Speech**  
> *2025 IEEE Symposium on Computational Intelligence in Health and Medicine (CIHM)*

---

## Key Features

- End-to-end reproducible modelling pipeline  
- Acoustic speech biomarker experiments  
- Linguistic and foundation model analysis  
- Multi-GPU execution support  
- Standardised experiment logging  
- Designed for clinical AI and dementia research  

---

## Repository Structure

```
CognoSpeak_system/
│
├── CognoSpeak_acoustics.py       # Acoustic feature classifiers
├── CognoSpeak_linguistics.py     # Linguistic + foundation model experiments
│
├── requirements.txt              # Python dependency list
├── requirements.yml              # Conda environment definition
│
├── logs/                         # Automatically generated experiment logs
│
└── README.md
```

---

## Installation

We strongly recommend using **Conda** to ensure experiment reproducibility.

### 1 — Clone Repository

```bash
git clone https://github.com/<username>/CognoSpeak_system.git
cd CognoSpeak_system
```

---

### 2 — Create Environment

Using Conda:

```bash
conda env create -f requirements.yml
conda activate CognoSpeak
```

Alternative (pip):

```bash
pip install -r requirements.txt
```

---

### 3 — Verify GPU Availability (Optional)

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

GPU acceleration is recommended for linguistic experiments.

---

## Experimental Workflow

The repository contains **two complementary modelling pipelines**.

---

## 1 — Acoustic Classification Pipeline

Runs classical machine learning experiments using extracted acoustic speech features.

### Execution

```bash
python CognoSpeak_acoustics.py <NUM_CPUS> |& tee -a ../logs/acoustic_results.txt
```

Example:

```bash
python CognoSpeak_acoustics.py 12 |& tee -a ../logs/acoustic_results.txt
```

### Functionality

- Loads acoustic feature representations  
- Executes classification experiments  
- Parallel CPU processing  
- Automatically logs results  

**Environment:** CPU recommended

---

## 2 — Linguistic & Foundation Model Pipeline

Runs linguistic modelling and large language model experiments.

### Execution

```bash
python CognoSpeak_linguistics.py <TOKEN_NAME> <GPU_IDS> |& tee -a ../logs/linguistics_results.txt
```

Example:

```bash
python CognoSpeak_linguistics.py CognoSpeak 0,1,2,3 |& tee -a ../logs/linguistics_results.txt
```

### Arguments

| Argument | Description |
|---|---|
| `TOKEN_NAME` | Experiment identifier |
| `GPU_IDS` | Comma-separated CUDA device IDs |

### Functionality

- Loads linguistic/transcript features  
- Runs foundation model experiments  
- Multi-GPU parallel execution  
- Full experiment logging  

**Environment:** GPU required

---

## Outputs

Running the pipelines generates:

- Model evaluation metrics  
- Classification results  
- Experimental logs  
- Reproducible experiment records  

All logs are written automatically to:

```
logs/
```

---

## Reproducibility & Open Science

This repository follows reproducible research principles:

- Environment-controlled dependencies  
- Script-based experiment execution  
- Publication-aligned modelling  
- Transparent experiment logging  

The framework may be extended for:

- speech biomarker discovery  
- dementia detection research  
- clinical conversational AI  
- multimodal cognitive assessment  

---

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{pahar2025cognospeak,
  title={CognoSpeak: an automatic, remote assessment of early cognitive decline in real-world conversational speech},
  author={Pahar, Madhurananda and Tao, Fuxiang and Mirheidari, Bahman and Pevy, Nathan and Bright, Rebecca and Gadgil, Swapnil and Sproson, Lise and Braun, Dorota and Illingworth, Caitlin and Blackburn, Daniel and Christensen, Heidi},
  booktitle={2025 IEEE Symposium on Computational Intelligence in Health and Medicine (CIHM)},
  pages={1--7},
  year={2025},
  doi={10.1109/CIHM64979.2025.10969487}
}
```

Pahar, M., Tao, F., Mirheidari, B., Pevy, N., Bright, R., Gadgil, S., Sproson, L., Braun, D., Illingworth, C., Blackburn, D., & Christensen, H. (2025).  
*CognoSpeak: an automatic, remote assessment of early cognitive decline in real-world conversational speech.*  
IEEE CIHM.

---

## Data Availability

⚠️ **No participant data is distributed in this repository.**

This repository contains **code only**.  
Access to speech data requires appropriate ethical approval and authorization.

---

## License

This project is licensed under the **Apache License 2.0**.

See the `LICENSE` file for full details.

---

## Zenodo Archive

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14515541.svg)](https://doi.org/10.5281/zenodo.14515541)

---

## Contact

**CognoSpeak Research Team**

For collaboration or research enquiries, please open an issue or contact the authors of the publication.

---

⭐ If this repository supports your research, please consider starring the project.
