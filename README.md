# A block β-model for community detection with node heterogeneity

This repository contains code for the paper *"A block β-model for community detection with node heterogeneity"*, including:
- `simulation/`: Simulated examples
- `realdata/`: Email-Eu-core network

---

## Requirements
Python packages (tested versions):
```
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.2.2
matplotlib==3.7.1
networkx==2.8.4
seaborn==0.12.2
```

## Quick Start

### Simulation Experiments
```bash
cd simulation/
python main_gbeta_1.py  # (All occurrences of 'gbeta' in the code correspond to 'BBM')
```
**Output**: Results will be saved in `simulation/`.

### Real Data Experiments
```bash
cd realdata/K=42
python main_realdata_gbeta.py 
```
**Output**: Results will be saved in `realdata/K=42`.

---

## Notes
- Contact: swye@link.cuhk.edu.hk for more questions.