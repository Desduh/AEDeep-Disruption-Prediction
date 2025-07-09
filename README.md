# AEDeep: Disruption Prediction with Machine Learning

> **Preliminary Study of Disruption Prediction with Machine Learning: from Solar Plasma to Tokamaks**  
>  
> **Authors:**  
> Carlos Eduardo Falandes<sup>1*</sup>, Reinaldo R. Rosa<sup>1</sup>, Salatiel A. A. Jordão<sup>1</sup>,  
> Rubens A. Sautter<sup>2</sup>, Luan O. Barauna<sup>1</sup>, Jiro Kawamura<sup>3</sup>,  
> Pablo A. Medina Sanchez<sup>4</sup>, Juan A. Valdivia<sup>5</sup>, Daniel A. S. Mendes<sup>1</sup>  
>  
> <sup>1</sup>National Institute for Space Research (INPE), Brazil  
> <sup>2</sup>Federal University of Technology – Paraná (UTFPR), Brazil  
> <sup>3</sup>ITER – International Thermonuclear Experimental Reactor, France  
> <sup>4</sup>University of Los Andes, Colombia  
> <sup>5</sup>University of Chile, Chile  
>  
> 📧 *carlos.falandes@inpe.br*

---

## 🔍 Project Overview

This repository contains the code, data, and environment files for **AEDeep**, a deep learning-based framework for disruption prediction in plasma systems using synthetic multifractal time series generated via the **p-model** and anomaly detection with **LSTM** networks.

---

## 📁 Directory Structure

```
AEDeep-Disruption-Prediction/
│
├── src/                  # Source code (models, utils, plotting)
├── data/                 # Image frames from time series (EUV, Heat Flux)
├── tests/                # CSV files with synthetic and real time series
├── environment.yaml      # Conda environment with dependencies
└── README.md             # This file
```

---

## ⚙️ Setting Up the Conda Environment

1. Make sure you have **Conda** installed ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) is recommended).

2. Clone the repository:

```bash
git clone https://github.com/Desduh/AEDeep-Disruption-Prediction.git
cd AEDeep-Disruption-Prediction
```

3. Create and activate the environment using the `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate aedeep
```

---

## 🚀 How to Run

### 1. Generate synthetic p-model time series:
```bash
python src/pmodel_generation.py
```

### 2. Train the LSTM model:
```bash
python src/train.py
```

### 3. Evaluate the model on real/simulated data:
```bash
python src/test.py
```

---

## 📊 Sample Results

- Time series with anomaly detections
- Prediction probability curves (10-step forecast horizon)
- Evaluation on:
  - Synthetic data (p-model)
  - Solar observational data (SDO/AIA)
  - Simulated heat flux (SPARC-to-ITER)

---

## 📌 Scientific Highlights

- Early prediction of extreme events in both solar and fusion plasmas
- Detection of multifractal cascade structures using synthetic modeling
- Applications in space weather forecasting and disruption mitigation in tokamaks

---

## 📄 Paper Reference

Falandes, C. E., Rosa, R. R., et al. (2025).  
**Preliminary Study of Disruption Prediction with Machine Learning: from Solar Plasma to Tokamaks**. _Unpublished manuscript_.

[📎 PDF of the Article](./falandes_unpublished_preliminary_study_of_disruption_prediction_with_machine_learning.pdf)

---

## 📚 Citation

```bibtex
@unpublished{falandes2025aedeep,
  title     = {Preliminary Study of Disruption Prediction with Machine Learning: from Solar Plasma to Tokamaks},
  author    = {Falandes, Carlos Eduardo and Rosa, Reinaldo R. and Jordão, Salatiel A. A. and Sautter, Rubens A. and Barauna, Luan O. and Kawamura, Jiro and Medina Sanchez, Pablo A. and Valdivia, Juan A. and Mendes, Daniel A. S.},
  year      = {2025},
  note      = {Unpublished manuscript},
}
```

---

## 📬 Contact

📧 **Carlos Eduardo Falandes**  
Email: [carlos.falandes@inpe.br](mailto:carlos.falandes@inpe.br)  
Project repository: [github.com/Desduh/AEDeep-Disruption-Prediction](https://github.com/Desduh/AEDeep-Disruption-Prediction)

---
