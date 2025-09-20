# Bg_Model: Bandgap Prediction for Solid-State Materials

Bg_Model is a Patolli-based framework for predicting bandgaps in crystalline (solid-state) materials and for exploring AI-driven approaches to accelerate materials discovery.  
It combines classical machine learning and deep learning methods with extensive feature engineering and analysis, providing a versatile platform for predictive experimentation in materials science.


---

## Features

- **General Exploration and Experimentation Notebook:** From descriptor generation and analysis to deep learning architecture training.
- **Feature Engineering & Dataset Preparation:** Generate descriptors and prepare unified datasets for model training.
- **Concatenated Descriptor Exploration:** Combine CNN-extracted features with Patolli descriptors for ML and DL modeling.
- **FGSM Analysis:** Evaluate model behaviour to adversarial perturbations.
- **Feature Abstraction Analysis:** Examine intermediate representations from the penultimate layer of top-performing models.
- **Bandgap Simulator:** Predict the bandgap of user-defined compounds (perovskite) using trained models.

---

## Repository Contents
```text
BgModel/
├── BandgapModel.ipynb                  # Main exploration (Patolli descriptors)
├── Descriptor_Concatenation.ipynb      # Building concatenated CNN + Patolli datasets
├── BandgapModel_Concatenation.ipynb    # CNN + Patolli combined features
├── BandgapModel_FGSM.ipynb             # Robustness analysis with FGSM
├── BandgapModel_FeatureAbstraction.ipynb # Penultimate layer feature analysis
├── ChemicalSpace_Dataset_Generation.ipynb # Materials API dataset building
├── BandgapSimulator_Perovskites.ipynb # Custom bandgap simulator
│    
│
├── data/      # Training and test datasets (limited due to file size)
├── models/    # Trained models (limited due to file size)
├── README.md
├── .gitignore
└── requirements.txt
```
---

## Dependencies

- Python 3.12.4  
- TensorFlow 2.17 
- keras 
- scikit-learn  
- pandas  
- numpy  
- matplotlib  
- XGBoost  
- pydot
- itertools
  

> Use `pip install -r requirements.txt` to install all dependencies.

---

## Usage

All notebooks are interactive and can be run in Jupyter or Colab. 

My workflow:

1. Explore and analyze Patolli-generated descriptors exclusively, including model architectures and training, in **`DL_Model_Exploration.ipynb`**.
2. Generate concatenated datasets that include CNN-extracted features from X-ray diffractogram models in **`FeatureEngineering_and_DatasetPreparation.ipynb`**.
3. Investigate the combined descriptor spaces in **`Concatenated_Descriptor_Analysis.ipynb`**.
4. Adversarial attack implementation (FGSM) in **`FGSM_Analysis.ipynb`**.
5. Analyze feature abstractions from top-performing models using **`FeatureAbstraction_PenultimateLayer.ipynb`**.
6. Simulate bandgap energy values for custom compounds in **`Bandgap_Simulator.ipynb`**.


---

## Datasets

The `data/` folder contains training and test sets.  
> Note: Trained model files and datasets are provided selectively due to their large size. Users can train models using the notebooks.

---

## Citation

This work is based upon:  

J. I. Gómez-Peralta, X. Bokhimi. Journal of Solid State Chemistry, Vol. 285 (2020) 121253.

GitHub repositories:
    1. [Patolli](https://github.com/gomezperalta/patolli_2021)
    2. [Perovskite_Simulator](https://github.com/gomezperalta/perovskites_simulator) 


> This project was developed as part of a research internship at LANNBIO-CINVESTAV.

---

## Notes
 
- Users can modify descriptors or model architectures to explore new material systems.  

---