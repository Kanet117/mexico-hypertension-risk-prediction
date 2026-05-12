# Mexico Hypertension Risk Prediction

## Project Overview
This project demonstrates an end-to-end Machine Learning pipeline to predict hypertension risk using health data from Mexico. The goal is to provide a predictive model that identifies individuals at high risk based on various clinical and demographic features.

## Architecture
The project is structured to ensure scalability, reproducibility, and clarity in the machine learning lifecycle:
- `data/`: Contains raw dataset (`Hipertension_Arterial_Mexico.csv`) and processed data.
- `src/`: Contains modularized Python scripts for different pipeline stages:
  - `data_prep.py`: Data ingestion, cleaning, imputation, and encoding.
  - `train.py`: Model training and hyperparameter tuning.
  - `evaluate.py`: Model evaluation, generating metrics and visualizations.
- `utilities/`: Contains helper scripts like `logger.py` for execution tracking.
- `models/`: Stores serialized machine learning models (e.g., Gradient Boosting, Random Forest, Logistic Regression).
- `output/`: Contains evaluation summaries, text reports, and plots.
- `docs/`: Contains project documentation, reports, and architecture details.
- `main.py`: The entry point that orchestrates the entire pipeline.

## Setup Instructions

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. **Install requirements**:
   ```bash
   pip install -r requierements.txt
   ```

## Running the Application
To run the complete pipeline (data preparation, model training, and evaluation), execute:
```bash
python main.py
```
This will process the data, train the models, generate evaluation metrics and plots in the `output/` directory, and save the trained models in the `models/` directory.

## What It Demonstrates
- **Exploratory Data Analysis (EDA)**: Statistical profiling of demographic and clinical data.
- **Data Preprocessing**: Handling missing values through mode and median imputation, capping outliers, and encoding categorical/ordinal features.
- **Modeling**: Training and comparing Logistic Regression, Random Forest, and Gradient Boosting models.
- **Evaluation**: Emphasizing metrics crucial for healthcare, specifically Recall and ROC-AUC, to minimize false negatives in medical diagnosis.