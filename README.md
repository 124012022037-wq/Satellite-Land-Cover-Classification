# Satellite Land Cover Classification

## Problem Statement
To build a machine-learning application that predicts a land-cover category from satellite spectral-band features.

## Objective
Compare classification algorithms and deploy an interactive Streamlit application for educational demonstration.

## Dataset
`dataset/Satellite_Land_Cover_700_Sample.csv` contains 700 **synthetic educational records** (70 records for each of 10 labels). Features: Blue, Green, Red, NIR, SWIR1, SWIR2 and NDVI. Target: Land_Cover.

**Important limitation:** This is synthetic practice data, not an authentic satellite image dataset or a validated remote-sensing product. Do not present its test accuracy as real-world mapping accuracy. For a research-quality project, replace it with labelled observations from a documented source such as EuroSAT and follow that dataset's format.

## Algorithms
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

The training script selects the best-performing model on a stratified 20% hold-out test split. See `results/model_comparison.csv` and `results/classification_report.csv`.

## Project Structure
```text
Satellite-Land-Cover-Classification/
├── app.py
├── train_model.py
├── model_training.ipynb
├── requirements.txt
├── README.md
├── dataset/
│   └── Satellite_Land_Cover_700_Sample.csv
├── models/
│   └── land_cover_model.joblib
├── results/
│   ├── model_comparison.csv
│   ├── classification_report.csv
│   └── confusion_matrix.png
└── screenshots/
```

## Run Locally
1. Install Python 3.10 or later.
2. Clone this repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Satellite-Land-Cover-Classification.git
   cd Satellite-Land-Cover-Classification
   ```
3. Create and activate a virtual environment (recommended).
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Train the models:
   ```bash
   python train_model.py
   ```
6. Start the application:
   ```bash
   streamlit run app.py
   ```

## Deployment
Deploy this repository on [Streamlit Community Cloud](https://share.streamlit.io/). Select `app.py` as the main file.

- **GitHub Repository URL:** Add your repository URL here.
- **Live Application URL:** Add your deployed Streamlit URL here.

## Screenshots
Add screenshots of the running Streamlit application to the `screenshots/` folder and embed them here, for example:
`![Application screenshot](screenshots/app_home.png)`

## Team
Add team member names and roles here.
