# Seismic Detection Model: STA/LTA and SVM

## Overview
This project develops an algorithm to analyze seismic data from Apollo lunar missions and Mars InSight mission to accurately identify seismic events. The approach combines traditional signal processing techniques (STA/LTA) with machine learning classification (SVM) to create a robust seismic detection pipeline.

## Project Objectives
- Develop an initial seismic event detection using the STA/LTA (Short-Term Average/Long-Term Average) algorithm
- Extract and normalize features from seismic data for machine learning
- Train a supervised learning model (SVM) to classify seismic vs. noise data
- Identify both cataloged and potentially uncataloged seismic events

## Datasets
The project uses two primary datasets:
- **Lunar Dataset**: Apollo 12 Grade A seismic recordings
- **Mars Dataset**: InSight mission seismic data

Data is provided in both CSV format (containing time and velocity information) and MiniSEED format (standard seismological data format).

## Methodology

### 1. Data Preparation and Exploration
- Explored and understood the data format from both CSV and MiniSEED files
- Created unified datasets in Parquet format for efficient analysis
- Merged features with labeled catalogs for supervised learning

### 2. STA/LTA Algorithm Implementation
The STA/LTA algorithm compares short-term and long-term signal averages to detect sudden changes in seismic activity:
- Short-Term Window: 400 seconds
- Long-Term Window: 4380 seconds 
- Trigger On Threshold: 3.4
- Trigger Off Threshold: 1.0

### 3. Feature Extraction
Extracted various features from the seismic data, including:
- Time-domain features: ZCR (Zero Crossing Rate), Energy, PGA (Peak Ground Acceleration), PGV (Peak Ground Velocity)
- Frequency-domain features: Dominant Frequency
- Statistical measures: SNR (Signal-to-Noise Ratio), RMS Amplitude, CAV (Cumulative Absolute Velocity)

### 4. Machine Learning Model
- Used Support Vector Machine (SVM) with RBF kernel for classification
- Applied dimensionality reduction with Kernel PCA
- Employed SMOTE for handling class imbalance
- Fine-tuned model parameters using GridSearchCV

## Results and Findings
- The SVM model achieved high performance on the training set but had difficulty generalizing to new data
- The limited number of seismic instances (77) in the training data likely contributed to this challenge
- The STA/LTA algorithm successfully flagged potential seismic events, but the SVM classifier struggled to confirm these detections

## Recommendations for Future Work
1. Include more seismic data from trusted Earth sources to improve model robustness
2. Consider alternative approaches for time-series feature preprocessing
3. Explore deep learning models (RNN, LSTM) that may better capture temporal patterns in seismic data
4. Develop a more sophisticated data augmentation strategy for seismic events

## Installation and Usage
1. Clone this repository
2. Install dependencies: `numpy`, `pandas`, `obspy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`
3. Run the Jupyter notebook to see the complete analysis and model development

## Dependencies
- Python 3.7+
- NumPy
- pandas
- ObsPy (for reading seismic data formats)
- scikit-learn
- matplotlib
- seaborn
- scipy
- pyarrow

## Project Structure
- `Seismic Detection Model STA:LTA-SVM.ipynb`: Main notebook with complete analysis and model development
- `df_lunar.parquet`: Processed lunar data
- `df_seismic_and_normal_data.parquet`: Dataset with extracted seismic and noise segments
- `df_seismic_and_normal_data_with_newfeatures.parquet`: Dataset with additional engineered features
- `standard_scaler.pkl`: Saved StandardScaler model
- `kernel_pca_model.pkl`: Saved Kernel PCA model
- `svm_model.pkl`: Trained SVM model
