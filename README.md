# Music-Genre-Classification
## Overview
This project classifies music into 10 genres using audio feature extraction and machine learning. Implements both **SVM with engineered features** and explores **CNN with spectrograms** for comparison.

## Dataset
[GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- 1,000 audio tracks
- 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
- 30-second clips

## Key Steps
1. Audio feature extraction (librosa)
2. Feature engineering (MFCCs, spectral features, tempo)
3. Data preprocessing (label encoding, standardization)
4. Model training (SVM classifier)
5. Performance evaluation

## Feature Extraction
Extracted 50+ audio features including:
- 20 MFCC coefficients (mean + std)
- Spectral centroid, bandwidth, rolloff
- Zero-crossing rate, RMS energy
- Chroma features, tempo
- Harmonic/percussive components

## Training
Model: Support Vector Machine (SVM)  
Metrics: Accuracy, Confusion Matrix  

## Results:
<img width="588" height="492" alt="cm music" src="https://github.com/user-attachments/assets/e79226ee-b21a-4544-9e61-0545e726d2c3" />

## Top Predictive Features
<img width="630" height="470" alt="imp features music" src="https://github.com/user-attachments/assets/e2df6fcd-34fc-4a68-addd-3ca757bbc480" />

## Requirements:
Python 3.12

librosa

scikit-learn

tensorflow

pandas

matplotlib

seaborn

tqdm 
