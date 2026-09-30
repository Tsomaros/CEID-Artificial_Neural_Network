# CEID – Artificial Neural Network

An Artificial Neural Network project developed as part of the **Computer Engineering & Informatics Department (CEID)** coursework.

The project uses a feed-forward neural network to classify **human activities** from the **HAR-PUC-Rio** dataset. The workflow includes dataset preprocessing, feature standardization, one-hot encoding, neural-network training with K-Fold cross-validation, early stopping, and accuracy evaluation.

## Overview

The goal of the project is to build a multiclass classifier capable of distinguishing between five human activities:

- **Sitting down**
- **Standing up**
- **Standing**
- **Walking**
- **Sitting**

The implementation is based on **Python**, **TensorFlow/Keras**, **NumPy**, **Pandas**, **Matplotlib**, and **scikit-learn**.

## Project Structure

```text
CEID-Artificial_Neural_Network/
│
├── main.py                         # Neural network training and evaluation
├── pre_pross.py                    # Dataset preprocessing
├── dataset-HAR-PUC-Rio.csv        # Original dataset
├── new_dataset.csv                 # Preprocessed dataset
└── Project_ΥΝ_2022-23_Μέρος-Α.pdf  # Project specification
```

## Dataset

The project uses the **HAR-PUC-Rio** dataset for Human Activity Recognition.

The preprocessing pipeline:

1. Reads the original CSV dataset.
2. Converts decimal separators and CSV delimiters to a standard format.
3. Removes the `user` column.
4. Encodes the activity labels into numerical classes:
   - `sittingdown → 1`
   - `standingup → 2`
   - `standing → 3`
   - `walking → 4`
   - `sitting → 5`
5. Encodes gender as:
   - `Woman → 0`
   - `Man → 1`
6. Removes an invalid/unwanted row.
7. Saves the processed data to `new_dataset.csv`.

## Neural Network

The classifier is implemented using **Keras Sequential API**.

### Architecture

```text
Input Layer
    │
    └── 17 input features
            │
            ▼
Dense Layer
    ├── 22 neurons
    ├── ReLU activation
    └── L1 regularization
            │
            ▼
Output Layer
    ├── 5 neurons
    └── Softmax activation
```

The model is compiled using:

- **Loss:** Categorical Cross-Entropy
- **Optimizer:** SGD
- **Learning rate:** 0.001
- **Momentum:** 0.2
- **Batch size:** 500
- **Maximum epochs:** 100

The input features are standardized using `StandardScaler`, while the target classes are converted to one-hot encoded vectors.

## Training & Evaluation

The dataset is evaluated using **5-Fold Cross-Validation**.

For each fold:

1. A new neural network is initialized.
2. The training subset is used to train the model.
3. 10% of the training data is used as a validation split.
4. Early stopping is applied during training.
5. Training and validation accuracy are plotted.
6. The model is evaluated on the held-out test fold.
7. The resulting accuracy is stored for the fold.

Finally, the mean accuracy across the five folds is reported.

## Requirements

Install the required Python packages with:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow keras
```

## Usage

### 1. Preprocess the dataset

Run:

```bash
python pre_pross.py
```

This generates the processed `new_dataset.csv` file.

### 2. Train and evaluate the neural network

Run:

```bash
python main.py
```

The program will train five models using K-Fold cross-validation and display the training/validation accuracy curves for each fold.

## Technologies

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **scikit-learn**
- **Matplotlib**
- **Artificial Neural Networks**
- **Human Activity Recognition**
- **K-Fold Cross-Validation**

## Key Concepts

This project demonstrates practical use of:

- Data preprocessing and cleaning
- Feature normalization
- Categorical and one-hot encoding
- Feed-forward Artificial Neural Networks
- ReLU and Softmax activation functions
- L1 regularization
- Stochastic Gradient Descent
- Early stopping
- K-Fold cross-validation
- Multiclass classification
- Model evaluation and visualization

## Course Project

This repository contains the implementation for the CEID Artificial Neural Network project for the **2022–2023 academic year**.

The project specification is included in the repository as:

`Project_ΥΝ_2022-23_Μέρος-Α.pdf`

## Author

**Tsomaros**

GitHub: [@Tsomaros](https://github.com/Tsomaros)

## License

This project is provided for educational purposes.
