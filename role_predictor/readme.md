# ExciPick Role Predictor

A PyTorch-based multi-label classification model that predicts the functional roles of excipients required for a pharmaceutical formulation based on the API and formulation context.

## Features

* Predicts multiple excipient functional roles simultaneously
* Multi-layer Perceptron (MLP) implemented in PyTorch
* Handles class imbalance using weighted binary cross-entropy loss
* Reports Precision, Recall, and F1 score for each role
* Supports inference for individual formulations or batch predictions

## Project Structure

```text
role_predictor/
├── build_role_dataset.py   # Build training dataset
├── model.py                # MLP model definition
├── train.py                # Train the model
├── evaluate.py             # Evaluate model performance
├── predict_roles.py        # Predict roles for new formulations
├── role_dataset.pkl        # Generated dataset
└── best_role_model.pt      # Trained model
```

## Functional Roles

The model predicts the following 13 functional roles:

* Binder
* Filler
* Disintegrant
* Lubricant
* Glidant
* Coating Agent
* Colorant
* Controlled Release
* Solvent
* Surfactant
* Plasticizer
* Alkalizing Agent
* Humectant

## Usage

### 1. Build the dataset

```bash
python role_predictor/build_role_dataset.py
```

### 2. Train the model

```bash
python role_predictor/train.py
```

### 3. Evaluate the model

```bash
python role_predictor/evaluate.py
```

### 4. Predict roles

Single formulation:

```bash
python role_predictor/predict_roles.py \
    --api U3H27498KS \
    --dose 25 \
    --unit 1 \
    --route ORAL \
    --form TABLET
```

Batch prediction:

```bash
python role_predictor/predict_roles.py --batch-from-oral 10
```

## Model

* Input: 23 features (API descriptors, dose, route, dosage form)
* Hidden layers: 128 → 64
* Output: 13 role probabilities
* Loss: `BCEWithLogitsLoss`
* Optimizer: Adam

## Requirements

* Python 3.10+
* PyTorch
* NumPy
* Pandas

Install dependencies:

```bash
pip install torch numpy pandas
```

## Output

The evaluation script reports:

* Precision
* Recall
* F1 score
* Macro F1
* Threshold sweep

The prediction script outputs:

* Predicted functional roles
* Confidence scores
* Ground truth comparison (if available)
* Precision, Recall, and F1 for the formulation
