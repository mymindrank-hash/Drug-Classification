# Drug Classification — Automated ML CI/CD with ONNX Runtime Web

A complete, production-grade Machine Learning CI/CD implementation featuring:
- **Scikit-learn** model training and preprocessing pipeline
- **ONNX export** via `skl2onnx` (`target_opset=17`, `zipmap=False`)
- **Iterative CML** (Continuous Machine Learning) for automated metric & confusion matrix reporting on Pull Requests and Commits
- **Hugging Face Static Spaces** continuous deployment via `huggingface_hub`
- **In-Browser Inference** using **ONNX Runtime Web** (zero-compute, 100% private client-side execution)

---

## 🏛️ Architecture Overview

```
                      ┌──────────────────────────────────────┐
                      │            GitHub Actions            │
                      └──────────────────┬───────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
 1. Code Formatting               2. Train & Evaluate              3. CML Reporting
 (black --check)                  (scikit-learn + skl2onnx)        (PR comment with metrics)
        │                                │                                │
        └────────────────────────────────┼────────────────────────────────┘
                                         │
                         4. Continuous Deployment (CD)
                            (huggingface_hub CLI)
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │    Hugging Face Static Space (Free)  │
                      ├──────────────────────────────────────┤
                      │ • index.html                         │
                      │ • model/drug_model.onnx              │
                      │ • results/metrics.txt                │
                      │ • results/confusion_matrix.png       │
                      │ • README.md                          │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │    Client Browser (ONNX Runtime Web) │
                      │ • Zero server compute cost           │
                      │ • Instant WebAssembly inference      │
                      │ • 100% Client-side privacy           │
                      └──────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Drug-classification/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD pipeline (Format, Train, CML, Deploy)
├── app/
│   ├── index.html                 # Modern ONNX Runtime Web frontend
│   └── README.md                  # Hugging Face Space metadata & description
├── data/
│   └── drug200.csv                # Drug classification dataset
├── model/
│   ├── drug_model.joblib          # Scikit-learn Python model
│   └── drug_model.onnx            # Serialized ONNX model for browser runtime
├── results/
│   ├── metrics.txt                # Accuracy, F1-score, and classification report
│   └── confusion_matrix.png       # Confusion matrix visualization
├── .gitignore                     # Standard Python ignore rules
├── Makefile                       # Automation commands (install, format, train, eval, test)
├── requirements.txt               # CI and local dependencies
├── train.py                       # Model training and ONNX export pipeline
├── verify_inference.py            # Consistency validation (Scikit-learn vs ONNX)
└── README.md                      # Documentation
```

---

## 🚀 Getting Started Locally

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.13)
- Make (optional, commands can be run directly with Python)

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Check Code Formatting
```bash
black --check train.py
```

### 4. Train the Model & Export to ONNX
```bash
python train.py
```
This produces:
- `model/drug_model.joblib`: Serialized Scikit-learn pipeline
- `model/drug_model.onnx`: ONNX model with full preprocessing graph
- `results/metrics.txt`: Model accuracy, F1-score, precision/recall
- `results/confusion_matrix.png`: Visual evaluation plot

### 5. Verify Parity (Scikit-learn vs ONNX)
```bash
python verify_inference.py
```

### 6. Test the Web App Locally
Serve the application locally with Python's HTTP server:
```bash
python -m http.server 8000
```
Open your browser and navigate to:
[http://localhost:8000/app/](http://localhost:8000/app/)

---

## ⚙️ Hugging Face & GitHub Actions Setup

### Step 1: Create Hugging Face Static Space
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces).
2. Click **Create new Space**.
3. Set the name (e.g., `drug-classification-lite`).
4. Select **Static** (or Static HTML) SDK.
5. Set visibility to **Public** (or Private).

### Step 2: Generate Hugging Face Access Token
1. Go to [Hugging Face Access Tokens](https://huggingface.co/settings/tokens).
2. Create a new token with **Write** permission.
3. Copy the token.

### Step 3: Configure GitHub Repository Secrets
In your GitHub repository, navigate to:
**Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add the following secrets:

| Secret Name | Value Example | Description |
|---|---|---|
| `HF_TOKEN` | `hf_xxxxxxxxxxxxxxxx` | Hugging Face Access Token with write permission |
| `HF_SPACE_ID` | `YOUR_USERNAME/drug-classification-lite` | Hugging Face username and Space name |

*Note: GitHub's built-in `GITHUB_TOKEN` is automatically provided to CML for commenting on PRs and commits.*

---

## 🔄 CI/CD Workflow Breakdown

Every push to `main` or pull request triggers `.github/workflows/ci.yml`:

| Stage | Action | Description |
|---|---|---|
| **1. Checkout** | `actions/checkout@v4` | Fetches repository source code |
| **2. Python Setup** | `actions/setup-python@v5` | Configures Python 3.11 environment |
| **3. CML Setup** | `iterative/setup-cml@v2` | Prepares Continuous Machine Learning CLI |
| **4. Dependencies** | `make install` | Installs libraries from `requirements.txt` |
| **5. Format Check** | `make format` | Verifies code standards with `black --check` |
| **6. Training** | `make train` | Fits Random Forest pipeline and exports ONNX |
| **7. Parity Check** | `make test` | Ensures Scikit-learn and ONNX output parity |
| **8. CML Report** | `cml comment create` | Posts evaluation metrics & confusion matrix plot to PR |
| **9. Deployment** | `hf upload` | Synchronizes `app/`, `model/`, and `results/` to Hugging Face Space |

---

## 🧪 Verification & Checklist

When the application is deployed to Hugging Face:
- [x] Page renders immediately without Pyodide load delay.
- [x] Status indicator changes to `Model loaded. Ready for inference.`
- [x] Five input fields (`Age`, `Sex`, `BP`, `Cholesterol`, `Na_to_K`) accept values.
- [x] Preset sample buttons load valid test cases.
- [x] Submitting form computes predicted class and confidence percentage.
- [x] Probability distribution bars display confidence across all 5 classes (`drugA`, `drugB`, `drugC`, `drugX`, `drugY`).
- [x] Predictions match Scikit-learn outputs identically.

---

## ⚠️ Disclaimer
*This project is an educational demonstration of automated ML CI/CD and in-browser edge inference. It is not intended for medical diagnosis, treatment, or clinical prescription.*
