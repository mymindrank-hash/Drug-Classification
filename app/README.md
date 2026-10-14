---
title: Drug Classification ML CI/CD
colorFrom: blue
colorTo: green
sdk: static
pinned: false
---

# Drug Classification - ONNX Runtime Web Inference

This application provides fast, private, client-side inference for drug classification using **ONNX Runtime Web**.

### Architecture
- **Training Pipeline**: Scikit-learn Pipeline with `ColumnTransformer` (OneHotEncoder + numerical passthrough) and `RandomForestClassifier`.
- **Model Export**: Serialized to portable ONNX format via `skl2onnx` (`target_opset=17`, `zipmap=False`).
- **Client Runtime**: Browser execution with `@onnxruntime/web` using WebAssembly / WebGL backend.
- **CI/CD**: GitHub Actions automated pipeline with CML (Continuous Machine Learning) pull-request reporting and automated Static Space deployment via `huggingface_hub`.

*Note: This application is an educational machine learning demonstration, not a medical diagnostic or prescription tool.*
