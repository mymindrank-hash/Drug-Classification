import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd

sample = pd.DataFrame(
    [
        {
            "Age": 30,
            "Sex": "F",
            "BP": "NORMAL",
            "Cholesterol": "NORMAL",
            "Na_to_K": 15.0,
        }
    ]
)

# Original scikit-learn pipeline
model = joblib.load("model/drug_model.joblib")
sk_pred = model.predict(sample)[0]
sk_probs = model.predict_proba(sample)[0]
print("sklearn:", sk_pred, sk_probs)

# ONNX Runtime, same inputs/types the browser will send
sess = rt.InferenceSession("model/drug_model.onnx")
feeds = {
    "Age": np.array([[30]], dtype=np.int64),
    "Sex": np.array([["F"]], dtype=object),
    "BP": np.array([["NORMAL"]], dtype=object),
    "Cholesterol": np.array([["NORMAL"]], dtype=object),
    "Na_to_K": np.array([[15.0]], dtype=np.float64),
}
label, probs = sess.run(None, feeds)
print("onnx:   ", label[0], probs)

assert sk_pred == label[0], f"Prediction mismatch: {sk_pred} vs {label[0]}"
np.testing.assert_allclose(sk_probs, probs[0], atol=1e-5)
print("Verification passed! Scikit-learn and ONNX models produce identical outputs.")
