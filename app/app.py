from flask import Flask, request, jsonify
import mlflow.sklearn, numpy as np, os

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
model = mlflow.sklearn.load_model(MODEL_DIR)
N = int(model.n_features_in_)

app = Flask(__name__)

@app.get("/")
def home():
    return f"OK - Model loaded. n_features={N}"

@app.get("/schema")
def schema():
    return {"n_features": N}

@app.post("/predict")
def predict():
    data = request.get_json(force=True)
    x = np.array(data["features"], dtype=float).reshape(1,-1)
    if x.shape[1] != N:
        return jsonify({"error": f"need {N} features, got {x.shape[1]}"}), 400
    y = int(model.predict(x)[0])
    prob = float(model.predict_proba(x)[0][-1]) if hasattr(model, "predict_proba") else None
    return jsonify({"prediction": y, "prob_pos": prob})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
