#app.py

from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model("../models/best_model.keras", compile=False)

with open("../models/class_names.json", "r") as f:
    CLASSES = json.load(f)

print("Loaded class order:", CLASSES)

# ── Load optimized thresholds saved by training script ───────────────────────
# These thresholds are found via 2D grid search on the validation set during
# training, maximizing macro F1. Using them here instead of a hardcoded 0.90
# whitelist means the app always reflects what was actually tuned per run.
#
# Priority order: downy_mildew → other_diseases → healthy
#   downy_mildew gets first priority as the primary target disease.
#   other_diseases is checked second — better to flag than miss.
#   healthy is the fallback when neither disease threshold is met.
_THRESHOLD_CONFIG_PATH = "../models/threshold_config.json"
if os.path.exists(_THRESHOLD_CONFIG_PATH):
    with open(_THRESHOLD_CONFIG_PATH) as f:
        _thresh = json.load(f)
    DOWNY_THRESHOLD = _thresh.get("downy_threshold", 0.45)
    OTHER_THRESHOLD = _thresh.get("other_threshold", 0.40)
    DOWNY_IDX       = _thresh.get("downy_class_index", 0)
    OTHER_IDX       = _thresh.get("other_class_index", 2)
    print(f"Loaded thresholds — Downy: {DOWNY_THRESHOLD}, Other: {OTHER_THRESHOLD}")
else:
    # Fallback defaults if config not found
    DOWNY_THRESHOLD = 0.45
    OTHER_THRESHOLD = 0.40
    DOWNY_IDX       = 0
    OTHER_IDX       = 2
    print("WARNING: threshold_config.json not found — using fallback defaults")

TREATMENTS = {
    "healthy":        "The plant is HEALTHY! Maintain proper watering and air circulation.",
    "downy_mildew":   "DOWNY MILDEW detected! Remove infected leaves and apply fungicide.",
    "other_diseases": "OTHER DISEASE detected. Consult an agricultural expert for accurate diagnosis.",
}

PLOTS_DIR = os.path.join("static", "plots")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_metrics():
    path = os.path.join(PLOTS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def classify_with_thresholds(preds):
    """
    Apply trained per-class thresholds to raw model output probabilities.

    Priority order:
      1. If downy_mildew confidence >= DOWNY_THRESHOLD → predict downy_mildew
      2. Else if other_diseases confidence >= OTHER_THRESHOLD → predict other_diseases
      3. Else → predict healthy

    """
    downy_score = float(preds[DOWNY_IDX])
    other_score = float(preds[OTHER_IDX])

    if downy_score >= DOWNY_THRESHOLD:
        return "downy_mildew", downy_score
    elif other_score >= OTHER_THRESHOLD:
        return "other_diseases", other_score
    else:
        healthy_score = float(preds[1])  # healthy is always index 1
        return "healthy", healthy_score


# ── routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/plots')
def plots():
    metrics = load_metrics()
    return render_template('plots.html', metrics=metrics)


@app.route('/api/metrics')
def api_metrics():
    metrics = load_metrics()
    if metrics is None:
        return jsonify({"error": "No metrics found. Run training first."}), 404
    return jsonify(metrics)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    x   = image.img_to_array(img)
    x   = np.expand_dims(x, axis=0)

    preds = model.predict(x, verbose=0)[0]

    # Use threshold-based classification
    predicted_class, decision_score = classify_with_thresholds(preds)

    # Build probability dict for all classes
    probs_dict = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}

    # Chart data
    chart_classes = ["downy_mildew", "healthy", "other_diseases"]
    chart_probs   = [
        probs_dict.get("downy_mildew",   0.0),
        probs_dict.get("healthy",        0.0),
        probs_dict.get("other_diseases", 0.0),
    ]

    # Raw argmax for logging/debugging
    raw_idx   = int(np.argmax(preds))
    raw_class = CLASSES[raw_idx]
    raw_conf  = float(preds[raw_idx])

    treatment = TREATMENTS.get(predicted_class, "Consult an agricultural expert.")

    print(f"RAW argmax: {raw_class} ({raw_conf:.1%}) | "
          f"Downy={preds[DOWNY_IDX]:.1%} (thresh={DOWNY_THRESHOLD}) | "
          f"Other={preds[OTHER_IDX]:.1%} (thresh={OTHER_THRESHOLD}) | "
          f"→ FINAL: {predicted_class}")

    all_probs = {CLASSES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASSES))}

    return jsonify({
        "prediction":     predicted_class,
        "confidence":     round(decision_score * 100, 2),
        "raw_class":      raw_class,
        "raw_confidence": round(raw_conf * 100, 2),
        "treatment":      treatment,
        "image_url":      f"/{filepath.replace(os.sep, '/')}",
        "probabilities":  chart_probs,
        "classes":        chart_classes,
        "all_probs":      all_probs,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    
    
    
    