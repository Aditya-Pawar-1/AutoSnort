from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import io
import csv
import json
import shap
import pickle
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---------------------- Paths ----------------------
MODEL_PATH = r"D:\AutoSNortCopy\XAI-AutoSnort\model_cic"
ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------- Load artifacts ----------------------
def _load_pickle(name):
    with open(os.path.join(MODEL_PATH, name), "rb") as f:
        return pickle.load(f)

try:
    model = _load_pickle("random_forest_model_cic.pkl")
    scaler_values = _load_pickle("scaler_values_cic.pkl")
    encoder = _load_pickle("encoder_cic.pkl")
    feature_names = _load_pickle("feature_names_cic.pkl")
    with open(os.path.join(MODEL_PATH, "feature_input_schema.json"), "r") as f:
        feature_schema = json.load(f)
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model files from {MODEL_PATH}\n{e}")
    model = None
    scaler_values = None
    encoder = None
    feature_names = []
    feature_schema = {}

print("=" * 50)
print("Flask App is running with the following 15 features:")
print(feature_names)
print("=" * 50)

# ---------------------- SHAP Explainer ----------------------
explainer_shap = None
if model and feature_names:
    try:
        explainer_shap = shap.Explainer(model, feature_names=feature_names)
        print("SHAP Explainer initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not initialize SHAP Explainer. {e}")

# ---------------------- Utils ----------------------
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a dataframe using saved per-column max values (paper method)."""
    df = df.reindex(columns=feature_names, fill_value=0)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        max_val = scaler_values.get(col, 0.0)
        if max_val and max_val != 0:
            df[col] = df[col] / max_val
        else:
            df[col] = 0.0
    df.fillna(0, inplace=True)
    return df

def protocol_number_to_name(proto):
    mapping = {6: "tcp", 17: "udp", 1: "icmp"}
    try:
        p = int(float(proto))
    except Exception:
        p = 6
    return mapping.get(p, "tcp")

def generate_snort_rule(predicted_class, top_feature, feature_score, protocol_num=6, sid_start=1200000):
    if str(predicted_class).upper() == "BENIGN":
        return None
    protocol = protocol_number_to_name(protocol_num)
    msg = (
        f'msg:"[AutoSnort] AI Alert: {predicted_class}. '
        f'Trigger: {top_feature} (score={feature_score:.4f})"; '
    )
    options = [msg]
    pc = str(predicted_class).lower()
    if "portscan" in pc:
        options.append("classtype:attempted-recon;")
        options.append("detection_filter: track by_src, count 20, seconds 10;")
    elif "dos" in pc:
        options.append("classtype:denial-of-service;")
        options.append("threshold: type limit, track by_src, count 1, seconds 60;")
    elif "bot" in pc or "infiltration" in pc:
        options.append("classtype:trojan-activity;")
        if protocol == "tcp":
            options.append("flow:established,to_server;")
    elif "brute force" in pc:
        options.append("classtype:attempted-admin;")
        options.append("detection_filter: track by_dst, count 10, seconds 60;")
    else:
        options.append("classtype:web-application-attack;")
        if protocol == "tcp":
            options.append("flow:established,to_server;")
    options.append(f"sid:{sid_start}; rev:1;")
    return f"alert {protocol} any any -> any any ({' '.join(options)})"

def explain_with_shap(processed_df: pd.DataFrame, class_index: int):
    """Return list of (feature, shap_value) sorted by |shap| desc for predicted class."""
    if not explainer_shap:
        return [("SHAP not initialized", 0.0)]
    try:
        exp = explainer_shap(processed_df.astype(float))
        contribs = np.asarray(exp.values)[0, :, class_index]
        pairs = sorted(zip(feature_names, contribs), key=lambda kv: abs(float(kv[1])), reverse=True)
        return [(n, float(v)) for n, v in pairs[:5]]
    except Exception as e:
        print(f"SHAP error: {e}")
        return [("SHAP Error", 0.0)]

def load_demo_samples():
    p = os.path.join(MODEL_PATH, "demo_samples_raw_by_class.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()

# ---------------------- Routes ----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if not (model and scaler_values and encoder and feature_names):
        return render_template("index.html", feature_names=[], error="Model not loaded.", show_results=False)

    demo_df = load_demo_samples()
    demo_labels = list(demo_df["Label"].unique()) if "Label" in demo_df.columns else []

    if request.method == "POST":
        # single prediction from form
        try:
            payload = {f: float(request.form.get(f)) for f in feature_names}
            protocol_from_form = int(float(request.form.get("Protocol", 6)))
        except Exception:
            return render_template("index.html",
                                   feature_names=feature_names,
                                   error="Invalid numeric input.",
                                   show_results=False,
                                   demo_labels=demo_labels)

        processed = preprocess_df(pd.DataFrame([payload]))
        pred_idx = int(model.predict(processed)[0])
        pred_label = encoder.inverse_transform([pred_idx])[0]

        proba = model.predict_proba(processed)[0]
        confidence_scores = dict(zip(encoder.classes_, proba))
        confidence = float(confidence_scores[pred_label]) * 100.0

        explanation = explain_with_shap(processed, pred_idx)
        if str(pred_label).upper() != "BENIGN" and explanation and explanation[0][0] not in ("SHAP Error", "SHAP not initialized"):
            top_feat, top_score = explanation[0]
            rule = generate_snort_rule(pred_label, top_feat, top_score, protocol_num=protocol_from_form)
        elif str(pred_label).upper() == "BENIGN":
            rule = "No rule needed for BENIGN traffic."
        else:
            rule = "Rule generation failed: SHAP explanation could not be created."

        return render_template("index.html",
                               feature_names=feature_names,
                               prediction=pred_label,
                               confidence=confidence,
                               confidence_scores=confidence_scores,
                               explanation=explanation,
                               snort_rule=rule,
                               show_results=True,
                               demo_labels=demo_labels)

    # GET
    return render_template("index.html", feature_names=feature_names, show_results=False, demo_labels=demo_labels)

# --- Use a stored demo sample (median raw per class) ---
@app.route("/demo/<label>")
def demo(label):
    demo_df = load_demo_samples()
    if demo_df.empty:
        return redirect(url_for("index"))

    row = demo_df.loc[demo_df["Label"] == label]
    if row.empty:
        return redirect(url_for("index"))

    # Drop the label column, run through same pipeline
    payload = row.drop(columns=["Label"]).iloc[0].to_dict()
    processed = preprocess_df(pd.DataFrame([payload]))
    pred_idx = int(model.predict(processed)[0])
    pred_label = encoder.inverse_transform([pred_idx])[0]
    proba = model.predict_proba(processed)[0]
    confidence_scores = dict(zip(encoder.classes_, proba))
    confidence = float(confidence_scores[pred_label]) * 100.0
    explanation = explain_with_shap(processed, pred_idx)
    if str(pred_label).upper() != "BENIGN" and explanation and explanation[0][0] not in ("SHAP Error", "SHAP not initialized"):
        top_feat, top_score = explanation[0]
        rule = generate_snort_rule(pred_label, top_feat, top_score, protocol_num=6)
    elif str(pred_label).upper() == "BENIGN":
        rule = "No rule needed for BENIGN traffic."
    else:
        rule = "Rule generation failed: SHAP explanation could not be created."

    return render_template("index.html",
                           feature_names=feature_names,
                           prediction=pred_label,
                           confidence=confidence,
                           confidence_scores=confidence_scores,
                           explanation=explanation,
                           snort_rule=rule,
                           show_results=True,
                           demo_labels=list(demo_df["Label"].unique()))

# --- Batch prediction via CSV upload ---
# CSV must contain the same 15 feature columns (RAW values). Optional column: Protocol (numeric).
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if "file" not in request.files:
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))

    df_in = pd.read_csv(file)
    protocol_col = df_in["Protocol"] if "Protocol" in df_in.columns else 6
    df_features = df_in[[c for c in feature_names if c in df_in.columns]].copy()

    # Reindex/Fill missing with 0 as in training, then preprocess
    df_features = df_features.reindex(columns=feature_names, fill_value=0)
    processed = preprocess_df(df_features)

    preds_idx = model.predict(processed)
    preds_label = encoder.inverse_transform(preds_idx)
    preds_proba = model.predict_proba(processed)

    out_rows = []
    for i in range(len(processed)):
        class_idx = int(preds_idx[i])
        label = preds_label[i]
        probs = preds_proba[i]
        confidence = float(probs[class_idx])

        # SHAP top feature
        try:
            exp = explainer_shap(processed.iloc[[i]].astype(float))
            contribs = np.asarray(exp.values)[0, :, class_idx]
            top_idx = int(np.argmax(np.abs(contribs)))
            top_feature = feature_names[top_idx]
            top_score = float(contribs[top_idx])
        except Exception:
            top_feature, top_score = "SHAP Error", 0.0

        proto_val = protocol_col[i] if isinstance(protocol_col, pd.Series) else protocol_col
        rule = generate_snort_rule(label, top_feature, top_score, protocol_num=proto_val) \
            if (label.upper() != "BENIGN" and top_feature != "SHAP Error") else ""

        row = {fn: df_in.iloc[i][fn] if fn in df_in.columns else None for fn in feature_names}
        row.update({
            "Predicted": label,
            "Confidence": round(confidence * 100.0, 4),
            "TopFeature": top_feature,
            "TopFeatureSHAP": round(top_score, 6),
            "SnortRule": rule
        })
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)

    # Return as downloadable CSV
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    mem = io.BytesIO()
    mem.write(buf.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem,
                     as_attachment=True,
                     download_name="batch_predictions_with_rules.csv",
                     mimetype="text/csv")


if __name__ == "__main__":
    app.run(debug=True)