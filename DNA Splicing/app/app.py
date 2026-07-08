from flask import Flask, render_template, request
import joblib
import os
from utils import encode_dna

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# ---------------- LOAD ----------------
models = joblib.load(os.path.join(BASE_DIR, "model/models.pkl"))
X_test = joblib.load(os.path.join(BASE_DIR, "model/X_test.pkl"))
y_test = joblib.load(os.path.join(BASE_DIR, "model/y_test.pkl"))

best_model_name = max(models.keys(), key=lambda m: models[m].score(X_test, y_test))
best_model = models[best_model_name]
print("Model expects:", best_model.n_features_in_, "features")

# dataset info
accuracy = best_model.score(X_test, y_test)

dataset_info = {
    "samples": X_test.shape[0],
    "features": X_test.shape[1],
    "accuracy": round(accuracy * 100, 2)
}


# ---------------- SPLICE ----------------
def find_splice_sites(seq):
    sites = []

    for i in range(len(seq) - 1):
        pair = seq[i:i+2]

        if pair == "GT":
            sites.append({
                "type": "Donor (GT)",
                "position": i,
                "confidence": "canonical"
            })

        elif pair == "AG":
            sites.append({
                "type": "Acceptor (AG)",
                "position": i,
                "confidence": "canonical"
            })

    return sites

def gc_content(seq):
    gc = seq.count("G") + seq.count("C")
    return round((gc / len(seq)) * 100, 2)


def at_content(seq):
    at = seq.count("A") + seq.count("T")
    return round((at / len(seq)) * 100, 2)


def nucleotide_count(seq):
    return {
        "A": seq.count("A"),
        "C": seq.count("C"),
        "G": seq.count("G"),
        "T": seq.count("T")
    }

# ---------------- INTERPRET ----------------
def interpret_prediction(pred, sites):

    if "Boundary" in pred and sites:
        return "Prediction aligns with presence of canonical splice motifs (GT/AG)."

    if "Neither" in pred and not sites:
        return "No canonical splice motifs detected; classified as non-splice region."

    return "Prediction uncertain due to weak or conflicting signals."


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    dna_input = request.form.get("dna", "").upper().strip()

    if not dna_input or any(c not in "ACGT" for c in dna_input):
        return render_template("index.html", error="Enter valid DNA (A,C,G,T only)")

    encoded = encode_dna(dna_input)
    print("="*60)
    print("DNA:", dna_input)

    pred = best_model.predict([encoded])[0]
    prob = best_model.predict_proba([encoded])[0]

    print("Prediction:", pred)
    print("Classes:", best_model.classes_)
    print("Probabilities:", prob)

    # prediction
    print("Classes:", best_model.classes_)
    print("Probabilities:", best_model.predict_proba([encoded]))

    raw_pred = best_model.predict([encoded])[0]
    print("Raw prediction:", raw_pred)
    print("Classes:", best_model.classes_)
    print("Probabilities:", best_model.predict_proba([encoded]))

    label_map = {
        
        1: "Exon-Intron Boundary",
        2: "Intron-Exon Boundary",
        3: "Neither (Non-splice)"

    }

    prediction = label_map.get(int(raw_pred), str(raw_pred))

    # probabilities
    if hasattr(best_model, "predict_proba"):
        probs_array = best_model.predict_proba([encoded])[0]
        confidence = round(max(probs_array) * 100, 2)
        probs = probs_array.tolist()
        classes = best_model.classes_.tolist()
    else:
        confidence = 0
        probs = []
        classes = []

    # model comparison
    model_results = {}

    for name, model in models.items():

        pred = model.predict([encoded])[0]

        if hasattr(model, "predict_proba"):
            prob = max(model.predict_proba([encoded])[0])
        else:
            prob = 0

        model_results[name] = {
            "prediction": label_map.get(pred, str(pred)),
            "confidence": round(prob * 100, 2)
        }

    splice_sites = find_splice_sites(dna_input)

    # Calculate sequence statistics
    gc = gc_content(dna_input)
    at = at_content(dna_input)
    counts = nucleotide_count(dna_input)

    interpretation = interpret_prediction(prediction, splice_sites)

    return render_template(
        "app.html",
        prediction=prediction,
        confidence=confidence,

        dna=dna_input,

        gc=gc,
        at=at,
        counts=counts,

        sites=splice_sites,

        probs=probs,
        classes=classes,

        model_results=model_results,

        dataset_info=dataset_info,

        interpretation=interpretation,

        model=best_model_name
    )   


if __name__ == "__main__":
    app.run(debug=True)