
# train SVM, KNN, MLP on hand_data.csv

import os
import json
import time
import pickle
import warnings

import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

import config


def train_models(
    data_csv=config.DATA_CSV,
    model_path=config.MODEL_PATH,
    models_dir=config.MODELS_DIR,
    preferred=config.PREFERRED_MODEL,
):
    warnings.filterwarnings("ignore")

    print("\n" + "="*55)
    print("  Training SVM, KNN, MLP")
    print("="*55)

    df = pd.read_csv(data_csv, header=None)
    X  = df.iloc[:, 1:].values.astype(float)
    y  = df.iloc[:, 0].values

    counts        = pd.Series(y).value_counts()
    valid_classes = counts[counts >= 5].index.tolist()
    dropped       = counts[counts < 5].index.tolist()
    if dropped:
        print(f"  ⚠️  Dropping classes with <5 samples: {dropped}")
        print(f"     These signs had too few detections. Try re-extracting.")
        mask = pd.Series(y).isin(valid_classes).values
        X, y = X[mask], y[mask]

    classes = sorted(set(y))
    print(f"\n  {len(y)} samples | {len(classes)} classes\n")

    le = LabelEncoder()
    y  = le.fit_transform(y)

    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                        probability=True, random_state=42)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, metric="euclidean",
                                         weights="distance")),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                                   max_iter=500, early_stopping=True,
                                   validation_fraction=0.1, random_state=42)),
        ]),
    }

    results = {}

    for name, model in models.items():
        print(f"  ▶  {name.upper()}  ", end="", flush=True)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        cv     = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)

        print(f"acc={acc:.4f}  cv={cv.mean():.4f}±{cv.std():.4f}  time={elapsed:.1f}s")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

        with open(os.path.join(models_dir, f"{name}_model.pkl"), "wb") as f:
            pickle.dump(model, f)

        results[name] = {"model": model, "acc": acc, "cv": cv}

    chosen = preferred if preferred in results else max(results, key=lambda n: results[n]["acc"])
    with open(model_path, "wb") as f:
        pickle.dump(results[chosen]["model"], f)

    # Metadata for the game repo's drift guard: lets detector_engine.py refuse
    # to load this model if src/shared/features.py has changed since training.
    meta_path = os.path.join(models_dir, "sign_model_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "feature_dim": X.shape[1],
            "chosen_model": chosen,
            "classes": le.classes_.tolist(),
            "trained_on": os.path.basename(data_csv),
        }, f, indent=2)

    print(f"\n  ✅  Saved '{chosen.upper()}' as active model → {model_path}")
    print(f"      Metadata → {meta_path}")
    print(f"      All models saved in {models_dir}/")
    return results


if __name__ == "__main__":
    train_models()
