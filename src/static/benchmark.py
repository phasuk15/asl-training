# ─────────────────────────────────────────────
#  src/static/benchmark.py  –  Compare SVM, KNN, and MLP on hand landmark data
#
#  Usage (from project root):
#      python -m src.static.benchmark
#      python -m src.static.benchmark --data data/processed/hand_data.csv --out figures/static
#      python -m src.static.benchmark --demo --out figures/static   (synthetic data, no CSV needed)
# ─────────────────────────────────────────────
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, learning_curve
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
import time

import config

# ── Style ──────────────────────────────────────────────────────────────────
PALETTE   = {"SVM": "#4C72B0", "KNN": "#DD8452", "MLP": "#55A868"}
BG        = "#F8F9FA"
GRID_COL  = "#E0E0E0"
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
})

# ── Models ─────────────────────────────────────────────────────────────────
def build_models():
    return {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=10, gamma="scale",
                           probability=True, random_state=42)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    MLPClassifier(hidden_layer_sizes=(256, 128),
                                     activation="relu", max_iter=500,
                                     early_stopping=True, random_state=42)),
        ]),
    }

# ── Data helpers ───────────────────────────────────────────────────────────
def load_data(path):
    df  = pd.read_csv(path, header=None)
    X   = df.iloc[:, 1:].values.astype(float)
    le  = LabelEncoder()
    y   = le.fit_transform(df.iloc[:, 0].values)
    return X, y, le.classes_

def make_demo_data(n_classes=5, n_per_class=200, n_features=63):
    rng = np.random.default_rng(0)
    Xs, ys = [], []
    for c in range(n_classes):
        centre = rng.uniform(-1, 1, n_features)
        Xs.append(rng.normal(centre, 0.3, (n_per_class, n_features)))
        ys.extend([c] * n_per_class)
    X = np.vstack(Xs)
    y = np.array(ys)
    classes = [chr(65 + i) for i in range(n_classes)]
    return X, y, classes

# ── Benchmarking ───────────────────────────────────────────────────────────
def run_benchmarks(X, y, classes, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models   = build_models()
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results  = {}

    print("\n" + "="*60)
    print("  SIGN LANGUAGE MODEL BENCHMARK")
    print("="*60)

    for name, model in models.items():
        print(f"\n▶  Training {name}...")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = (time.perf_counter() - t0) / len(X_test) * 1000  # ms per sample

        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

        results[name] = {
            "accuracy":    accuracy_score(y_test, y_pred),
            "f1_macro":    f1_score(y_test, y_pred, average="macro"),
            "precision":   precision_score(y_test, y_pred, average="macro"),
            "recall":      recall_score(y_test, y_pred, average="macro"),
            "cv_mean":     cv_scores.mean(),
            "cv_std":      cv_scores.std(),
            "train_time":  train_time,
            "infer_ms":    infer_time,
            "y_pred":      y_pred,
            "model":       model,
            "cv_scores":   cv_scores,
        }

        print(f"   Accuracy : {results[name]['accuracy']:.3f}")
        print(f"   F1 Macro : {results[name]['f1_macro']:.3f}")
        print(f"   CV Mean  : {results[name]['cv_mean']:.3f} ± {results[name]['cv_std']:.3f}")
        print(f"   Train    : {train_time:.2f}s  |  Infer: {infer_time:.3f}ms/sample")

    # ── Figures ────────────────────────────────────────────────────────────
    plot_metric_comparison(results, out_dir)
    plot_confusion_matrices(results, y_test, classes, out_dir)
    plot_cv_boxplot(results, out_dir)
    plot_learning_curves(models, X, y, out_dir)
    plot_speed_vs_accuracy(results, out_dir)
    plot_per_class_f1(results, y_test, classes, out_dir)
    save_summary_table(results, out_dir)

    print(f"\n✅  All figures saved to '{out_dir}/'")
    return results

# ── Figure 1: Metric comparison bar chart ─────────────────────────────────
def plot_metric_comparison(results, out_dir):
    metrics = ["accuracy", "f1_macro", "precision", "recall"]
    labels  = ["Accuracy", "F1 (Macro)", "Precision", "Recall"]
    names   = list(results.keys())
    x       = np.arange(len(metrics))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)

    for i, name in enumerate(names):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width,
                      label=name, color=PALETTE[name], zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color=GRID_COL, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_metric_comparison.png"), dpi=150)
    plt.close(fig)
    print("   Saved fig1_metric_comparison.png")

# ── Figure 2: Confusion matrices ──────────────────────────────────────────
def plot_confusion_matrices(results, y_test, classes, out_dir):
    names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor(BG)

    for ax, name in zip(axes, names):
        cm   = confusion_matrix(y_test, results[name]["y_pred"])
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=classes, yticklabels=classes,
                    ax=ax, cbar=False, linewidths=0.5)
        ax.set_title(f"{name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle("Normalised Confusion Matrices", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_confusion_matrices.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   Saved fig2_confusion_matrices.png")

# ── Figure 3: Cross-validation boxplot ────────────────────────────────────
def plot_cv_boxplot(results, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(BG)

    names  = list(results.keys())
    data   = [results[n]["cv_scores"] for n in names]
    colors = [PALETTE[n] for n in names]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("CV Accuracy")
    ax.set_title("5-Fold Cross-Validation Accuracy Distribution",
                 fontsize=13, fontweight="bold", pad=10)
    ax.yaxis.grid(True, color=GRID_COL, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_cv_boxplot.png"), dpi=150)
    plt.close(fig)
    print("   Saved fig3_cv_boxplot.png")

# ── Figure 4: Learning curves ─────────────────────────────────────────────
def plot_learning_curves(models, X, y, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.patch.set_facecolor(BG)

    for ax, (name, model) in zip(axes, models.items()):
        sizes = np.linspace(0.1, 1.0, 8)
        train_s, train_sc, val_sc = learning_curve(
            model, X, y, train_sizes=sizes, cv=5,
            scoring="accuracy", n_jobs=-1
        )
        t_mean, t_std = train_sc.mean(axis=1), train_sc.std(axis=1)
        v_mean, v_std = val_sc.mean(axis=1),   val_sc.std(axis=1)

        ax.fill_between(train_s, t_mean - t_std, t_mean + t_std,
                        alpha=0.15, color=PALETTE[name])
        ax.fill_between(train_s, v_mean - v_std, v_mean + v_std,
                        alpha=0.15, color="grey")
        ax.plot(train_s, t_mean, "o-", color=PALETTE[name], label="Train")
        ax.plot(train_s, v_mean, "s--", color="grey", label="Validation")

        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Training samples")
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, color=GRID_COL)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Learning Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_learning_curves.png"), dpi=150)
    plt.close(fig)
    print("   Saved fig4_learning_curves.png")

# ── Figure 5: Speed vs Accuracy scatter ───────────────────────────────────
def plot_speed_vs_accuracy(results, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG)

    for name, r in results.items():
        ax.scatter(r["infer_ms"], r["accuracy"],
                   color=PALETTE[name], s=200, zorder=5, label=name)
        ax.annotate(name, (r["infer_ms"], r["accuracy"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=11, fontweight="bold", color=PALETTE[name])

    ax.set_xlabel("Inference Time (ms / sample)", fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title("Speed vs Accuracy Trade-off", fontsize=13,
                 fontweight="bold", pad=10)
    ax.yaxis.grid(True, color=GRID_COL, zorder=0)
    ax.xaxis.grid(True, color=GRID_COL, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_speed_vs_accuracy.png"), dpi=150)
    plt.close(fig)
    print("   Saved fig5_speed_vs_accuracy.png")

# ── Figure 6: Per-class F1 heatmap ────────────────────────────────────────
def plot_per_class_f1(results, y_test, classes, out_dir):
    rows = {}
    for name, r in results.items():
        report = classification_report(
            y_test, r["y_pred"], target_names=classes, output_dict=True
        )
        rows[name] = {c: report[c]["f1-score"] for c in classes}

    df = pd.DataFrame(rows).T   # models as rows, classes as cols

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.2), 3.5))
    fig.patch.set_facecolor(BG)
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGn",
                vmin=0, vmax=1, linewidths=0.5,
                ax=ax, cbar_kws={"label": "F1 Score"})
    ax.set_title("Per-Class F1 Score by Model", fontsize=13,
                 fontweight="bold", pad=10)
    ax.set_xlabel("Sign Class")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_per_class_f1.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   Saved fig6_per_class_f1.png")

# ── Summary table ──────────────────────────────────────────────────────────
def save_summary_table(results, out_dir):
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":        name,
            "Accuracy":     f"{r['accuracy']:.4f}",
            "F1 (Macro)":   f"{r['f1_macro']:.4f}",
            "Precision":    f"{r['precision']:.4f}",
            "Recall":       f"{r['recall']:.4f}",
            "CV Mean":      f"{r['cv_mean']:.4f}",
            "CV Std":       f"{r['cv_std']:.4f}",
            "Train Time(s)":f"{r['train_time']:.2f}",
            "Infer(ms)":    f"{r['infer_ms']:.3f}",
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "summary_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n   Summary table → {csv_path}")
    print("\n" + df.to_string(index=False))

    # Also save as a figure
    fig, ax = plt.subplots(figsize=(13, 1.8))
    fig.patch.set_facecolor(BG)
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values, colLabels=df.columns,
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#E8EEF7")
        cell.set_edgecolor(GRID_COL)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig0_summary_table.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   Saved fig0_summary_table.png")

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  type=str, default=config.DATA_CSV,
                        help="Path to landmark CSV (label, f1..f78)")
    parser.add_argument("--out",   type=str, default=config.FIGURES_DIR,
                        help="Output directory for figures")
    parser.add_argument("--demo",  action="store_true",
                        help="Use synthetic demo data instead of real CSV")
    args = parser.parse_args()

    if args.demo:
        print("[benchmark] Using synthetic demo data (5 classes, 200 samples each)")
        X, y, classes = make_demo_data()
    else:
        if not os.path.exists(args.data):
            print(f"[benchmark] '{args.data}' not found. Use --demo to test with synthetic data.")
            exit(1)
        print(f"[benchmark] Loading data from {args.data}")
        X, y, classes = load_data(args.data)

    print(f"[benchmark] {len(y)} samples | {len(classes)} classes: {list(classes)}")
    run_benchmarks(X, y, classes, args.out)


# real data
# python -m src.static.benchmark --data data/processed/hand_data.csv --out figures/static

# with synthetic data
# python -m src.static.benchmark --demo --out figures/static
