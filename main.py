import os
import json
import time
import urllib.request
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,log_loss,confusion_matrix,
                             roc_auc_score,roc_curve,precision_recall_curve,)
import matplotlib
matplotlib.use("Agg")  # zapis do plików, bez okna
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 160,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# ======================================================================================================================
# PARAMETRY
# ======================================================================================================================
# Banknote Authentication
UCI_RAW_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
LOCAL_DATA_FILE = "data_banknote_authentication.txt"

# Podział danych
TEST_SIZE = 0.2
SEED = 123

# Manual logistic regression
LR = 0.05
EPOCHS = 2000
L2 = 0.0          # 0.0 = brak regularyzacji (np. 0.1 = lekka regularyzacja)
INIT = "zeros"    # zeros,small_random

THRESHOLD = 0.5   # Próg klasyfikacji
PRINT_EVERY = 200 # Co ile epok wypisywać logi

# Sklearn: C duże dla minimalnej regularyzacji (tylko do porównania wag)
SKLEARN_C = 1e6

# ======================================================================================================================
# POBIERANIE + WCZYTANIE
# ======================================================================================================================

def download_if_needed(url, out_path):
    if os.path.exists(out_path):
        return
    print(f"[download] Pobieram dataset z UCI do pliku: {out_path}")
    urllib.request.urlretrieve(url, out_path)
    print("[download] OK")

def load_banknote_txt(path):
    # Plik ma 5 kolumn: variance, skewness, curtosis, entropy, class
    # class jest 0/1.
    data = np.loadtxt(path, delimiter=",")
    X = data[:, :4].astype(float)
    y = data[:, 4].astype(int)
    return X, y

def add_bias_column(X):
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack([ones, X])

# ======================================================================================================================
# MANUAL LOGISTIC REGRESSION
# ======================================================================================================================

def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def binary_log_loss(y_true, p_pred, l2=0.0, w=None):
    # clipping żeby nie było log(0)
    eps = 1e-12
    p_pred = np.clip(p_pred, eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))

    # L2 (nie karzemy biasu -> w[0])
    if l2 > 0.0 and w is not None:
        loss += 0.5 * l2 * np.sum(w[1:] ** 2)

    return float(loss)

def fit_logreg_manual(X_train, y_train, lr, epochs, l2=0.0, init="zeros"):

    # Uczenie regresji logistycznej (batch gradient descent) w wersji macierzowej:
    # p = sigmoid(Xw)
    # grad = (1/n) X^T (p - y) + l2*w (bez biasu)
    # w = w - lr * grad

    n, d = X_train.shape
    if init == "zeros":
        w = np.zeros(d, dtype=float)
    else:
        rng = np.random.RandomState(SEED)
        w = 0.01 * rng.randn(d).astype(float)
    history = []

    for epoch in range(1, epochs + 1):
        z = X_train @ w
        p = sigmoid(z)

        grad = (X_train.T @ (p - y_train)) / n

        if l2 > 0.0:
            grad[1:] += l2 * w[1:]

        w = w - lr * grad

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == epochs:
            loss = binary_log_loss(y_train, p, l2=l2, w=w)
            grad_norm = float(np.linalg.norm(grad))
            print(f"[manual] epoch={epoch:5d} loss={loss:.6f} grad_norm={grad_norm:.4e}")
            history.append((epoch, loss, grad_norm))
    return w, history

def predict_proba_manual(X, w):
    return sigmoid(X @ w)

# ======================================================================================================================
# METRYKI + WYKRESY
# ======================================================================================================================

def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def plot_loss(history, out_png):
    epoki = [h[0] for h in history]
    straty = [h[1] for h in history]
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(epoki, straty, linewidth=2.0, label="Strata (trening)")
    plt.xlabel("Epoka")
    plt.ylabel("Strata (log-loss)")
    plt.grid(True, alpha=0.25)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_confusion(cm, out_png, nazwa_modelu="model"):
    plt.figure(figsize=(5.4, 4.8))
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar(label="Liczba próbek")
    plt.xticks([0, 1], ["Klasa 0", "Klasa 1"])
    plt.yticks([0, 1], ["Klasa 0", "Klasa 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Przewidziana klasa")
    plt.ylabel("Rzeczywista klasa")
    plt.grid(False)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_roc(y_true, y_proba, out_png, nazwa_modelu="model"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(fpr, tpr, linewidth=2.2, label=f"{nazwa_modelu} (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.6, label="Losowy klasyfikator")
    plt.xlabel("Odsetek fałszywie pozytywnych (FPR)")
    plt.ylabel("Odsetek prawdziwie pozytywnych (TPR)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.25)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_pr(y_true, y_proba, out_png, nazwa_modelu="model"):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(recall, precision, linewidth=2.2, label=nazwa_modelu)
    plt.xlabel("Czułość (Recall)")
    plt.ylabel("Precyzja (Precision)")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.25)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_proba_compare(p_manual, p_sklearn, out_scatter, out_hist):
    plt.figure(figsize=(6.2, 4.8))
    plt.scatter(
        p_sklearn, p_manual,
        s=38, alpha=0.75, edgecolors="none",
        label="Próbki testowe"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.8,
             label="Idealnie: manual = sklearn")

    plt.xlabel("Prawdopodobieństwo (sklearn)")
    plt.ylabel("Prawdopodobieństwo (manual)")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.25)
    plt.savefig(out_scatter, bbox_inches="tight")
    plt.close()

    diff = p_manual - p_sklearn
    plt.figure(figsize=(6.2, 4.6))
    plt.hist(diff, bins=30, alpha=0.85, label="manual - sklearn")
    plt.xlabel("Różnica prawdopodobieństw")
    plt.ylabel("Liczba próbek")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.savefig(out_hist, bbox_inches="tight")
    plt.close()

def plot_weights_compare(w_manual, clf, feature_names, out_png):
    # po standaryzacji
    # sa różne skale wiec normalizujemy wektory wag cech (bez biasu), a bias wypisujemy osobno.

    w_m = np.array(w_manual, dtype=float)
    b_m = float(w_m[0])
    w_m_feat = w_m[1:]

    w_s_feat = clf.coef_.reshape(-1).astype(float)
    b_s = float(clf.intercept_.reshape(-1)[0])

    eps = 1e-12
    w_m_norm = w_m_feat / (np.linalg.norm(w_m_feat) + eps)
    w_s_norm = w_s_feat / (np.linalg.norm(w_s_feat) + eps)

    x = np.arange(len(feature_names))
    width = 0.35

    plt.figure(figsize=(7.2, 4.6))
    plt.bar(x - width/2, w_m_norm, width, label="manual")
    plt.bar(x + width/2, w_s_norm, width, label="sklearn (C duże)")
    plt.xticks(x, feature_names, rotation=20)
    plt.xlabel("Cecha")
    plt.ylabel("Waga (znormalizowana)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    print("\n=== Bias (wyraz wolny) ===")
    print("manual bias =", b_m)
    print("sklearn intercept_ (C duże) =", b_s)

def plot_loss_compare(loss_manual_train, loss_manual_test, loss_sklearn_train, loss_sklearn_test, out_png):
    labels = ["Trening", "Test"]
    manual_vals = [loss_manual_train, loss_manual_test]
    sklearn_vals = [loss_sklearn_train, loss_sklearn_test]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6.6, 4.2))
    plt.bar(x - width/2, manual_vals, width, label="manual")
    plt.bar(x + width/2, sklearn_vals, width, label="sklearn")
    for i, v in enumerate(manual_vals):
        plt.text(i - width / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    for i, v in enumerate(sklearn_vals):
        plt.text(i + width / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    plt.xticks(x, labels)
    plt.xlabel("Zbiór")
    plt.ylabel("Strata (log-loss)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

# ======================================================================================================================
def main():
    # dataset
    download_if_needed(UCI_RAW_URL, LOCAL_DATA_FILE)
    X, y = load_banknote_txt(LOCAL_DATA_FILE)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # manual training
    X_train_b = add_bias_column(X_train_sc)
    X_test_b = add_bias_column(X_test_sc)
    w_manual, history = fit_logreg_manual(
        X_train_b, y_train, lr=LR, epochs=EPOCHS, l2=L2, init=INIT
    )

    # manual: proby + metryki train/test
    p_manual_train = predict_proba_manual(X_train_b, w_manual)
    yhat_manual_train = (p_manual_train >= THRESHOLD).astype(int)
    metrics_manual_train = compute_metrics(y_train, yhat_manual_train, p_manual_train)

    p_manual_test = predict_proba_manual(X_test_b, w_manual)
    yhat_manual_test = (p_manual_test >= THRESHOLD).astype(int)
    metrics_manual_test = compute_metrics(y_test, yhat_manual_test, p_manual_test)
    cm_manual_test = confusion_matrix(y_test, yhat_manual_test)

    # ------------------------------------------------------------------------------------------------------------------
    # sklearn: DWA MODELE
    # 1) clf_proba = domyślny -> metryki + proby (proba_scatter)
    # 2) clf_weights = C duże -> tylko wagi (żeby ograniczyć wpływ regularyzacji na współczynniki)
    # ------------------------------------------------------------------------------------------------------------------

    # 1) sklearn: do metryk i prawdopodobieństw
    clf_proba = LogisticRegression(max_iter=5000, random_state=SEED)
    clf_proba.fit(X_train_sc, y_train)

    p_sklearn_train = clf_proba.predict_proba(X_train_sc)[:, 1]
    yhat_sklearn_train = (p_sklearn_train >= THRESHOLD).astype(int)
    metrics_sklearn_train = compute_metrics(y_train, yhat_sklearn_train, p_sklearn_train)

    p_sklearn_test = clf_proba.predict_proba(X_test_sc)[:, 1]
    yhat_sklearn_test = (p_sklearn_test >= THRESHOLD).astype(int)
    metrics_sklearn_test = compute_metrics(y_test, yhat_sklearn_test, p_sklearn_test)
    cm_sklearn_test = confusion_matrix(y_test, yhat_sklearn_test)

    #2) sklearn: do wag
    clf_weights = LogisticRegression(max_iter=5000, random_state=SEED, C=SKLEARN_C)
    clf_weights.fit(X_train_sc, y_train)

    # porównanie prawdopodobieństw test (manual vs sklearn)
    proba_corr = float(np.corrcoef(p_manual_test, p_sklearn_test)[0, 1])
    proba_mae = float(np.mean(np.abs(p_manual_test - p_sklearn_test)))
    proba_rmse = float(np.sqrt(np.mean((p_manual_test - p_sklearn_test) ** 2)))

    # zapis wyników
    run_name = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("runs", run_name)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # konsola
    print("\n=== Metryki (trening) ===")
    print("manual =", metrics_manual_train)
    print("sklearn=", metrics_sklearn_train)
    print("\n=== Metryki (test) ===")
    print("manual =", metrics_manual_test)
    print("sklearn=", metrics_sklearn_test)
    print("\n=== Wagi (sklearn do metryk/prob) ===")
    print("sklearn coef_ (4 cechy)   =", clf_proba.coef_.reshape(-1))
    print("sklearn intercept_        =", clf_proba.intercept_.reshape(-1))
    print("\n=== Porównanie prawdopodobieństw (test) ===")
    print({"proba_corr": proba_corr, "proba_mae": proba_mae, "proba_rmse": proba_rmse})

    #JSON
    summary = {
        "data": {
            "source_url": UCI_RAW_URL,
            "local_file": LOCAL_DATA_FILE,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        },
        "params": {
            "test_size": TEST_SIZE,
            "seed": SEED,
            "lr": LR,
            "epochs": EPOCHS,
            "l2": L2,
            "init": INIT,
            "threshold": THRESHOLD,
            "sklearn_C_for_weights": SKLEARN_C,
        },
        "manual": {
            "metrics_train": metrics_manual_train,
            "metrics_test": metrics_manual_test,
            "weights": w_manual.tolist(),
            "confusion_matrix_test": cm_manual_test.tolist(),
        },
        "sklearn_proba": {
            "metrics_train": metrics_sklearn_train,
            "metrics_test": metrics_sklearn_test,
            "coef": clf_proba.coef_.reshape(-1).tolist(),
            "intercept": clf_proba.intercept_.reshape(-1).tolist(),
            "confusion_matrix_test": cm_sklearn_test.tolist(),
        },
        "sklearn_weights": {
            "coef": clf_weights.coef_.reshape(-1).tolist(),
            "intercept": clf_weights.intercept_.reshape(-1).tolist(),
        },
        "proba_compare_test": {
            "proba_corr": proba_corr,
            "proba_mae": proba_mae,
            "proba_rmse": proba_rmse,
        }
    }
    save_json(os.path.join(out_dir, "summary.json"), summary)

    # wykresy
    plot_loss(history, os.path.join(plots_dir, "loss_manual.png"))
    plot_confusion(cm_manual_test, os.path.join(plots_dir, "cm_manual.png"), nazwa_modelu="manual")
    plot_confusion(cm_sklearn_test, os.path.join(plots_dir, "cm_sklearn.png"), nazwa_modelu="sklearn")
    plot_roc(y_test, p_manual_test, os.path.join(plots_dir, "roc_manual.png"), nazwa_modelu="manual")
    plot_roc(y_test, p_sklearn_test, os.path.join(plots_dir, "roc_sklearn.png"), nazwa_modelu="sklearn")
    plot_pr(y_test, p_manual_test, os.path.join(plots_dir, "pr_manual.png"), nazwa_modelu="manual")
    plot_pr(y_test, p_sklearn_test, os.path.join(plots_dir, "pr_sklearn.png"), nazwa_modelu="sklearn")
    feature_names = ["variance", "skewness", "curtosis", "entropy"]

    # z modelem clf_weights (C duże)
    plot_weights_compare(w_manual, clf_weights, feature_names, os.path.join(plots_dir, "wagi_porownanie.png"))

    # z modelem clf_proba (domyślny)
    plot_proba_compare(
        p_manual_test, p_sklearn_test,
        os.path.join(plots_dir, "proba_scatter.png"),
        os.path.join(plots_dir, "proba_diff_hist.png"),
    )

    plot_loss_compare(
        metrics_manual_train["log_loss"], metrics_manual_test["log_loss"],
        metrics_sklearn_train["log_loss"], metrics_sklearn_test["log_loss"],
        os.path.join(plots_dir, "loss_porownanie.png")
    )
    print(f"\n=== Zapisano do folderu: {out_dir} ===")
    print("Pliki: summary.json + plots/*.png")

if __name__ == "__main__":
    main()
