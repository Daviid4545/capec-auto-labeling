"""
05_weak_supervision_srbh2020.py

Tecnica 4.4: Supervisión débil/distante sobre SR-BH 2020.

Diseño:
    - Carga eval_set.csv como conjunto común de evaluación.
    - Usa como fuentes débiles:
        1) predicciones de reglas;
        2) predicciones de embeddings;
        3) puntuaciones de similitud de embeddings;
        4) umbrales calibrados de embeddings.
    - Agrega las fuentes débiles por etiqueta.
    - Si Snorkel está instalado, usa LabelModel binario por etiqueta.
    - Si Snorkel no está instalado o falla, usa voting ponderado.
    - No entrena un clasificador final.
    - No usa las etiquetas originales para generar predicciones; solo para
      calcular métricas rápidas de comprobación.
    - Si ninguna etiqueta de ataque queda activa, predice 000 - Normal.

Uso recomendado:
    python 05_weak_supervision_srbh2020.py sampling_srbh2020/eval_set.csv --method weighted_vote
o bien para forzar un método específico:
    python 05_weak_supervision_srbh2020.py sampling_srbh2020/eval_set.csv ^
        --rules-file rules_labeling_srbh2020/predictions_rules.csv ^
        --embeddings-file embeddings_labeling_srbh2020/predictions_embeddings.csv ^
        --scores-file embeddings_labeling_srbh2020/embeddings_scores.csv ^
        --thresholds-file embeddings_labeling_srbh2020/embeddings_thresholds.csv

Dependencias mínimas:
    python -m pip install pandas numpy scikit-learn

Dependencia opcional:
    python -m pip install snorkel

Salidas en ./weak_supervision_srbh2020/:
    - predictions_weak_supervision.csv
    - weak_supervision_summary.txt
    - weak_supervision_lf_summary.csv
    - weak_supervision_predicted_distribution.csv
    - weak_supervision_quick_overall_metrics.csv
    - weak_supervision_quick_per_label_metrics.csv
"""

from pathlib import Path
import argparse
import csv
import re
import sys

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        hamming_loss,
        jaccard_score,
        accuracy_score,
        cohen_kappa_score,
    )
except ModuleNotFoundError:
    print("ERROR: falta scikit-learn.")
    print("Ejecuta: python -m pip install scikit-learn")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
OUT_DIR = Path("weak_supervision_srbh2020")
PRED_PREFIX = "pred__"
SCORE_PREFIX = "score__"
CONF_PREFIX = "conf__"

NORMAL_LABEL = "000 - Normal"

ATTACK_LABELS = [
    "272 - Protocol Manipulation",
    "242 - Code Injection",
    "88 - OS Command Injection",
    "126 - Path Traversal",
    "66 - SQL Injection",
    "16 - Dictionary-based Password Attack",
    "310 - Scanning for Vulnerable Software",
    "153 - Input Data Manipulation",
    "274 - HTTP Verb Tampering",
    "194 - Fake the Source of Data",
    "34 - HTTP Response Splitting",
    "33 - HTTP Request Smuggling",
]

ALL_LABELS = [NORMAL_LABEL] + ATTACK_LABELS

ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1

MAX_ATTACK_LABELS = 2
SNORKEL_EPOCHS = 300
SNORKEL_SEED = 42

# Voting ponderado usado como fallback y también como referencia.
# La lógica es simple:
#   - reglas positivas pesan más porque son trazables;
#   - embeddings positivos aportan señal semántica;
#   - puntuaciones altas de embeddings aportan apoyo adicional;
#   - predicción Normal de una técnica aporta señal negativa débil.
VOTE_WEIGHTS = {
    "rules_positive": 0.70,
    "rules_normal_negative": -0.35,
    "embeddings_positive": 0.45,
    "embeddings_normal_negative": -0.25,
    "embedding_score_high": 0.30,
    "embedding_score_low": -0.20,
}

VOTE_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------------------------
def safe_col(label):
    return PRED_PREFIX + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def safe_score_col(label):
    return SCORE_PREFIX + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def safe_conf_col(label):
    return CONF_PREFIX + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def load_csv(path, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe {name}: {path}")
    return pd.read_csv(path, low_memory=False)


def validate_eval_set(eval_df):
    missing = [label for label in ALL_LABELS if label not in eval_df.columns]
    if missing:
        raise ValueError(f"Faltan etiquetas en eval_set: {missing}")


def validate_prediction_file(df, name):
    missing = [safe_col(label) for label in ALL_LABELS if safe_col(label) not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de predicción en {name}: {missing}")


def load_thresholds(path):
    thresholds_df = load_csv(path, "thresholds-file")
    thresholds = {}

    if "label" not in thresholds_df.columns or "threshold" not in thresholds_df.columns:
        raise ValueError("embeddings_thresholds.csv debe contener columnas label y threshold")

    for _, row in thresholds_df.iterrows():
        label = str(row["label"])
        if label in ATTACK_LABELS:
            thresholds[label] = float(row["threshold"])

    missing = [label for label in ATTACK_LABELS if label not in thresholds]
    if missing:
        raise ValueError(f"Faltan umbrales para: {missing}")

    return thresholds


def validate_scores_file(scores_df):
    missing = [safe_score_col(label) for label in ATTACK_LABELS if safe_score_col(label) not in scores_df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de score en embeddings_scores.csv: {missing}")


def same_length_or_error(base_df, other_df, name):
    if len(base_df) != len(other_df):
        raise ValueError(
            f"Longitud no coincidente: eval_set tiene {len(base_df)} filas, "
            f"pero {name} tiene {len(other_df)} filas"
        )


def build_labeling_matrix(label, rules_df, embeddings_df, scores_df, thresholds):
    """
    Construye matriz de fuentes débiles para una etiqueta concreta.
    Columnas:
        0 rules_positive
        1 rules_normal_negative
        2 embeddings_positive
        3 embeddings_normal_negative
        4 embedding_score_high
        5 embedding_score_low
    """
    n = len(rules_df)
    L = np.full((n, 6), ABSTAIN, dtype=int)

    pred_col = safe_col(label)
    normal_col = safe_col(NORMAL_LABEL)
    score_col = safe_score_col(label)
    threshold = thresholds[label]

    rules_pos = rules_df[pred_col].astype(int).values == 1
    rules_norm = rules_df[normal_col].astype(int).values == 1

    emb_pos = embeddings_df[pred_col].astype(int).values == 1
    emb_norm = embeddings_df[normal_col].astype(int).values == 1

    scores = scores_df[score_col].astype(float).values

    # LF 1: reglas positivas
    L[rules_pos, 0] = POSITIVE

    # LF 2: reglas predicen Normal -> señal negativa débil
    L[rules_norm, 1] = NEGATIVE

    # LF 3: embeddings positivos
    L[emb_pos, 2] = POSITIVE

    # LF 4: embeddings predicen Normal -> señal negativa débil
    L[emb_norm, 3] = NEGATIVE

    # LF 5: score alto respecto al umbral -> positivo
    L[scores >= (threshold * 1.05), 4] = POSITIVE

    # LF 6: score claramente bajo -> negativo
    L[scores <= (threshold * 0.70), 5] = NEGATIVE

    lf_names = [
        "rules_positive",
        "rules_normal_negative",
        "embeddings_positive",
        "embeddings_normal_negative",
        "embedding_score_high",
        "embedding_score_low",
    ]

    return L, lf_names


def weighted_vote_predict(L, lf_names):
    """
    Agregación por voting ponderado.
    Devuelve:
        pred: vector 0/1
        confidence: puntuación normalizada aproximada
        raw_score: puntuación ponderada antes de normalizar
    """
    n = L.shape[0]
    raw_scores = np.zeros(n, dtype=float)

    for j, lf_name in enumerate(lf_names):
        weight = VOTE_WEIGHTS[lf_name]
        votes = L[:, j]

        if weight >= 0:
            raw_scores[votes == POSITIVE] += weight
        else:
            raw_scores[votes == NEGATIVE] += weight

    pred = (raw_scores >= VOTE_THRESHOLD).astype(int)

    # Confianza aproximada entre 0 y 1.
    # No es probabilidad calibrada, pero sirve para ordenar etiquetas por fila.
    confidence = 1.0 / (1.0 + np.exp(-raw_scores))

    return pred, confidence, raw_scores


def snorkel_predict_or_none(L):
    """
    Intenta usar Snorkel LabelModel. Si Snorkel no está instalado o la matriz
    no tiene suficiente información, devuelve None.
    """
    try:
        from snorkel.labeling.model import LabelModel
    except ModuleNotFoundError:
        return None

    # Snorkel no aporta si todas las fuentes abstienen o si no hay variedad.
    observed = L[L != ABSTAIN]
    if len(observed) == 0:
        return None

    if len(set(observed.tolist())) < 2:
        return None

    try:
        label_model = LabelModel(cardinality=2, verbose=False)
        label_model.fit(
            L_train=L,
            n_epochs=SNORKEL_EPOCHS,
            seed=SNORKEL_SEED,
            log_freq=0,
        )
        probs = label_model.predict_proba(L)
        pred = (probs[:, 1] >= 0.5).astype(int)
        confidence = probs[:, 1]
        return pred, confidence
    except Exception:
        return None


def apply_max_labels(y_pred_attack, confidence_attack):
    """
    Limita el número de etiquetas de ataque por instancia.
    Mantiene las etiquetas con mayor confianza.
    """
    y_limited = np.zeros_like(y_pred_attack, dtype=int)

    for i in range(y_pred_attack.shape[0]):
        active = np.where(y_pred_attack[i] == 1)[0]

        if len(active) <= MAX_ATTACK_LABELS:
            y_limited[i, active] = 1
            continue

        active_sorted = sorted(
            active,
            key=lambda j: confidence_attack[i, j],
            reverse=True,
        )

        for j in active_sorted[:MAX_ATTACK_LABELS]:
            y_limited[i, j] = 1

    return y_limited


def compute_quick_metrics(eval_df, y_pred_all):
    y_true = eval_df[ALL_LABELS].astype(int).values

    overall = {
        "rows": len(eval_df),
        "micro_f1": f1_score(y_true, y_pred_all, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred_all, average="macro", zero_division=0),
        "micro_precision": precision_score(y_true, y_pred_all, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred_all, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred_all, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred_all, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred_all),
        "jaccard_samples": jaccard_score(y_true, y_pred_all, average="samples", zero_division=0),
        "subset_accuracy": accuracy_score(y_true, y_pred_all),
    }

    per_label = []

    for i, label in enumerate(ALL_LABELS):
        yt = y_true[:, i]
        yp = y_pred_all[:, i]

        per_label.append({
            "label": label,
            "support": int(yt.sum()),
            "predicted": int(yp.sum()),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "kappa": cohen_kappa_score(yt, yp),
        })

    return overall, per_label


def write_label_distribution(path, y_pred_all):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "predicted_count", "percentage_eval_set"])

        n = y_pred_all.shape[0]

        for i, label in enumerate(ALL_LABELS):
            count = int(y_pred_all[:, i].sum())
            pct = (count / n * 100) if n else 0
            writer.writerow([label, count, round(pct, 6)])


def write_lf_summary(path, lf_summaries):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label",
            "lf_name",
            "positive_votes",
            "negative_votes",
            "abstentions",
        ])

        for row in lf_summaries:
            writer.writerow([
                row["label"],
                row["lf_name"],
                row["positive_votes"],
                row["negative_votes"],
                row["abstentions"],
            ])


def write_overall_metrics(path, overall):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in overall.items():
            writer.writerow([key, round(value, 6) if isinstance(value, float) else value])


def write_per_label_metrics(path, per_label):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "label", "support", "predicted", "precision", "recall", "f1", "kappa"
        ])
        writer.writeheader()

        for row in per_label:
            writer.writerow({
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in row.items()
            })


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_file", help="Ruta a sampling_srbh2020/eval_set.csv")
    parser.add_argument(
        "--rules-file",
        default="rules_labeling_srbh2020/predictions_rules.csv",
        help="Predicciones generadas por 03_rules_labeling_srbh2020.py",
    )
    parser.add_argument(
        "--embeddings-file",
        default="embeddings_labeling_srbh2020/predictions_embeddings.csv",
        help="Predicciones generadas por 04_embeddings_labeling_srbh2020.py",
    )
    parser.add_argument(
        "--scores-file",
        default="embeddings_labeling_srbh2020/embeddings_scores.csv",
        help="Scores generados por 04_embeddings_labeling_srbh2020.py",
    )
    parser.add_argument(
        "--thresholds-file",
        default="embeddings_labeling_srbh2020/embeddings_thresholds.csv",
        help="Umbrales generados por 04_embeddings_labeling_srbh2020.py",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "snorkel", "weighted_vote"],
        default="auto",
        help="Metodo de agregacion",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print(f"Cargando eval_set: {args.eval_file}")
    eval_df = load_csv(args.eval_file, "eval_file")
    validate_eval_set(eval_df)

    print(f"Cargando reglas: {args.rules_file}")
    rules_df = load_csv(args.rules_file, "rules-file")
    validate_prediction_file(rules_df, "rules-file")
    same_length_or_error(eval_df, rules_df, "rules-file")

    print(f"Cargando embeddings: {args.embeddings_file}")
    embeddings_df = load_csv(args.embeddings_file, "embeddings-file")
    validate_prediction_file(embeddings_df, "embeddings-file")
    same_length_or_error(eval_df, embeddings_df, "embeddings-file")

    print(f"Cargando scores de embeddings: {args.scores_file}")
    scores_df = load_csv(args.scores_file, "scores-file")
    validate_scores_file(scores_df)
    same_length_or_error(eval_df, scores_df, "scores-file")

    print(f"Cargando umbrales: {args.thresholds_file}")
    thresholds = load_thresholds(args.thresholds_file)

    y_pred_attack = np.zeros((len(eval_df), len(ATTACK_LABELS)), dtype=int)
    confidence_attack = np.zeros((len(eval_df), len(ATTACK_LABELS)), dtype=float)

    aggregation_used = {}
    lf_summaries = []

    print("Agregando fuentes debiles por etiqueta...")

    for j, label in enumerate(ATTACK_LABELS):
        L, lf_names = build_labeling_matrix(label, rules_df, embeddings_df, scores_df, thresholds)

        for lf_idx, lf_name in enumerate(lf_names):
            lf_col = L[:, lf_idx]
            lf_summaries.append({
                "label": label,
                "lf_name": lf_name,
                "positive_votes": int((lf_col == POSITIVE).sum()),
                "negative_votes": int((lf_col == NEGATIVE).sum()),
                "abstentions": int((lf_col == ABSTAIN).sum()),
            })

        pred = None
        confidence = None
        method_used = None

        if args.method in {"auto", "snorkel"}:
            snorkel_result = snorkel_predict_or_none(L)

            if snorkel_result is not None:
                pred, confidence = snorkel_result
                method_used = "snorkel_label_model"

        if pred is None:
            pred, confidence, _raw = weighted_vote_predict(L, lf_names)
            method_used = "weighted_vote"

            if args.method == "snorkel":
                print(f"Aviso: Snorkel no disponible o no aplicable para {label}; se usa weighted_vote.")

        y_pred_attack[:, j] = pred
        confidence_attack[:, j] = confidence
        aggregation_used[label] = method_used

    # Limitar número máximo de etiquetas de ataque por instancia.
    y_pred_attack = apply_max_labels(y_pred_attack, confidence_attack)

    y_pred_normal = (y_pred_attack.sum(axis=1) == 0).astype(int).reshape(-1, 1)
    y_pred_all = np.hstack([y_pred_normal, y_pred_attack])

    overall, per_label = compute_quick_metrics(eval_df, y_pred_all)

    # -----------------------------------------------------------------------
    # Guardar salidas
    # -----------------------------------------------------------------------
    predictions_path = OUT_DIR / "predictions_weak_supervision.csv"
    summary_path = OUT_DIR / "weak_supervision_summary.txt"
    lf_summary_path = OUT_DIR / "weak_supervision_lf_summary.csv"
    distribution_path = OUT_DIR / "weak_supervision_predicted_distribution.csv"
    overall_path = OUT_DIR / "weak_supervision_quick_overall_metrics.csv"
    per_label_path = OUT_DIR / "weak_supervision_quick_per_label_metrics.csv"

    out_df = eval_df.copy()

    for i, label in enumerate(ALL_LABELS):
        out_df[safe_col(label)] = y_pred_all[:, i].astype(int)

    for j, label in enumerate(ATTACK_LABELS):
        out_df[safe_conf_col(label)] = confidence_attack[:, j]

    selected_labels = []
    for i in range(len(out_df)):
        labs = [label for j, label in enumerate(ATTACK_LABELS) if y_pred_attack[i, j] == 1]
        if not labs:
            labs = [NORMAL_LABEL]
        selected_labels.append(";".join(labs))

    out_df["weak_supervision_selected_labels"] = selected_labels
    out_df.to_csv(predictions_path, index=False)

    write_lf_summary(lf_summary_path, lf_summaries)
    write_label_distribution(distribution_path, y_pred_all)
    write_overall_metrics(overall_path, overall)
    write_per_label_metrics(per_label_path, per_label)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("RESUMEN DE SUPERVISION DEBIL / DISTANTE\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Archivo de evaluacion: {args.eval_file}\n")
        f.write(f"Archivo de reglas: {args.rules_file}\n")
        f.write(f"Archivo de embeddings: {args.embeddings_file}\n")
        f.write(f"Archivo de scores: {args.scores_file}\n")
        f.write(f"Archivo de umbrales: {args.thresholds_file}\n")
        f.write(f"Metodo solicitado: {args.method}\n")
        f.write(f"Filas evaluadas: {len(eval_df)}\n")
        f.write(f"MAX_ATTACK_LABELS: {MAX_ATTACK_LABELS}\n\n")

        f.write("--- Metodo usado por etiqueta ---\n")
        for label in ATTACK_LABELS:
            f.write(f"  - {label}: {aggregation_used[label]}\n")

        f.write("\n--- Fuentes debiles utilizadas ---\n")
        for lf_name, weight in VOTE_WEIGHTS.items():
            f.write(f"  - {lf_name}: peso {weight}\n")

        f.write("\n--- Distribucion de etiquetas predichas ---\n")
        for i, label in enumerate(ALL_LABELS):
            f.write(f"  - {label}: {int(y_pred_all[:, i].sum())}\n")

        f.write("\n--- Metricas rapidas sobre eval_set ---\n")
        for key, value in overall.items():
            if isinstance(value, float):
                f.write(f"  - {key}: {value:.6f}\n")
            else:
                f.write(f"  - {key}: {value}\n")

        f.write("\n--- Nota metodologica ---\n")
        f.write(
            "La supervision debil combina fuentes imperfectas de etiquetado, "
            "principalmente reglas, predicciones de embeddings y puntuaciones de "
            "similitud. Las etiquetas originales del dataset no se usan para generar "
            "las predicciones, solo para calcular metricas rapidas de comprobacion. "
            "Esta tecnica actua como agregador reproducible de senales debiles y no "
            "como clasificador final entrenado.\n"
        )

    print()
    print("Supervision debil completada.")
    print(f"Predicciones: {predictions_path}")
    print(f"Resumen: {summary_path}")
    print(f"Micro-F1 rapido: {overall['micro_f1']:.4f}")
    print(f"Macro-F1 rapido: {overall['macro_f1']:.4f}")
    print(f"Hamming loss rapido: {overall['hamming_loss']:.4f}")
    print(f"Jaccard rapido: {overall['jaccard_samples']:.4f}")
    print(f"Subset accuracy rapido: {overall['subset_accuracy']:.4f}")


if __name__ == "__main__":
    main()
