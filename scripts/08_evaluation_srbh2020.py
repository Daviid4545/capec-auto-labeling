"""
08_evaluation_srbh2020.py

Evaluacion comun de tecnicas de etiquetado automatico sobre SR-BH 2020.

Objetivo:
    Centralizar y homogeneizar la evaluacion final de todas las tecnicas:
        - Reglas / heuristicas
        - Embeddings / similitud
        - Supervision debil / distante
        - LLM zero-shot
        - LLM few-shot
        - Humano en el bucle simulado con budgets 50, 100 y 200

Diseno:
    - No entrena modelos.
    - No modifica predicciones previas.
    - Lee los CSV generados por los scripts 03, 04, 05, 06 y 07.
    - Calcula metricas multietiqueta de forma homogenea.
    - Distingue entre:
        1) comparacion principal sobre eval_set completo;
        2) comparacion LLM sobre llm_eval_subset_srbh2020;
        3) analisis de sensibilidad HITL por presupuesto.
    - Trata HITL como simulacion de revision humana asistida, no como tecnica
      automatica pura.

Uso:
    python 08_evaluation_srbh2020.py sampling_srbh2020/eval_set.csv

Dependencias:
    python -m pip install pandas numpy scikit-learn

Salidas en ./evaluation_srbh2020/:
    - method_catalog.csv
    - overall_metrics.csv
    - main_results_table.csv
    - eval_set_comparison_table.csv
    - llm_comparison_table.csv
    - hitl_budget_comparison_table.csv
    - metric_deltas.csv
    - per_label_metrics.csv
    - confusion_matrices_per_label.csv
    - agreement_between_methods_eval_set.csv
    - evaluation_summary.txt

Metricas:
    - precision micro/macro
    - recall micro/macro
    - F1 micro/macro
    - Hamming loss
    - Jaccard samples
    - subset accuracy
    - Cohen kappa flat
    - Cohen kappa medio por etiqueta
    - precision/recall/F1/kappa por etiqueta
    - TN/FP/FN/TP por etiqueta
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
        accuracy_score,
        cohen_kappa_score,
        f1_score,
        hamming_loss,
        jaccard_score,
        precision_score,
        recall_score,
        multilabel_confusion_matrix,
    )
except ModuleNotFoundError:
    print("ERROR: falta scikit-learn.")
    print("Ejecuta: python -m pip install scikit-learn")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------
OUT_DIR = Path("evaluation_srbh2020")
PRED_PREFIX = "pred__"

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

METHODS = [
    {
        "method": "rules",
        "display_name": "Reglas / heuristicas",
        "family": "reglas_heuristicas",
        "scope": "eval_set",
        "comparison_group": "eval_set_main",
        "is_main_result": True,
        "is_automatic": True,
        "path": "rules_labeling_srbh2020/predictions_rules.csv",
        "notes": "Baseline interpretable basada en reglas manuales.",
    },
    {
        "method": "embeddings",
        "display_name": "Similaridad / embeddings",
        "family": "similaridad_embeddings",
        "scope": "eval_set",
        "comparison_group": "eval_set_main",
        "is_main_result": True,
        "is_automatic": True,
        "path": "embeddings_labeling_srbh2020/predictions_embeddings.csv",
        "notes": "Modelo de embeddings congelado con umbrales calibrados en train_set.",
    },
    {
        "method": "weak_supervision",
        "display_name": "Supervision debil / distante",
        "family": "supervision_debil",
        "scope": "eval_set",
        "comparison_group": "eval_set_main",
        "is_main_result": True,
        "is_automatic": True,
        "path": "weak_supervision_srbh2020/predictions_weak_supervision.csv",
        "notes": "Agregacion ponderada de fuentes debiles.",
    },
    {
        "method": "llm_zero_shot",
        "display_name": "LLM zero-shot",
        "family": "llm",
        "scope": "llm_eval_subset",
        "comparison_group": "llm_subset",
        "is_main_result": False,
        "is_automatic": True,
        "path": "llm_labeling_srbh2020/predictions_llm_zero_shot.csv",
        "notes": "LLM local sin ejemplos, evaluado sobre subconjunto LLM.",
    },
    {
        "method": "llm_few_shot",
        "display_name": "LLM few-shot",
        "family": "llm",
        "scope": "llm_eval_subset",
        "comparison_group": "llm_subset",
        "is_main_result": True,
        "is_automatic": True,
        "path": "llm_labeling_srbh2020/predictions_llm_few_shot.csv",
        "notes": "LLM local con ejemplos de train_set; variante principal de la familia LLM.",
    },
    {
        "method": "hitl_budget_50",
        "display_name": "Humano en el bucle simulado, budget 50",
        "family": "hitl_simulado",
        "scope": "eval_set",
        "comparison_group": "hitl_budget",
        "is_main_result": False,
        "is_automatic": False,
        "path": "human_in_the_loop_srbh2020/budget_50/predictions_human_in_the_loop.csv",
        "notes": "Revision humana simulada con oracle sobre 50 casos priorizados.",
    },
    {
        "method": "hitl_budget_100",
        "display_name": "Humano en el bucle simulado, budget 100",
        "family": "hitl_simulado",
        "scope": "eval_set",
        "comparison_group": "hitl_budget",
        "is_main_result": True,
        "is_automatic": False,
        "path": "human_in_the_loop_srbh2020/budget_100/predictions_human_in_the_loop.csv",
        "notes": "Revision humana simulada con oracle sobre 100 casos priorizados; escenario principal HITL.",
    },
    {
        "method": "hitl_budget_200",
        "display_name": "Humano en el bucle simulado, budget 200",
        "family": "hitl_simulado",
        "scope": "eval_set",
        "comparison_group": "hitl_budget",
        "is_main_result": False,
        "is_automatic": False,
        "path": "human_in_the_loop_srbh2020/budget_200/predictions_human_in_the_loop.csv",
        "notes": "Revision humana simulada con oracle sobre 200 casos priorizados.",
    },
]


# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------
def safe_col(label):
    return PRED_PREFIX + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def validate_true_labels(df, file_name):
    missing = [label for label in ALL_LABELS if label not in df.columns]
    if missing:
        raise ValueError(f"Faltan etiquetas originales en {file_name}: {missing}")


def validate_pred_labels(df, file_name):
    missing = [safe_col(label) for label in ALL_LABELS if safe_col(label) not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de prediccion en {file_name}: {missing}")


def read_prediction_file(path, method_name, required=False):
    path = Path(path)

    if not path.exists():
        if required:
            raise FileNotFoundError(f"No existe archivo requerido para {method_name}: {path}")
        print(f"Aviso: no existe {method_name}: {path}. Se omite.")
        return None

    df = pd.read_csv(path, low_memory=False)
    validate_true_labels(df, str(path))
    validate_pred_labels(df, str(path))
    return df


def get_y_true_pred(df):
    y_true = df[ALL_LABELS].fillna(0).astype(int).values
    y_pred = df[[safe_col(label) for label in ALL_LABELS]].fillna(0).astype(int).values

    normal_idx = ALL_LABELS.index(NORMAL_LABEL)

    # Si una fila no tiene ninguna etiqueta predicha, la marcamos como Normal.
    empty_pred = y_pred.sum(axis=1) == 0
    if empty_pred.any():
        y_pred[empty_pred, normal_idx] = 1

    # Si Normal aparece junto con ataque, se elimina Normal.
    attack_sum = y_pred[:, 1:].sum(axis=1)
    conflict = (y_pred[:, normal_idx] == 1) & (attack_sum > 0)
    if conflict.any():
        y_pred[conflict, normal_idx] = 0

    return y_true, y_pred


def safe_kappa(y_true, y_pred):
    try:
        value = cohen_kappa_score(y_true, y_pred)
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def compute_overall_metrics(y_true, y_pred):
    per_label_kappas = [
        safe_kappa(y_true[:, i], y_pred[:, i])
        for i in range(y_true.shape[1])
    ]

    return {
        "rows": int(y_true.shape[0]),
        "labels": int(y_true.shape[1]),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "jaccard_samples": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "cohen_kappa_flat": safe_kappa(y_true.ravel(), y_pred.ravel()),
        "cohen_kappa_label_mean": float(np.mean(per_label_kappas)),
        "avg_true_labels_per_row": float(y_true.sum(axis=1).mean()),
        "avg_pred_labels_per_row": float(y_pred.sum(axis=1).mean()),
    }


def compute_per_label_metrics(y_true, y_pred):
    rows = []

    for i, label in enumerate(ALL_LABELS):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        rows.append({
            "label": label,
            "support": int(yt.sum()),
            "predicted": int(yp.sum()),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "cohen_kappa": safe_kappa(yt, yp),
        })

    return rows


def compute_confusion_rows(y_true, y_pred):
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    rows = []

    for i, label in enumerate(ALL_LABELS):
        tn, fp, fn, tp = mcm[i].ravel()

        rows.append({
            "label": label,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        })

    return rows


def prediction_sets(y_pred):
    sets = []

    for row in y_pred:
        active = {ALL_LABELS[i] for i, value in enumerate(row) if int(value) == 1}
        attacks = {label for label in active if label != NORMAL_LABEL}
        sets.append(attacks if attacks else {NORMAL_LABEL})

    return sets


def avg_jaccard_between_sets(sets_a, sets_b):
    scores = []

    for a, b in zip(sets_a, sets_b):
        union = a | b
        scores.append(len(a & b) / len(union) if union else 1.0)

    return float(np.mean(scores)) if scores else 0.0


def exact_match_between_sets(sets_a, sets_b):
    return float(np.mean([a == b for a, b in zip(sets_a, sets_b)])) if sets_a else 0.0


def compute_agreement(method_predictions):
    rows = []
    methods = list(method_predictions.keys())

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1 = methods[i]
            m2 = methods[j]
            y1 = method_predictions[m1]
            y2 = method_predictions[m2]

            if y1.shape != y2.shape:
                continue

            s1 = prediction_sets(y1)
            s2 = prediction_sets(y2)

            rows.append({
                "method_a": m1,
                "method_b": m2,
                "rows": int(y1.shape[0]),
                "avg_jaccard_agreement": avg_jaccard_between_sets(s1, s2),
                "exact_match_agreement": exact_match_between_sets(s1, s2),
                "cohen_kappa_flat": safe_kappa(y1.ravel(), y2.ravel()),
            })

    return rows


def round_float(value):
    if isinstance(value, float):
        return round(value, 6)
    return value


def write_dict_rows(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: round_float(row.get(key, ""))
                for key in fieldnames
            })


def write_dataframe(path, df):
    df_out = df.copy()

    for col in df_out.columns:
        if pd.api.types.is_float_dtype(df_out[col]):
            df_out[col] = df_out[col].round(6)

    df_out.to_csv(path, index=False)


def make_table(df, columns):
    existing = [col for col in columns if col in df.columns]
    return df[existing].copy()


def add_delta_columns(df, baseline_method, metrics):
    """
    Anade deltas frente a un metodo base dentro del mismo DataFrame/scope.
    """
    out = df.copy()

    baseline = out[out["method"] == baseline_method]

    if baseline.empty:
        for metric in metrics:
            out[f"delta_vs_{baseline_method}_{metric}"] = np.nan
        return out

    baseline_row = baseline.iloc[0]

    for metric in metrics:
        out[f"delta_vs_{baseline_method}_{metric}"] = out[metric] - baseline_row[metric]

    return out


def add_llm_delta(df):
    out = df.copy()
    baseline = out[out["method"] == "llm_zero_shot"]

    if baseline.empty:
        return out

    metrics = ["micro_f1", "macro_f1", "hamming_loss", "jaccard_samples", "subset_accuracy"]
    baseline_row = baseline.iloc[0]

    for metric in metrics:
        out[f"delta_vs_zero_shot_{metric}"] = out[metric] - baseline_row[metric]

    return out


def extract_budget(method_name):
    match = re.search(r"budget_(\d+)", method_name)
    return int(match.group(1)) if match else np.nan


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_file", help="Ruta a sampling_srbh2020/eval_set.csv")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Carpeta de salida")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        raise FileNotFoundError(f"No existe eval_file: {eval_path}")

    eval_df = pd.read_csv(eval_path, low_memory=False)
    validate_true_labels(eval_df, str(eval_path))

    method_catalog = pd.DataFrame(METHODS)

    overall_rows = []
    per_label_rows = []
    confusion_rows = []
    eval_set_predictions = {}

    loaded_methods = []
    skipped_methods = []

    print("Evaluando tecnicas disponibles...")

    for config in METHODS:
        method_name = config["method"]
        df = read_prediction_file(config["path"], method_name, required=False)

        if df is None:
            skipped_methods.append(method_name)
            continue

        y_true, y_pred = get_y_true_pred(df)
        overall = compute_overall_metrics(y_true, y_pred)

        row = {
            "method": method_name,
            "display_name": config["display_name"],
            "family": config["family"],
            "scope": config["scope"],
            "comparison_group": config["comparison_group"],
            "is_main_result": config["is_main_result"],
            "is_automatic": config["is_automatic"],
            "source_file": config["path"],
            "notes": config["notes"],
            **overall,
        }
        overall_rows.append(row)

        for label_row in compute_per_label_metrics(y_true, y_pred):
            per_label_rows.append({
                "method": method_name,
                "display_name": config["display_name"],
                "family": config["family"],
                "scope": config["scope"],
                "comparison_group": config["comparison_group"],
                **label_row,
            })

        for conf_row in compute_confusion_rows(y_true, y_pred):
            confusion_rows.append({
                "method": method_name,
                "display_name": config["display_name"],
                "family": config["family"],
                "scope": config["scope"],
                "comparison_group": config["comparison_group"],
                **conf_row,
            })

        if config["scope"] == "eval_set":
            eval_set_predictions[method_name] = y_pred

        loaded_methods.append(method_name)
        print(f"  OK: {method_name} ({len(df)} filas, scope={config['scope']})")

    overall_df = pd.DataFrame(overall_rows)

    if overall_df.empty:
        raise RuntimeError("No se ha podido evaluar ningun metodo. Revisa que existan los archivos de predicciones.")

    # Tablas derivadas para la memoria.
    main_columns = [
        "method",
        "display_name",
        "family",
        "scope",
        "rows",
        "micro_f1",
        "macro_f1",
        "hamming_loss",
        "jaccard_samples",
        "subset_accuracy",
        "cohen_kappa_flat",
        "notes",
    ]

    main_results_df = make_table(
        overall_df[overall_df["is_main_result"] == True].sort_values(
            by=["scope", "family", "method"]
        ),
        main_columns,
    )

    eval_set_df = overall_df[overall_df["scope"] == "eval_set"].copy()
    eval_set_df = add_delta_columns(
        eval_set_df,
        "weak_supervision",
        ["micro_f1", "macro_f1", "hamming_loss", "jaccard_samples", "subset_accuracy"],
    )
    eval_set_comparison_df = make_table(
        eval_set_df.sort_values(by="micro_f1", ascending=False),
        [
            "method",
            "display_name",
            "family",
            "rows",
            "micro_f1",
            "macro_f1",
            "hamming_loss",
            "jaccard_samples",
            "subset_accuracy",
            "cohen_kappa_flat",
            "delta_vs_weak_supervision_micro_f1",
            "delta_vs_weak_supervision_macro_f1",
            "delta_vs_weak_supervision_hamming_loss",
            "notes",
        ],
    )

    llm_df = overall_df[overall_df["comparison_group"] == "llm_subset"].copy()
    llm_df = add_llm_delta(llm_df)
    llm_comparison_df = make_table(
        llm_df.sort_values(by="micro_f1", ascending=False),
        [
            "method",
            "display_name",
            "rows",
            "micro_f1",
            "macro_f1",
            "hamming_loss",
            "jaccard_samples",
            "subset_accuracy",
            "cohen_kappa_flat",
            "delta_vs_zero_shot_micro_f1",
            "delta_vs_zero_shot_macro_f1",
            "delta_vs_zero_shot_hamming_loss",
            "notes",
        ],
    )

    hitl_df = overall_df[overall_df["comparison_group"] == "hitl_budget"].copy()
    hitl_df["budget"] = hitl_df["method"].apply(extract_budget)
    weak_baseline = overall_df[overall_df["method"] == "weak_supervision"]

    if not weak_baseline.empty:
        baseline = weak_baseline.iloc[0]
        for metric in ["micro_f1", "macro_f1", "hamming_loss", "jaccard_samples", "subset_accuracy"]:
            hitl_df[f"delta_vs_weak_supervision_{metric}"] = hitl_df[metric] - baseline[metric]

    hitl_budget_df = make_table(
        hitl_df.sort_values(by="budget"),
        [
            "method",
            "display_name",
            "budget",
            "rows",
            "micro_f1",
            "macro_f1",
            "hamming_loss",
            "jaccard_samples",
            "subset_accuracy",
            "cohen_kappa_flat",
            "delta_vs_weak_supervision_micro_f1",
            "delta_vs_weak_supervision_macro_f1",
            "delta_vs_weak_supervision_hamming_loss",
            "notes",
        ],
    )

    # Tabla larga de deltas para analisis.
    delta_rows = []

    if not weak_baseline.empty:
        base = weak_baseline.iloc[0]
        for _, row in eval_set_df.iterrows():
            for metric in ["micro_f1", "macro_f1", "hamming_loss", "jaccard_samples", "subset_accuracy"]:
                delta_rows.append({
                    "comparison": "eval_set_vs_weak_supervision",
                    "baseline_method": "weak_supervision",
                    "method": row["method"],
                    "metric": metric,
                    "baseline_value": base[metric],
                    "method_value": row[metric],
                    "delta": row[metric] - base[metric],
                })

    if not llm_df[llm_df["method"] == "llm_zero_shot"].empty:
        base = llm_df[llm_df["method"] == "llm_zero_shot"].iloc[0]
        for _, row in llm_df.iterrows():
            for metric in ["micro_f1", "macro_f1", "hamming_loss", "jaccard_samples", "subset_accuracy"]:
                delta_rows.append({
                    "comparison": "llm_few_vs_zero",
                    "baseline_method": "llm_zero_shot",
                    "method": row["method"],
                    "metric": metric,
                    "baseline_value": base[metric],
                    "method_value": row[metric],
                    "delta": row[metric] - base[metric],
                })

    metric_deltas_df = pd.DataFrame(delta_rows)

    # Acuerdo entre metodos sobre eval_set.
    agreement_rows = compute_agreement(eval_set_predictions)

    # Rutas de salida.
    method_catalog_path = out_dir / "method_catalog.csv"
    overall_path = out_dir / "overall_metrics.csv"
    main_results_path = out_dir / "main_results_table.csv"
    eval_set_table_path = out_dir / "eval_set_comparison_table.csv"
    llm_table_path = out_dir / "llm_comparison_table.csv"
    hitl_table_path = out_dir / "hitl_budget_comparison_table.csv"
    metric_deltas_path = out_dir / "metric_deltas.csv"
    per_label_path = out_dir / "per_label_metrics.csv"
    confusion_path = out_dir / "confusion_matrices_per_label.csv"
    agreement_path = out_dir / "agreement_between_methods_eval_set.csv"
    summary_path = out_dir / "evaluation_summary.txt"

    # Guardar CSV.
    write_dataframe(method_catalog_path, method_catalog)
    write_dataframe(overall_path, overall_df)
    write_dataframe(main_results_path, main_results_df)
    write_dataframe(eval_set_table_path, eval_set_comparison_df)
    write_dataframe(llm_table_path, llm_comparison_df)
    write_dataframe(hitl_table_path, hitl_budget_df)

    if metric_deltas_df.empty:
        metric_deltas_df = pd.DataFrame(columns=[
            "comparison", "baseline_method", "method", "metric",
            "baseline_value", "method_value", "delta"
        ])
    write_dataframe(metric_deltas_path, metric_deltas_df)

    write_dataframe(per_label_path, pd.DataFrame(per_label_rows))
    write_dataframe(confusion_path, pd.DataFrame(confusion_rows))
    write_dataframe(agreement_path, pd.DataFrame(agreement_rows))

    # Resumen textual.
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("RESUMEN DE EVALUACION COMUN - SR-BH 2020\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Archivo eval_set de referencia: {eval_path}\n")
        f.write(f"Filas en eval_set de referencia: {len(eval_df)}\n")
        f.write(f"Etiquetas evaluadas: {len(ALL_LABELS)} -> {ALL_LABELS}\n\n")

        f.write("--- Metodos cargados ---\n")
        for method in loaded_methods:
            f.write(f"  - {method}\n")

        if skipped_methods:
            f.write("\n--- Metodos omitidos porque no se encontro su archivo ---\n")
            for method in skipped_methods:
                f.write(f"  - {method}\n")

        f.write("\n--- Tabla principal sugerida para memoria ---\n")
        for _, row in main_results_df.iterrows():
            f.write(
                f"  - {row['method']} | scope={row['scope']} | rows={row['rows']} | "
                f"micro_f1={row['micro_f1']:.6f} | macro_f1={row['macro_f1']:.6f} | "
                f"hamming_loss={row['hamming_loss']:.6f} | jaccard={row['jaccard_samples']:.6f} | "
                f"subset_accuracy={row['subset_accuracy']:.6f} | kappa={row['cohen_kappa_flat']:.6f}\n"
            )

        f.write("\n--- Comparacion principal sobre eval_set ---\n")
        for _, row in eval_set_comparison_df.iterrows():
            f.write(
                f"  - {row['method']} | micro_f1={row['micro_f1']:.6f} | "
                f"macro_f1={row['macro_f1']:.6f} | hamming_loss={row['hamming_loss']:.6f}\n"
            )

        f.write("\n--- Comparacion LLM sobre llm_eval_subset_srbh2020 ---\n")
        for _, row in llm_comparison_df.iterrows():
            f.write(
                f"  - {row['method']} | micro_f1={row['micro_f1']:.6f} | "
                f"macro_f1={row['macro_f1']:.6f} | hamming_loss={row['hamming_loss']:.6f}\n"
            )

        f.write("\n--- Analisis HITL por presupuesto ---\n")
        for _, row in hitl_budget_df.iterrows():
            f.write(
                f"  - budget={int(row['budget'])} | micro_f1={row['micro_f1']:.6f} | "
                f"macro_f1={row['macro_f1']:.6f} | hamming_loss={row['hamming_loss']:.6f}\n"
            )

        f.write("\n--- Notas de interpretacion ---\n")
        f.write(
            "La comparacion principal entre reglas, embeddings, supervision debil y HITL "
            "se realiza sobre eval_set. Las metricas de LLM se calculan sobre "
            "llm_eval_subset_srbh2020 por coste computacional, por lo que la comparacion "
            "directa mas fuerte dentro de LLM es zero-shot frente a few-shot.\n"
        )
        f.write(
            "El escenario HITL usa simulacion oracle: las etiquetas originales corrigen "
            "las instancias priorizadas. Debe interpretarse como estimacion del impacto "
            "potencial de una revision humana dirigida, no como tecnica automatica pura.\n"
        )
        f.write(
            "Para la tabla final del TFM se recomienda usar main_results_table.csv, "
            "explicando explicitamente el scope de cada metodo.\n"
        )

        f.write("\n--- Archivos generados ---\n")
        for path in [
            method_catalog_path,
            overall_path,
            main_results_path,
            eval_set_table_path,
            llm_table_path,
            hitl_table_path,
            metric_deltas_path,
            per_label_path,
            confusion_path,
            agreement_path,
        ]:
            f.write(f"  - {path}\n")

    print()
    print("Evaluacion comun completada.")
    print(f"Catalogo de metodos: {method_catalog_path}")
    print(f"Metricas globales completas: {overall_path}")
    print(f"Tabla principal TFM: {main_results_path}")
    print(f"Comparacion eval_set: {eval_set_table_path}")
    print(f"Comparacion LLM: {llm_table_path}")
    print(f"Comparacion HITL budgets: {hitl_table_path}")
    print(f"Deltas: {metric_deltas_path}")
    print(f"Metricas por etiqueta: {per_label_path}")
    print(f"Matrices por etiqueta: {confusion_path}")
    print(f"Acuerdo entre tecnicas: {agreement_path}")
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()
