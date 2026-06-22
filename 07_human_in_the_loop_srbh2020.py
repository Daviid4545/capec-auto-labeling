"""
07_human_in_the_loop_srbh2020.py

Tecnica 4.6: Humano en el bucle sobre SR-BH 2020.

Diseño:
    - No implementa una aplicacion de anotacion completa.
    - Prioriza instancias que deberian ser revisadas por una persona.
    - Parte de una tecnica base, por defecto supervision debil.
    - Usa senales de desacuerdo entre reglas, embeddings, supervision debil
      y opcionalmente LLM.
    - Genera un CSV de candidatos para revision humana.
    - Genera una plantilla editable para que un humano introduzca etiquetas.
    - Opcionalmente simula la revision humana usando las etiquetas originales
      del dataset como "oracle" para medir el impacto potencial de revisar N casos.

Uso recomendado para generar candidatos sin simular:
    python 07_human_in_the_loop_srbh2020.py sampling_srbh2020/eval_set.csv

Uso recomendado para evaluar el impacto potencial con oracle:
    python 07_human_in_the_loop_srbh2020.py sampling_srbh2020/eval_set.csv --simulate-oracle

Por defecto, con --simulate-oracle genera tres escenarios separados:
    - budget_50
    - budget_100
    - budget_200

Para usar un unico presupuesto:
    python 07_human_in_the_loop_srbh2020.py sampling_srbh2020/eval_set.csv --simulate-oracle --budgets 100

Uso con revision manual real:
    1) Ejecutar:
       python 07_human_in_the_loop_srbh2020.py sampling_srbh2020/eval_set.csv --budgets 100
    2) Rellenar human_in_the_loop_srbh2020/budget_100/human_review_template.csv.
    3) Ejecutar:
       python 07_human_in_the_loop_srbh2020.py sampling_srbh2020/eval_set.csv --manual-review-file human_in_the_loop_srbh2020/budget_100/human_review_template.csv --budgets 100

Dependencias:
    python -m pip install pandas numpy scikit-learn

Salidas en ./human_in_the_loop_srbh2020/:
    - budget_50/
    - budget_100/
    - budget_200/
    - hitl_budget_comparison.csv

Dentro de cada carpeta budget_N:
    - human_review_candidates.csv
    - human_review_template.csv
    - predictions_human_in_the_loop.csv
    - hitl_predicted_distribution.csv
    - hitl_quick_overall_metrics.csv
    - hitl_quick_per_label_metrics.csv
    - human_in_the_loop_summary.txt

Notas:
    - Si se usa --simulate-oracle, las etiquetas originales se usan SOLO para
      simular la correccion humana de las instancias seleccionadas. Esto debe
      describirse como simulacion de revision humana, no como tecnica automatica pura.
    - Si no se usa --simulate-oracle ni --manual-review-file, el script solo
      prioriza candidatos y evalua la tecnica base sin correccion.
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
# CONFIGURACION
# ---------------------------------------------------------------------------
OUT_DIR = Path("human_in_the_loop_srbh2020")
PRED_PREFIX = "pred__"
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

DEFAULT_BUDGETS = "50,100,200"

DEFAULT_BASE_FILE = "weak_supervision_srbh2020/predictions_weak_supervision.csv"
DEFAULT_RULES_FILE = "rules_labeling_srbh2020/predictions_rules.csv"
DEFAULT_EMBEDDINGS_FILE = "embeddings_labeling_srbh2020/predictions_embeddings.csv"
DEFAULT_LLM_FEW_FILE = "llm_labeling_srbh2020/predictions_llm_few_shot.csv"
DEFAULT_LLM_ZERO_FILE = "llm_labeling_srbh2020/predictions_llm_zero_shot.csv"


# ---------------------------------------------------------------------------
# UTILIDADES DE COLUMNAS Y ETIQUETAS
# ---------------------------------------------------------------------------
def safe_col(label):
    return PRED_PREFIX + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def safe_conf_col(label):
    return CONF_PREFIX + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def validate_eval_set(df):
    missing = [label for label in ALL_LABELS if label not in df.columns]

    if missing:
        raise ValueError(f"Faltan etiquetas en eval_set: {missing}")

    if "text_input" not in df.columns:
        raise ValueError("Falta text_input en eval_set")


def validate_prediction_file(df, name):
    missing = [safe_col(label) for label in ALL_LABELS if safe_col(label) not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas de prediccion en {name}: {missing}")


def load_csv_if_exists(path, name, required=False):
    path = Path(path)

    if not path.exists():
        if required:
            raise FileNotFoundError(f"No existe {name}: {path}")
        return None

    return pd.read_csv(path, low_memory=False)


def prediction_set_from_row(row, prefix=PRED_PREFIX):
    labels = []

    for label in ALL_LABELS:
        col = safe_col(label)

        if col in row.index and int(row.get(col, 0)) == 1:
            labels.append(label)

    attacks = [label for label in labels if label != NORMAL_LABEL]

    if attacks:
        return set(attacks)

    return {NORMAL_LABEL}


def true_set_from_row(row):
    labels = []

    for label in ALL_LABELS:
        if int(row.get(label, 0)) == 1:
            labels.append(label)

    attacks = [label for label in labels if label != NORMAL_LABEL]

    if attacks:
        return set(attacks)

    return {NORMAL_LABEL}


def labels_to_string(labels):
    if not labels:
        return NORMAL_LABEL

    ordered = [label for label in ALL_LABELS if label in labels]
    return ";".join(ordered) if ordered else NORMAL_LABEL


def parse_reviewed_labels(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    raw_parts = [part.strip() for part in re.split(r"[;,]", text) if part.strip()]
    labels = []

    for part in raw_parts:
        if part in ALL_LABELS:
            labels.append(part)
            continue

        # Permitir solo CAPEC ID.
        for label in ALL_LABELS:
            if part == label.split(" - ")[0]:
                labels.append(label)
                break

    clean = []

    for label in labels:
        if label not in clean:
            clean.append(label)

    attacks = [label for label in clean if label != NORMAL_LABEL]

    if attacks:
        return set(attacks)

    return {NORMAL_LABEL}


def build_pred_matrix_from_sets(label_sets):
    matrix = np.zeros((len(label_sets), len(ALL_LABELS)), dtype=int)

    for i, label_set in enumerate(label_sets):
        if not label_set:
            label_set = {NORMAL_LABEL}

        attacks = [label for label in label_set if label != NORMAL_LABEL]

        if attacks:
            active = attacks
        else:
            active = [NORMAL_LABEL]

        for label in active:
            if label in ALL_LABELS:
                matrix[i, ALL_LABELS.index(label)] = 1

    return matrix


# ---------------------------------------------------------------------------
# ALINEACION DE PREDICCIONES
# ---------------------------------------------------------------------------
def align_full_prediction_file(eval_df, pred_df, name):
    """
    Alinea archivos de prediccion que deberian tener la misma longitud que eval_set.
    """
    if pred_df is None:
        return None

    validate_prediction_file(pred_df, name)

    if len(pred_df) != len(eval_df):
        print(f"Aviso: {name} no tiene la misma longitud que eval_set. Se intentara alinear por text_input.")
        return align_by_text_input(eval_df, pred_df, name)

    return pred_df.reset_index(drop=True)


def align_by_text_input(eval_df, pred_df, name):
    """
    Alinea predicciones parciales, por ejemplo LLM sobre llm_eval_subset.
    Se usa text_input como clave.
    """
    if pred_df is None:
        return None

    validate_prediction_file(pred_df, name)

    if "text_input" not in pred_df.columns:
        print(f"Aviso: {name} no tiene text_input; se ignora.")
        return None

    pred_cols = [safe_col(label) for label in ALL_LABELS]
    extra_cols = [col for col in pred_df.columns if col.startswith("llm_")]

    keep_cols = ["text_input"] + pred_cols + extra_cols
    partial = pred_df[keep_cols].copy()

    # Si hay duplicados de text_input, nos quedamos con el primero.
    partial = partial.drop_duplicates(subset=["text_input"], keep="first")

    aligned = eval_df[["text_input"]].merge(
        partial,
        on="text_input",
        how="left",
    )

    matched = aligned[pred_cols].notna().any(axis=1).sum()
    print(f"{name}: {matched} filas alineadas por text_input.")

    return aligned


def get_method_sets(df, method_name):
    """
    Devuelve una lista de conjuntos de etiquetas para cada fila.
    Si el metodo no tiene prediccion para una fila, devuelve None en esa posicion.
    """
    pred_cols = [safe_col(label) for label in ALL_LABELS]

    label_sets = []

    for _, row in df.iterrows():
        has_any = False

        for col in pred_cols:
            if col in row.index and not pd.isna(row.get(col)):
                has_any = True
                break

        if not has_any:
            label_sets.append(None)
            continue

        row_filled = row.copy()
        for col in pred_cols:
            if col in row_filled.index and pd.isna(row_filled[col]):
                row_filled[col] = 0

        label_sets.append(prediction_set_from_row(row_filled))

    return label_sets


# ---------------------------------------------------------------------------
# PUNTUACION PARA SELECCION HUMANA
# ---------------------------------------------------------------------------
def jaccard_distance(a, b):
    if a is None or b is None:
        return None

    if not a and not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return 1.0 - (len(a & b) / len(union))


def average_pairwise_disagreement(method_sets_for_row):
    valid_sets = [s for s in method_sets_for_row if s is not None]

    if len(valid_sets) < 2:
        return 0.0

    distances = []

    for i in range(len(valid_sets)):
        for j in range(i + 1, len(valid_sets)):
            distances.append(jaccard_distance(valid_sets[i], valid_sets[j]))

    if not distances:
        return 0.0

    return float(np.mean(distances))


def normal_attack_conflict(method_sets_for_row):
    valid_sets = [s for s in method_sets_for_row if s is not None]

    if len(valid_sets) < 2:
        return 0.0

    has_normal = any(s == {NORMAL_LABEL} for s in valid_sets)
    has_attack = any(any(label != NORMAL_LABEL for label in s) for s in valid_sets)

    return 1.0 if has_normal and has_attack else 0.0


def multi_label_signal(method_sets_for_row):
    valid_sets = [s for s in method_sets_for_row if s is not None]

    if not valid_sets:
        return 0.0

    max_labels = max(len([label for label in s if label != NORMAL_LABEL]) for s in valid_sets)

    if max_labels <= 1:
        return 0.0

    return min(max_labels / 3.0, 1.0)


def rare_predicted_label_score(base_set, predicted_label_counts, n_rows):
    attacks = [label for label in base_set if label != NORMAL_LABEL]

    if not attacks:
        return 0.0

    scores = []

    for label in attacks:
        count = predicted_label_counts.get(label, 0)
        freq = count / n_rows if n_rows else 0

        # Cuanto menos frecuente sea la etiqueta predicha, mayor prioridad.
        scores.append(1.0 - min(freq / 0.10, 1.0))

    return float(max(scores)) if scores else 0.0


def weak_confidence_uncertainty(base_df, row_idx):
    """
    Si existe conf__ por etiqueta en supervision debil, usa incertidumbre.
    Si no existe, devuelve 0.
    """
    conf_cols = [safe_conf_col(label) for label in ATTACK_LABELS]

    existing = [col for col in conf_cols if col in base_df.columns]

    if not existing:
        return 0.0

    values = []

    for col in existing:
        value = base_df.iloc[row_idx].get(col, np.nan)

        if pd.isna(value):
            continue

        try:
            values.append(float(value))
        except ValueError:
            continue

    if not values:
        return 0.0

    max_conf = max(values)

    # Incertidumbre alta si la confianza maxima esta cerca de 0.5.
    uncertainty = 1.0 - min(abs(max_conf - 0.5) * 2, 1.0)

    return float(uncertainty)


def build_review_scores(eval_df, base_df, method_sets):
    """
    Calcula puntuacion de prioridad para revision humana.
    """
    n_rows = len(eval_df)

    base_sets = method_sets["base"]
    predicted_label_counts = {}

    for label_set in base_sets:
        if label_set is None:
            continue

        for label in label_set:
            predicted_label_counts[label] = predicted_label_counts.get(label, 0) + 1

    rows = []

    for i in range(n_rows):
        sets_for_row = [
            method_sets[name][i]
            for name in method_sets
            if method_sets[name][i] is not None
        ]

        base_set = base_sets[i] if base_sets[i] is not None else {NORMAL_LABEL}

        disagreement = average_pairwise_disagreement(sets_for_row)
        conflict = normal_attack_conflict(sets_for_row)
        multi_signal = multi_label_signal(sets_for_row)
        rare_score = rare_predicted_label_score(base_set, predicted_label_counts, n_rows)
        uncertainty = weak_confidence_uncertainty(base_df, i)

        score = (
            0.45 * disagreement
            + 0.20 * conflict
            + 0.15 * uncertainty
            + 0.10 * multi_signal
            + 0.10 * rare_score
        )

        reasons = []

        if disagreement >= 0.50:
            reasons.append("desacuerdo_alto_entre_tecnicas")
        elif disagreement > 0:
            reasons.append("desacuerdo_parcial_entre_tecnicas")

        if conflict == 1.0:
            reasons.append("conflicto_normal_vs_ataque")

        if uncertainty >= 0.50:
            reasons.append("baja_confianza_o_incertidumbre")

        if multi_signal > 0:
            reasons.append("posible_multietiqueta")

        if rare_score >= 0.50:
            reasons.append("etiqueta_predicha_poco_frecuente")

        if not reasons:
            reasons.append("revision_por_presupuesto")

        rows.append({
            "row_position": i,
            "review_priority_score": score,
            "disagreement_score": disagreement,
            "normal_attack_conflict": conflict,
            "uncertainty_score": uncertainty,
            "multi_label_score": multi_signal,
            "rare_predicted_label_score": rare_score,
            "review_reason": ";".join(reasons),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# APLICACION DE REVISION HUMANA
# ---------------------------------------------------------------------------
def apply_oracle_review(eval_df, base_sets, candidate_positions):
    """
    Simula revision humana usando etiquetas originales como oracle.
    """
    updated_sets = list(base_sets)

    for pos in candidate_positions:
        row = eval_df.iloc[pos]
        updated_sets[pos] = true_set_from_row(row)

    return updated_sets


def apply_manual_review(base_sets, manual_review_file):
    review_path = Path(manual_review_file)

    if not review_path.exists():
        raise FileNotFoundError(f"No existe manual-review-file: {review_path}")

    review_df = pd.read_csv(review_path, low_memory=False)

    if "row_position" not in review_df.columns or "reviewed_labels" not in review_df.columns:
        raise ValueError("manual-review-file debe tener columnas row_position y reviewed_labels")

    updated_sets = list(base_sets)
    applied = 0

    for _, row in review_df.iterrows():
        labels = parse_reviewed_labels(row.get("reviewed_labels", ""))

        if labels is None:
            continue

        pos = int(row["row_position"])

        if pos < 0 or pos >= len(updated_sets):
            continue

        updated_sets[pos] = labels
        applied += 1

    return updated_sets, applied


# ---------------------------------------------------------------------------
# METRICAS Y SALIDAS
# ---------------------------------------------------------------------------
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


def write_distribution(path, y_pred_all):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "predicted_count", "percentage_eval_set"])

        n = y_pred_all.shape[0]

        for i, label in enumerate(ALL_LABELS):
            count = int(y_pred_all[:, i].sum())
            pct = (count / n * 100) if n else 0
            writer.writerow([label, count, round(pct, 6)])


def build_candidates_output(eval_df, review_scores, method_sets, budget):
    ranked = review_scores.sort_values(
        by=["review_priority_score", "disagreement_score", "uncertainty_score"],
        ascending=False,
    ).head(budget).copy()

    records = []

    for review_id, (_, row) in enumerate(ranked.iterrows(), start=1):
        pos = int(row["row_position"])
        eval_row = eval_df.iloc[pos]

        record = {
            "review_id": review_id,
            "row_position": pos,
            "review_priority_score": round(float(row["review_priority_score"]), 6),
            "review_reason": row["review_reason"],
            "text_input": eval_row.get("text_input", ""),
            "base_predicted_labels": labels_to_string(method_sets["base"][pos]),
        }

        for name, sets_list in method_sets.items():
            if name == "base":
                continue

            label_set = sets_list[pos]

            if label_set is None:
                record[f"{name}_predicted_labels"] = ""
            else:
                record[f"{name}_predicted_labels"] = labels_to_string(label_set)

        records.append(record)

    return pd.DataFrame(records)


def build_review_template(candidates_df):
    template = candidates_df.copy()
    template["reviewed_labels"] = ""
    template["review_notes"] = ""

    cols_first = [
        "review_id",
        "row_position",
        "reviewed_labels",
        "review_notes",
        "review_priority_score",
        "review_reason",
        "text_input",
    ]

    remaining = [col for col in template.columns if col not in cols_first]
    return template[cols_first + remaining]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def parse_budgets(value):
    """
    Convierte una cadena tipo '50,100,200' en lista ordenada de presupuestos.
    """
    budgets = []

    for part in str(value).split(","):
        part = part.strip()

        if not part:
            continue

        try:
            budget = int(part)
        except ValueError as exc:
            raise ValueError(f"Presupuesto no valido: {part}") from exc

        if budget <= 0:
            raise ValueError(f"El presupuesto debe ser positivo: {budget}")

        budgets.append(budget)

    budgets = sorted(set(budgets))

    if not budgets:
        raise ValueError("Debes indicar al menos un presupuesto valido")

    return budgets


def save_hitl_run(
    out_dir,
    eval_df,
    method_sets,
    base_sets,
    candidates_df,
    review_template_df,
    final_sets,
    y_pred_all,
    overall,
    per_label,
    args,
    budget,
    review_mode,
    applied_reviews,
):
    """
    Guarda todos los ficheros de salida de un escenario HITL concreto.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = eval_df.copy()

    for i, label in enumerate(ALL_LABELS):
        out_df[safe_col(label)] = y_pred_all[:, i].astype(int)

    out_df["hitl_selected_labels"] = [labels_to_string(s) for s in final_sets]
    out_df["hitl_reviewed"] = False

    reviewed_positions = set(candidates_df["row_position"].astype(int).tolist())

    if args.simulate_oracle or args.manual_review_file:
        out_df.loc[list(reviewed_positions), "hitl_reviewed"] = True

    candidates_path = out_dir / "human_review_candidates.csv"
    template_path = out_dir / "human_review_template.csv"
    predictions_path = out_dir / "predictions_human_in_the_loop.csv"
    distribution_path = out_dir / "hitl_predicted_distribution.csv"
    overall_path = out_dir / "hitl_quick_overall_metrics.csv"
    per_label_path = out_dir / "hitl_quick_per_label_metrics.csv"
    summary_path = out_dir / "human_in_the_loop_summary.txt"

    candidates_df.to_csv(candidates_path, index=False)
    review_template_df.to_csv(template_path, index=False)
    out_df.to_csv(predictions_path, index=False)
    write_distribution(distribution_path, y_pred_all)
    write_overall_metrics(overall_path, overall)
    write_per_label_metrics(per_label_path, per_label)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("RESUMEN DE HUMANO EN EL BUCLE\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Archivo de evaluacion: {args.eval_file}\n")
        f.write(f"Prediccion base: {args.base_file}\n")
        f.write(f"Presupuesto de revision: {budget}\n")
        f.write(f"Modo de revision: {review_mode}\n")
        f.write(f"Revisiones aplicadas: {applied_reviews}\n")
        f.write(f"Filas evaluadas: {len(eval_df)}\n")
        f.write(f"Carpeta de salida: {out_dir}\n\n")

        f.write("--- Fuentes usadas para priorizacion ---\n")
        for name in method_sets:
            f.write(f"  - {name}\n")

        f.write("\n--- Criterios de priorizacion ---\n")
        f.write("  - desacuerdo entre tecnicas\n")
        f.write("  - conflicto Normal frente a ataque\n")
        f.write("  - incertidumbre/confianza cuando esta disponible\n")
        f.write("  - posibles casos multietiqueta\n")
        f.write("  - etiquetas predichas poco frecuentes\n\n")

        f.write("--- Distribucion de etiquetas predichas tras HITL ---\n")
        for i, label in enumerate(ALL_LABELS):
            f.write(f"  - {label}: {int(y_pred_all[:, i].sum())}\n")

        f.write("\n--- Metricas rapidas sobre eval_set ---\n")
        for key, value in overall.items():
            if isinstance(value, float):
                f.write(f"  - {key}: {value:.6f}\n")
            else:
                f.write(f"  - {key}: {value}\n")

        f.write("\n--- Nota metodologica ---\n")
        if args.simulate_oracle:
            f.write(
                "Se ha utilizado --simulate-oracle, por lo que las etiquetas originales "
                "se han usado para simular la revision humana de las instancias "
                "priorizadas. Este resultado debe interpretarse como una estimacion "
                "del impacto potencial de revisar manualmente un numero limitado de "
                "casos, no como una tecnica automatica pura.\n"
            )
        elif args.manual_review_file:
            f.write(
                "Se ha aplicado un archivo de revision manual. Las etiquetas revisadas "
                "sobrescriben la prediccion base solo en las instancias indicadas.\n"
            )
        else:
            f.write(
                "No se han aplicado correcciones. El script ha generado candidatos "
                "priorizados y plantilla de revision, manteniendo las predicciones base.\n"
            )

    return {
        "budget": budget,
        "output_dir": str(out_dir),
        "review_mode": review_mode,
        "applied_reviews": applied_reviews,
        **overall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_file", help="Ruta a sampling_srbh2020/eval_set.csv")
    parser.add_argument("--base-file", default=DEFAULT_BASE_FILE, help="Predicciones base, por defecto supervision debil")
    parser.add_argument("--rules-file", default=DEFAULT_RULES_FILE, help="Predicciones de reglas")
    parser.add_argument("--embeddings-file", default=DEFAULT_EMBEDDINGS_FILE, help="Predicciones de embeddings")
    parser.add_argument("--llm-few-shot-file", default=DEFAULT_LLM_FEW_FILE, help="Predicciones LLM few-shot, opcional")
    parser.add_argument("--llm-zero-shot-file", default=DEFAULT_LLM_ZERO_FILE, help="Predicciones LLM zero-shot, opcional")
    parser.add_argument(
        "--budgets",
        default=DEFAULT_BUDGETS,
        help="Presupuestos separados por coma. Por defecto: 50,100,200",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Compatibilidad: si se indica, equivale a --budgets N",
    )
    parser.add_argument("--simulate-oracle", action="store_true", help="Simula revision humana usando etiquetas originales")
    parser.add_argument("--manual-review-file", default=None, help="CSV de revision manual con row_position y reviewed_labels")
    args = parser.parse_args()

    if args.simulate_oracle and args.manual_review_file:
        raise ValueError("No uses simultaneamente --simulate-oracle y --manual-review-file")

    if args.budget is not None:
        budgets = [args.budget]
    else:
        budgets = parse_budgets(args.budgets)

    if args.manual_review_file and len(budgets) != 1:
        raise ValueError("Con --manual-review-file usa un unico presupuesto, por ejemplo --budgets 100")

    OUT_DIR.mkdir(exist_ok=True)

    print(f"Cargando eval_set: {args.eval_file}")
    eval_df = pd.read_csv(args.eval_file, low_memory=False).reset_index(drop=True)
    validate_eval_set(eval_df)

    print(f"Cargando predicciones base: {args.base_file}")
    base_df = load_csv_if_exists(args.base_file, "base-file", required=True)
    base_df = align_full_prediction_file(eval_df, base_df, "base-file")

    print("Cargando fuentes auxiliares...")
    rules_df = align_full_prediction_file(
        eval_df,
        load_csv_if_exists(args.rules_file, "rules-file", required=False),
        "rules-file",
    )
    embeddings_df = align_full_prediction_file(
        eval_df,
        load_csv_if_exists(args.embeddings_file, "embeddings-file", required=False),
        "embeddings-file",
    )

    llm_few_df = align_by_text_input(
        eval_df,
        load_csv_if_exists(args.llm_few_shot_file, "llm-few-shot-file", required=False),
        "llm-few-shot-file",
    )
    llm_zero_df = align_by_text_input(
        eval_df,
        load_csv_if_exists(args.llm_zero_shot_file, "llm-zero-shot-file", required=False),
        "llm-zero-shot-file",
    )

    method_sets = {
        "base": get_method_sets(base_df, "base"),
    }

    if rules_df is not None:
        method_sets["rules"] = get_method_sets(rules_df, "rules")

    if embeddings_df is not None:
        method_sets["embeddings"] = get_method_sets(embeddings_df, "embeddings")

    if llm_few_df is not None:
        method_sets["llm_few_shot"] = get_method_sets(llm_few_df, "llm_few_shot")

    if llm_zero_df is not None:
        method_sets["llm_zero_shot"] = get_method_sets(llm_zero_df, "llm_zero_shot")

    print("Calculando prioridades de revision humana...")
    review_scores = build_review_scores(eval_df, base_df, method_sets)
    base_sets = method_sets["base"]

    comparison_rows = []

    for budget in budgets:
        print()
        print(f"Procesando escenario HITL con presupuesto {budget}...")

        scenario_dir = OUT_DIR / f"budget_{budget}"

        candidates_df = build_candidates_output(eval_df, review_scores, method_sets, budget)
        review_template_df = build_review_template(candidates_df)

        review_mode = "candidate_selection_only"
        applied_reviews = 0

        if args.simulate_oracle:
            candidate_positions = candidates_df["row_position"].astype(int).tolist()
            final_sets = apply_oracle_review(eval_df, base_sets, candidate_positions)
            review_mode = "simulated_oracle"
            applied_reviews = len(candidate_positions)
        elif args.manual_review_file:
            final_sets, applied_reviews = apply_manual_review(base_sets, args.manual_review_file)
            review_mode = "manual_review_file"
        else:
            final_sets = base_sets

        y_pred_all = build_pred_matrix_from_sets(final_sets)
        overall, per_label = compute_quick_metrics(eval_df, y_pred_all)

        comparison_rows.append(
            save_hitl_run(
                out_dir=scenario_dir,
                eval_df=eval_df,
                method_sets=method_sets,
                base_sets=base_sets,
                candidates_df=candidates_df,
                review_template_df=review_template_df,
                final_sets=final_sets,
                y_pred_all=y_pred_all,
                overall=overall,
                per_label=per_label,
                args=args,
                budget=budget,
                review_mode=review_mode,
                applied_reviews=applied_reviews,
            )
        )

        print(f"  Micro-F1: {overall['micro_f1']:.4f}")
        print(f"  Macro-F1: {overall['macro_f1']:.4f}")
        print(f"  Hamming loss: {overall['hamming_loss']:.4f}")
        print(f"  Jaccard: {overall['jaccard_samples']:.4f}")
        print(f"  Subset accuracy: {overall['subset_accuracy']:.4f}")
        print(f"  Salidas: {scenario_dir}")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = OUT_DIR / "hitl_budget_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print()
    print("Humano en el bucle completado.")
    print(f"Comparativa de presupuestos: {comparison_path}")
    print(f"Carpeta principal: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
