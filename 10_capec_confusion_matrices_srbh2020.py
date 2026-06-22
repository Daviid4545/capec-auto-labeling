"""
10_capec_confusion_matrices_srbh2020.py

Analisis complementario: matrices inter-etiqueta CAPEC/Normal.

Objetivo:
    Calcular y graficar matrices de relacion entre etiquetas reales y etiquetas
    predichas para cada tecnica. Este script NO sustituye a las metricas
    principales del script 08; sirve para analizar de forma cualitativa que
    categorias se confunden entre si.

Que es esta matriz en multietiqueta:
    Celda (i, j) = numero de instancias donde la etiqueta real i esta activa
    y la etiqueta predicha j tambien esta activa.

    Como el problema es multietiqueta, una misma instancia puede contribuir a
    varias celdas si tiene varias etiquetas reales o varias predicciones.

Alcance:
    - Tecnicas sobre eval_set: reglas, embeddings, supervision debil, HITL 100.
    - Tecnicas LLM: zero-shot y few-shot sobre llm_eval_subset_srbh2020.
    - Por tanto, no se debe comparar directamente la intensidad absoluta de
      matrices LLM con matrices de eval_set sin mencionar el distinto numero
      de filas.

Salidas en ./plots_results_srbh2020/:
    - fig_10_capec_confusion_<tecnica>.png
    - fig_10_grid_capec_confusion.png
    - capec_confusion_raw_<tecnica>.csv
    - capec_confusion_normalized_<tecnica>.csv
    - capec_confusion_support_<tecnica>.csv
    - figure_index_capec_confusion.csv
    - capec_confusion_summary.txt

Uso:
    python 10_capec_confusion_matrices_srbh2020.py

Dependencias:
    python -m pip install pandas numpy matplotlib
"""

from pathlib import Path
import argparse
import csv
import re
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except ModuleNotFoundError:
    print("ERROR: falta matplotlib.")
    print("Ejecuta: python -m pip install matplotlib")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------
DEFAULT_OUT_DIR = Path("plots_results_srbh2020")
DPI = 300

mpl.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

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

TECHNIQUES = [
    {
        "name": "rules",
        "display": "Reglas",
        "scope": "eval_set",
        "path": "rules_labeling_srbh2020/predictions_rules.csv",
    },
    {
        "name": "embeddings",
        "display": "Embeddings",
        "scope": "eval_set",
        "path": "embeddings_labeling_srbh2020/predictions_embeddings.csv",
    },
    {
        "name": "weak_supervision",
        "display": "Supervisión débil",
        "scope": "eval_set",
        "path": "weak_supervision_srbh2020/predictions_weak_supervision.csv",
    },
    {
        "name": "llm_zero_shot",
        "display": "LLM zero-shot",
        "scope": "llm_eval_subset_srbh2020",
        "path": "llm_labeling_srbh2020/predictions_llm_zero_shot.csv",
    },
    {
        "name": "llm_few_shot",
        "display": "LLM few-shot",
        "scope": "llm_eval_subset_srbh2020",
        "path": "llm_labeling_srbh2020/predictions_llm_few_shot.csv",
    },
    {
        "name": "hitl_budget_100",
        "display": "HITL 100",
        "scope": "eval_set",
        "path": "human_in_the_loop_srbh2020/budget_100/predictions_human_in_the_loop.csv",
    },
]


# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------
def safe_col(label):
    return "pred__" + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def short_label(label, max_len=24):
    if len(label) <= max_len:
        return label
    return label[:max_len - 1] + "…"


def validate_file(df, path):
    missing_true = [label for label in ALL_LABELS if label not in df.columns]
    missing_pred = [safe_col(label) for label in ALL_LABELS if safe_col(label) not in df.columns]

    if missing_true:
        raise ValueError(f"Faltan columnas de etiquetas reales en {path}: {missing_true}")
    if missing_pred:
        raise ValueError(f"Faltan columnas de prediccion en {path}: {missing_pred}")


def get_true_pred_matrices(df):
    """
    Devuelve y_true e y_pred saneados.

    Reglas:
        - Si una fila no tiene etiqueta/prediccion activa, se marca como Normal.
        - Si Normal aparece junto con una etiqueta de ataque, se elimina Normal.
          Esto evita que Normal contamine matrices de ataque en caso de conflicto.
    """
    y_true = df[ALL_LABELS].fillna(0).astype(int).values
    y_pred = df[[safe_col(label) for label in ALL_LABELS]].fillna(0).astype(int).values

    normal_idx = ALL_LABELS.index(NORMAL_LABEL)

    empty_true = y_true.sum(axis=1) == 0
    if empty_true.any():
        y_true[empty_true, normal_idx] = 1

    true_attack_sum = y_true[:, 1:].sum(axis=1)
    true_conflict = (y_true[:, normal_idx] == 1) & (true_attack_sum > 0)
    if true_conflict.any():
        y_true[true_conflict, normal_idx] = 0

    empty_pred = y_pred.sum(axis=1) == 0
    if empty_pred.any():
        y_pred[empty_pred, normal_idx] = 1

    pred_attack_sum = y_pred[:, 1:].sum(axis=1)
    pred_conflict = (y_pred[:, normal_idx] == 1) & (pred_attack_sum > 0)
    if pred_conflict.any():
        y_pred[pred_conflict, normal_idx] = 0

    return y_true, y_pred


def build_interlabel_matrix(y_true, y_pred):
    """
    M[i, j] = numero de filas donde etiqueta real i=1 y etiqueta predicha j=1.
    """
    return y_true.T @ y_pred


def normalize_by_true_support(matrix, y_true):
    support = y_true.sum(axis=0).astype(float)
    normalized = np.zeros_like(matrix, dtype=float)

    for i, supp in enumerate(support):
        if supp > 0:
            normalized[i, :] = matrix[i, :] / supp

    return normalized, support.astype(int)


def save_matrix_csv(path, matrix):
    pd.DataFrame(matrix, index=ALL_LABELS, columns=ALL_LABELS).to_csv(path)


def save_support_csv(path, support):
    pd.DataFrame({
        "label": ALL_LABELS,
        "support": support,
    }).to_csv(path, index=False)


def plot_confusion(matrix_raw, matrix_norm, support, technique_display, scope, total_rows, out_path):
    n = matrix_raw.shape[0]

    fig, ax = plt.subplots(figsize=(11.8, 10.2))
    im = ax.imshow(matrix_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_title(
        f"Matriz inter-etiqueta CAPEC/Normal: {technique_display}\n"
        f"normalizada por soporte real; scope={scope}; n={total_rows}"
    )
    ax.set_xlabel("Etiqueta predicha")
    ax.set_ylabel("Etiqueta real")

    x_labels = [short_label(label) for label in ALL_LABELS]
    y_labels = [
        f"{short_label(label)}\n(n={int(support[i])})"
        for i, label in enumerate(ALL_LABELS)
    ]

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(y_labels)

    for i in range(n):
        for j in range(n):
            raw = int(matrix_raw[i, j])
            if raw == 0:
                continue

            pct = matrix_norm[i, j]
            color = "white" if pct > 0.55 else "black"
            ax.text(
                j,
                i,
                f"{raw}\n{pct * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=6.7,
                color=color,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Proporción sobre soporte real de la etiqueta", rotation=270, labelpad=16)

    for i in range(n):
        rect = plt.Rectangle(
            (i - 0.5, i - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#0066CC",
            linewidth=1.5,
        )
        ax.add_patch(rect)

    ax.text(
        0.99,
        -0.16,
        "La diagonal equivale al recall por etiqueta; celdas fuera de diagonal indican co-predicciones/confusiones.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_grid(matrices_dict, out_path):
    items = list(matrices_dict.items())

    if not items:
        return

    cols = 3
    rows = (len(items) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.8 * rows))
    axes = np.array(axes).reshape(-1)

    n = len(ALL_LABELS)
    tick_labels = [short_label(label, 15) for label in ALL_LABELS]

    for idx, (_, data) in enumerate(items):
        ax = axes[idx]
        matrix_norm = data["matrix_norm"]
        display = data["display"]
        scope = data["scope"]
        total_rows = data["total_rows"]

        ax.imshow(matrix_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title(f"{display}\n{scope}; n={total_rows}", fontsize=10)

        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(tick_labels, fontsize=6)

        for i in range(n):
            rect = plt.Rectangle(
                (i - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#0066CC",
                linewidth=0.8,
            )
            ax.add_patch(rect)

    for idx in range(len(items), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        "Matrices inter-etiqueta CAPEC/Normal por técnica\n"
        "(normalizadas por soporte real; diagonal = recall por etiqueta)",
        fontsize=14,
        y=1.0,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def record_figure(index_rows, fig_id, filename, title, description, source):
    index_rows.append({
        "figure_id": fig_id,
        "filename": filename,
        "title": title,
        "description": description,
        "source": source,
    })


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Carpeta de salida")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print("Generando matrices inter-etiqueta CAPEC/Normal...\n")

    matrices = {}
    figure_rows = []
    skipped = []

    for tech in TECHNIQUES:
        path = Path(tech["path"])

        if not path.exists():
            print(f"  AVISO: no existe {path}. Se omite {tech['display']}.")
            skipped.append(tech["name"])
            continue

        try:
            df = pd.read_csv(path, low_memory=False)
            validate_file(df, path)
        except Exception as exc:
            print(f"  ERROR cargando {path}: {exc}")
            skipped.append(tech["name"])
            continue

        y_true, y_pred = get_true_pred_matrices(df)
        matrix_raw = build_interlabel_matrix(y_true, y_pred)
        matrix_norm, support = normalize_by_true_support(matrix_raw, y_true)

        total_rows = len(df)

        matrices[tech["name"]] = {
            "matrix_raw": matrix_raw,
            "matrix_norm": matrix_norm,
            "support": support,
            "display": tech["display"],
            "scope": tech["scope"],
            "total_rows": total_rows,
            "source": tech["path"],
        }

        raw_csv = out_dir / f"capec_confusion_raw_{tech['name']}.csv"
        norm_csv = out_dir / f"capec_confusion_normalized_{tech['name']}.csv"
        support_csv = out_dir / f"capec_confusion_support_{tech['name']}.csv"
        png = out_dir / f"fig_10_capec_confusion_{tech['name']}.png"

        save_matrix_csv(raw_csv, matrix_raw)
        save_matrix_csv(norm_csv, np.round(matrix_norm, 6))
        save_support_csv(support_csv, support)

        plot_confusion(
            matrix_raw=matrix_raw,
            matrix_norm=matrix_norm,
            support=support,
            technique_display=tech["display"],
            scope=tech["scope"],
            total_rows=total_rows,
            out_path=png,
        )

        record_figure(
            figure_rows,
            f"fig_10_{tech['name']}",
            png.name,
            f"Matriz inter-etiqueta {tech['display']}",
            f"Matriz normalizada por soporte real para {tech['display']} ({tech['scope']}).",
            tech["path"],
        )

        print(f"  OK {tech['display']:25s} -> {png.name}")

    if matrices:
        grid_path = out_dir / "fig_10_grid_capec_confusion.png"
        plot_grid(matrices, grid_path)

        record_figure(
            figure_rows,
            "fig_10_grid",
            grid_path.name,
            "Resumen de matrices inter-etiqueta",
            "Grid comparativo de matrices normalizadas por soporte real.",
            "predicciones de tecnicas",
        )

        print(f"\n  Grid resumen: {grid_path.name}")

    index_path = out_dir / "figure_index_capec_confusion.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["figure_id", "filename", "title", "description", "source"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in figure_rows:
            writer.writerow(row)

    summary_path = out_dir / "capec_confusion_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("RESUMEN MATRICES INTER-ETIQUETA CAPEC/NORMAL\n")
        f.write("=" * 70 + "\n\n")
        f.write("Definicion: M[i,j] cuenta instancias donde la etiqueta real i esta activa y la etiqueta predicha j esta activa.\n")
        f.write("Normalizacion: M_norm[i,j] = M[i,j] / soporte_real_i.\n")
        f.write("Interpretacion: la diagonal normalizada equivale al recall por etiqueta; las celdas fuera de diagonal indican co-predicciones o confusiones.\n\n")

        f.write("--- Tecnicas generadas ---\n")
        for name, data in matrices.items():
            f.write(
                f"  - {name}: {data['display']} | scope={data['scope']} | "
                f"n={data['total_rows']} | fuente={data['source']}\n"
            )

        if skipped:
            f.write("\n--- Tecnicas omitidas ---\n")
            for name in skipped:
                f.write(f"  - {name}\n")

        f.write("\n--- Nota metodologica ---\n")
        f.write(
            "Las matrices LLM se calculan sobre llm_eval_subset_srbh2020, mientras que "
            "reglas, embeddings, supervision debil y HITL 100 se calculan sobre eval_set. "
            "Por tanto, deben interpretarse como analisis cualitativo complementario y no "
            "como ranking principal de tecnicas.\n"
        )

    print(f"\nTotal matrices generadas: {len(matrices)}")
    print(f"Indice: {index_path}")
    print(f"Resumen: {summary_path}")
    print(f"Salidas en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
