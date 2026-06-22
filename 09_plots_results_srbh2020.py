"""
09_plots_results_srbh2020.py

Generacion de graficas finales sobre SR-BH 2020.

Entradas esperadas en ./evaluation_srbh2020/:
    - main_results_table.csv
    - eval_set_comparison_table.csv
    - llm_comparison_table.csv
    - hitl_budget_comparison_table.csv
    - per_label_metrics.csv
    - confusion_matrices_per_label.csv
    - agreement_between_methods_eval_set.csv
    - overall_metrics.csv

Salidas en ./plots_results_srbh2020/:
    - fig_01_main_results_pr_f1.png
    - fig_02a_eval_set_precision_recall_f1.png
    - fig_02b_eval_set_other_metrics.png
    - fig_03_llm_zero_vs_few.png
    - fig_04_hitl_budget_curve.png
    - fig_05a_per_label_f1_eval_set.png
    - fig_05b_per_label_f1_llm_eval_subset_srbh2020.png
    - fig_06_agreement_between_methods.png
    - fig_07_hamming_loss_eval_set.png
    - fig_08_main_metrics_matrix.png
    - fig_09_fpr_fnr_eval_set.png
    - plot_summary.txt
    - figure_index.csv

Uso:
    python 09_plots_results_srbh2020.py
    python 09_plots_results_srbh2020.py --input-dir evaluation_srbh2020 --out-dir plots_results_srbh2020

Dependencias:
    python -m pip install pandas matplotlib numpy
"""

from pathlib import Path
import argparse
import csv
import textwrap
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
# CONFIGURACION VISUAL
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DIR = Path("evaluation_srbh2020")
DEFAULT_OUT_DIR = Path("plots_results_srbh2020")
DPI = 300

mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
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

METRIC_LABELS = {
    "micro_precision": "Precision micro",
    "micro_recall": "Recall micro",
    "micro_f1": "F1 micro",
    "macro_precision": "Precision macro",
    "macro_recall": "Recall macro",
    "macro_f1": "F1 macro",
    "hamming_loss": "Hamming loss",
    "hamming_accuracy": "1 - Hamming (acierto)",
    "jaccard_samples": "Jaccard",
    "subset_accuracy": "Subset accuracy",
    "cohen_kappa_flat": "Kappa (flat)",
    "cohen_kappa_label_mean": "Kappa (medio por etiqueta)",
}

METHOD_LABELS = {
    "rules": "Reglas",
    "embeddings": "Embeddings",
    "weak_supervision": "Supervisión débil",
    "llm_zero_shot": "LLM zero-shot",
    "llm_few_shot": "LLM few-shot",
    "hitl_budget_50": "HITL 50",
    "hitl_budget_100": "HITL 100",
    "hitl_budget_200": "HITL 200",
}

# Paletas consistentes
COLOR_PRECISION = "#4C72B0"
COLOR_RECALL = "#DD8452"
COLOR_F1 = "#55A868"

EVAL_SET_METHOD_ORDER = [
    "rules",
    "embeddings",
    "weak_supervision",
    "hitl_budget_50",
    "hitl_budget_100",
    "hitl_budget_200",
]

MAIN_METHOD_ORDER = [
    "rules",
    "embeddings",
    "weak_supervision",
    "llm_few_shot",
    "hitl_budget_100",
]

PER_LABEL_EVAL_METHODS = [
    "rules",
    "embeddings",
    "weak_supervision",
    "hitl_budget_100",
]

PER_LABEL_LLM_METHODS = [
    "llm_zero_shot",
    "llm_few_shot",
]


# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------
def read_csv_or_none(path):
    path = Path(path)
    if not path.exists():
        print(f"Aviso: no existe {path}. Se omite la figura dependiente.")
        return None
    return pd.read_csv(path, low_memory=False)


def method_display(method):
    return METHOD_LABELS.get(method, str(method))


def wrap_label(label, width=18):
    return "\n".join(textwrap.wrap(str(label), width=width))


def ensure_numeric(df, columns):
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def order_by_method(df, order):
    out = df.copy()
    out["__order__"] = out["method"].apply(
        lambda x: order.index(x) if x in order else len(order)
    )
    out = out.sort_values("__order__").drop(columns=["__order__"])
    return out


def save_figure(fig, out_path):
    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def add_value_labels_bars(ax, bars, fmt="{:.3f}", fontsize=8):
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def add_lower_is_better_note(ax):
    ax.text(
        0.99, 0.02, "Menor es mejor",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, style="italic", color="#555",
    )


def record_figure(index_rows, fig_id, filename, title, description, source):
    index_rows.append({
        "figure_id": fig_id,
        "filename": filename,
        "title": title,
        "description": description,
        "source_csv": source,
    })


def safe_plot_call(func, *args):
    try:
        func(*args)
    except Exception as exc:
        print(f"Aviso: no se pudo generar {func.__name__}: {exc}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# FIGURA 01: Principales con Precision/Recall/F1 micro y macro
# ---------------------------------------------------------------------------
def plot_main_results_pr_f1(main_df, overall_df, out_dir, index_rows):
    """
    Fig 01: Para cada tecnica principal, muestra P, R y F1 en micro y macro.
    Doble panel (micro y macro lado a lado) con 3 barras por tecnica.
    """
    if main_df is None or main_df.empty:
        return

    # Necesitamos las metricas de overall_metrics (no estan en main_results)
    if overall_df is None or overall_df.empty:
        print("Aviso: fig_01 necesita overall_metrics.csv")
        return

    metrics_needed = [
        "micro_precision", "micro_recall", "micro_f1",
        "macro_precision", "macro_recall", "macro_f1",
    ]
    if not {"method", *metrics_needed}.issubset(overall_df.columns):
        return

    df = ensure_numeric(overall_df, metrics_needed)
    df = df[df["method"].isin(MAIN_METHOD_ORDER)]
    df = order_by_method(df, MAIN_METHOD_ORDER)

    if df.empty:
        return

    x = np.arange(len(df))
    width = 0.27
    labels = [wrap_label(method_display(m), 14) for m in df["method"]]

    fig, (ax_micro, ax_macro) = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)

    # Panel micro
    b1 = ax_micro.bar(x - width, df["micro_precision"], width,
                      label="Precision", color=COLOR_PRECISION)
    b2 = ax_micro.bar(x, df["micro_recall"], width,
                     label="Recall", color=COLOR_RECALL)
    b3 = ax_micro.bar(x + width, df["micro_f1"], width,
                     label="F1", color=COLOR_F1)
    ax_micro.set_title("Métricas micro")
    ax_micro.set_ylabel("Valor")
    ax_micro.set_ylim(0, 1.05)
    ax_micro.set_xticks(x)
    ax_micro.set_xticklabels(labels)
    ax_micro.legend(loc="upper right")
    ax_micro.grid(axis="y", alpha=0.25)
    add_value_labels_bars(ax_micro, b1, fontsize=7)
    add_value_labels_bars(ax_micro, b2, fontsize=7)
    add_value_labels_bars(ax_micro, b3, fontsize=7)

    # Panel macro
    b4 = ax_macro.bar(x - width, df["macro_precision"], width,
                     label="Precision", color=COLOR_PRECISION)
    b5 = ax_macro.bar(x, df["macro_recall"], width,
                     label="Recall", color=COLOR_RECALL)
    b6 = ax_macro.bar(x + width, df["macro_f1"], width,
                     label="F1", color=COLOR_F1)
    ax_macro.set_title("Métricas macro")
    ax_macro.set_ylim(0, 1.05)
    ax_macro.set_xticks(x)
    ax_macro.set_xticklabels(labels)
    ax_macro.legend(loc="upper right")
    ax_macro.grid(axis="y", alpha=0.25)
    add_value_labels_bars(ax_macro, b4, fontsize=7)
    add_value_labels_bars(ax_macro, b5, fontsize=7)
    add_value_labels_bars(ax_macro, b6, fontsize=7)

    fig.suptitle("Resultados principales: Precision, Recall y F1 por familia",
                 y=1.02, fontsize=14, fontweight="bold")

    path = save_figure(fig, out_dir / "fig_01_main_results_pr_f1.png")

    record_figure(
        index_rows, "fig_01", path.name,
        "Resultados principales (P/R/F1 micro y macro)",
        "Precision, Recall y F1 (micro y macro) para la variante principal de cada familia; LLM se evalua sobre llm_eval_subset_srbh2020.",
        "overall_metrics.csv",
    )


# ---------------------------------------------------------------------------
# FIGURA 02a: 6 metricas en eval_set
# ---------------------------------------------------------------------------
def plot_eval_set_pr_f1(overall_df, out_dir, index_rows):
    """
    Fig 02a: P/R/F1 (micro y macro) para todas las tecnicas evaluadas
    sobre eval_set.
    """
    if overall_df is None or overall_df.empty:
        return

    metrics_needed = [
        "micro_precision", "micro_recall", "micro_f1",
        "macro_precision", "macro_recall", "macro_f1",
    ]
    if not {"method", "scope", *metrics_needed}.issubset(overall_df.columns):
        return

    df = ensure_numeric(overall_df, metrics_needed)
    df = df[df["scope"] == "eval_set"]
    df = df[df["method"].isin(EVAL_SET_METHOD_ORDER)]
    df = order_by_method(df, EVAL_SET_METHOD_ORDER)

    if df.empty:
        return

    x = np.arange(len(df))
    width = 0.13
    labels = [wrap_label(method_display(m), 14) for m in df["method"]]

    fig, ax = plt.subplots(figsize=(13.5, 6))

    metrics_plot = [
        ("micro_precision", "Precision micro", "#2E5C8A"),
        ("micro_recall", "Recall micro", "#B25A2B"),
        ("micro_f1", "F1 micro", "#3D7A48"),
        ("macro_precision", "Precision macro", "#7BA3D0"),
        ("macro_recall", "Recall macro", "#E5A77E"),
        ("macro_f1", "F1 macro", "#83B58D"),
    ]

    for idx, (metric, label, color) in enumerate(metrics_plot):
        offset = (idx - (len(metrics_plot) - 1) / 2) * width
        ax.bar(x + offset, df[metric], width, label=label, color=color)

    ax.set_title("Precision, Recall y F1 sobre eval_set (técnicas comparables)")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(ncol=3, loc="upper right", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25)

    path = save_figure(fig, out_dir / "fig_02a_eval_set_precision_recall_f1.png")

    record_figure(
        index_rows, "fig_02a", path.name,
        "P/R/F1 sobre eval_set",
        "Precision, Recall y F1 (micro y macro) de todas las técnicas evaluadas sobre eval_set.",
        "overall_metrics.csv",
    )


# ---------------------------------------------------------------------------
# FIGURA 02b: otras metricas en eval_set
# ---------------------------------------------------------------------------
def plot_eval_set_other_metrics(overall_df, out_dir, index_rows):
    """
    Fig 02b: Jaccard, subset accuracy, kappa flat y kappa label_mean
    sobre eval_set.
    """
    if overall_df is None or overall_df.empty:
        return

    metrics_needed = [
        "jaccard_samples", "subset_accuracy",
        "cohen_kappa_flat", "cohen_kappa_label_mean",
    ]
    if not {"method", "scope", *metrics_needed}.issubset(overall_df.columns):
        return

    df = ensure_numeric(overall_df, metrics_needed)
    df = df[df["scope"] == "eval_set"]
    df = df[df["method"].isin(EVAL_SET_METHOD_ORDER)]
    df = order_by_method(df, EVAL_SET_METHOD_ORDER)

    if df.empty:
        return

    x = np.arange(len(df))
    width = 0.2
    labels = [wrap_label(method_display(m), 14) for m in df["method"]]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))

    metrics_plot = [
        ("jaccard_samples", "Jaccard"),
        ("subset_accuracy", "Subset accuracy"),
        ("cohen_kappa_flat", "Kappa (flat)"),
        ("cohen_kappa_label_mean", "Kappa (medio por etiqueta)"),
    ]

    for idx, (metric, label) in enumerate(metrics_plot):
        offset = (idx - (len(metrics_plot) - 1) / 2) * width
        bars = ax.bar(x + offset, df[metric], width, label=label)
        add_value_labels_bars(ax, bars, fontsize=7)

    ax.set_title("Métricas complementarias sobre eval_set")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(ncol=2, loc="upper right", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25)

    path = save_figure(fig, out_dir / "fig_02b_eval_set_other_metrics.png")

    record_figure(
        index_rows, "fig_02b", path.name,
        "Métricas complementarias sobre eval_set",
        "Jaccard, subset accuracy, Cohen kappa (flat y medio por etiqueta).",
        "overall_metrics.csv",
    )


# ---------------------------------------------------------------------------
# FIGURA 03: LLM zero vs few con P/R/F1
# ---------------------------------------------------------------------------
def plot_llm_comparison(overall_df, out_dir, index_rows):
    """
    Fig 03: zero-shot vs few-shot con 6 metricas (P/R/F1 micro y macro).
    """
    if overall_df is None or overall_df.empty:
        return

    metrics_needed = [
        "micro_precision", "micro_recall", "micro_f1",
        "macro_precision", "macro_recall", "macro_f1",
    ]
    if not {"method", *metrics_needed}.issubset(overall_df.columns):
        return

    order = ["llm_zero_shot", "llm_few_shot"]
    df = ensure_numeric(overall_df, metrics_needed)
    df = df[df["method"].isin(order)]
    df = order_by_method(df, order)

    if df.empty:
        return

    metrics_plot = [
        ("micro_precision", "P micro"),
        ("micro_recall", "R micro"),
        ("micro_f1", "F1 micro"),
        ("macro_precision", "P macro"),
        ("macro_recall", "R macro"),
        ("macro_f1", "F1 macro"),
    ]

    x = np.arange(len(metrics_plot))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for idx, (_, row) in enumerate(df.iterrows()):
        offset = (idx - (len(df) - 1) / 2) * width
        values = [row[m] for m, _ in metrics_plot]
        bars = ax.bar(x + offset, values, width,
                      label=method_display(row["method"]))
        add_value_labels_bars(ax, bars, fontsize=8)

    ax.set_title("LLM: zero-shot vs few-shot (P, R y F1) sobre llm_eval_subset_srbh2020")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics_plot])
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    path = save_figure(fig, out_dir / "fig_03_llm_zero_vs_few.png")

    record_figure(
        index_rows, "fig_03", path.name,
        "LLM zero-shot vs few-shot",
        "Compara P, R y F1 (micro y macro) entre zero-shot y few-shot sobre llm_eval_subset_srbh2020.",
        "overall_metrics.csv",
    )


# ---------------------------------------------------------------------------
# FIGURA 04: HITL budget curve
# ---------------------------------------------------------------------------
def plot_hitl_budget_curve(hitl_df, out_dir, index_rows):
    if hitl_df is None or hitl_df.empty:
        return

    needed = {"budget", "micro_f1", "macro_f1",
              "jaccard_samples", "subset_accuracy"}
    if not needed.issubset(hitl_df.columns):
        return

    metrics_up = ["micro_f1", "macro_f1", "jaccard_samples", "subset_accuracy"]
    df = ensure_numeric(hitl_df, ["budget"] + metrics_up)
    df = df.sort_values("budget")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for metric in metrics_up:
        ax.plot(df["budget"], df[metric], marker="o", linewidth=2,
                markersize=8, label=METRIC_LABELS[metric])

    ax.set_title("Impacto del presupuesto de revisión humana simulada")
    ax.set_xlabel("Número de instancias revisadas")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(df["budget"].astype(int).tolist())
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    path = save_figure(fig, out_dir / "fig_04_hitl_budget_curve.png")

    record_figure(
        index_rows, "fig_04", path.name,
        "Impacto del presupuesto HITL",
        "Evolución de métricas al revisar 50, 100 y 200 casos.",
        "hitl_budget_comparison_table.csv",
    )


# ---------------------------------------------------------------------------
# FIGURA 05a: F1 por etiqueta en eval_set CON SOPORTE
# ---------------------------------------------------------------------------
def plot_per_label_f1_eval_set(per_label_df, out_dir, index_rows):
    if per_label_df is None or per_label_df.empty:
        return

    needed = {"method", "label", "scope", "f1", "support"}
    if not needed.issubset(per_label_df.columns):
        return

    df = per_label_df[
        (per_label_df["scope"] == "eval_set")
        & (per_label_df["method"].isin(PER_LABEL_EVAL_METHODS))
    ].copy()

    if df.empty:
        return

    df = ensure_numeric(df, ["f1", "support"])

    pivot = df.pivot_table(index="label", columns="method",
                           values="f1", aggfunc="mean")
    pivot = pivot.reindex(ALL_LABELS)

    # Soporte por etiqueta (es el mismo para todas las tecnicas, lo cogemos
    # de una cualquiera)
    support_map = df.groupby("label")["support"].max().to_dict()

    existing_methods = [m for m in PER_LABEL_EVAL_METHODS
                        if m in pivot.columns]
    pivot = pivot[existing_methods]

    # Etiqueta con soporte
    x_labels = []
    for lab in pivot.index:
        n = int(support_map.get(lab, 0))
        x_labels.append(f"{wrap_label(lab, 22)}\n(n={n})")

    x = np.arange(len(pivot.index))
    width = 0.78 / max(len(existing_methods), 1)

    fig, ax = plt.subplots(figsize=(16, 7))

    for idx, method in enumerate(existing_methods):
        offset = (idx - (len(existing_methods) - 1) / 2) * width
        ax.bar(x + offset, pivot[method], width,
               label=method_display(method))

    ax.set_title("F1 por etiqueta CAPEC sobre eval_set "
                 "(con soporte n entre paréntesis)")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.legend(ncol=2, loc="upper right", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25)

    path = save_figure(fig, out_dir / "fig_05a_per_label_f1_eval_set.png")

    record_figure(
        index_rows, "fig_05a", path.name,
        "F1 por etiqueta CAPEC en eval_set",
        "F1 por etiqueta para reglas, embeddings, supervisión débil y HITL-100 con soporte de cada clase.",
        "per_label_metrics.csv",
    )


# ---------------------------------------------------------------------------
# FIGURA 05b: F1 por etiqueta en llm_eval_subset_srbh2020 (zero vs few)
# ---------------------------------------------------------------------------
def plot_per_label_f1_llm(per_label_df, out_dir, index_rows):
    if per_label_df is None or per_label_df.empty:
        return

    needed = {"method", "label", "scope", "f1", "support"}
    if not needed.issubset(per_label_df.columns):
        return

    df = per_label_df[
        (per_label_df["scope"] == "llm_eval_subset")
        & (per_label_df["method"].isin(PER_LABEL_LLM_METHODS))
    ].copy()

    if df.empty:
        print("Aviso: fig_05b no se genera, no hay datos en scope llm_eval_subset.")
        return

    df = ensure_numeric(df, ["f1", "support"])

    pivot = df.pivot_table(index="label", columns="method",
                           values="f1", aggfunc="mean")
    pivot = pivot.reindex(ALL_LABELS)

    support_map = df.groupby("label")["support"].max().to_dict()

    existing_methods = [m for m in PER_LABEL_LLM_METHODS
                        if m in pivot.columns]
    pivot = pivot[existing_methods]

    x_labels = []
    for lab in pivot.index:
        n = int(support_map.get(lab, 0))
        x_labels.append(f"{wrap_label(lab, 22)}\n(n={n})")

    x = np.arange(len(pivot.index))
    width = 0.4

    fig, ax = plt.subplots(figsize=(15, 6.5))

    colors = {"llm_zero_shot": "#9B6FBF", "llm_few_shot": "#D4734F"}
    for idx, method in enumerate(existing_methods):
        offset = (idx - (len(existing_methods) - 1) / 2) * width
        ax.bar(x + offset, pivot[method], width,
               label=method_display(method),
               color=colors.get(method, None))

    ax.set_title("F1 por etiqueta CAPEC sobre llm_eval_subset_srbh2020 "
                 "(zero-shot vs few-shot, con soporte n)")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25)

    path = save_figure(fig, out_dir / "fig_05b_per_label_f1_llm_eval_subset_srbh2020.png")

    record_figure(
        index_rows, "fig_05b", path.name,
        "F1 por etiqueta CAPEC en llm_eval_subset_srbh2020",
        "Compara F1 por etiqueta entre LLM zero-shot y few-shot sobre llm_eval_subset_srbh2020.",
        "per_label_metrics.csv",
    )


# ---------------------------------------------------------------------------
# FIGURA 06: acuerdo entre tecnicas
# ---------------------------------------------------------------------------
def plot_agreement(agreement_df, out_dir, index_rows):
    """
    Figura 06: acuerdo entre tecnicas sobre eval_set.

    La barra azul representa el acuerdo Jaccard medio.
    La barra roja, mas fina y centrada sobre la azul, representa Cohen's kappa flat.
    Ambas usan la misma escala 0-1 para facilitar la comparacion visual.
    """
    if agreement_df is None or agreement_df.empty:
        return

    needed = {"method_a", "method_b", "avg_jaccard_agreement"}
    if not needed.issubset(agreement_df.columns):
        return

    df = ensure_numeric(
        agreement_df,
        ["avg_jaccard_agreement", "exact_match_agreement", "cohen_kappa_flat"],
    )

    df["pair"] = (
        df["method_a"].apply(method_display)
        + " vs "
        + df["method_b"].apply(method_display)
    )

    df = df.sort_values("avg_jaccard_agreement", ascending=True)

    fig_height = max(6, 0.5 * len(df) + 2)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    y = np.arange(len(df))

    # Barra principal: acuerdo Jaccard medio.
    bars_jaccard = ax.barh(
        y,
        df["avg_jaccard_agreement"],
        height=0.72,
        color="#4C72B0",
        label="Jaccard medio",
        alpha=0.92,
    )

    # Barra secundaria: Cohen's kappa, mas fina y centrada dentro de la azul.
    has_kappa = "cohen_kappa_flat" in df.columns and df["cohen_kappa_flat"].notna().any()
    if has_kappa:
        ax.barh(
            y,
            df["cohen_kappa_flat"].fillna(0),
            height=0.28,
            color="#C44E52",
            label="Cohen's kappa",
            alpha=0.95,
        )

    ax.set_title("Acuerdo entre técnicas evaluadas sobre eval_set")
    ax.set_xlabel("Acuerdo (0 a 1)")
    ax.set_xlim(0, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels(df["pair"], fontsize=9)
    ax.grid(axis="x", alpha=0.25)

    # Etiqueta numerica del Jaccard, colocada al final de la barra azul.
    for bar, value in zip(bars_jaccard, df["avg_jaccard_agreement"]):
        ax.annotate(
            f"{value:.3f}",
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )

    # Etiqueta numerica del kappa dentro o cerca de la barra roja.
    if has_kappa:
        for yi, value in zip(y, df["cohen_kappa_flat"].fillna(0)):
            if value <= 0:
                continue

            # Si el valor es alto, texto dentro de la barra; si es bajo, fuera.
            if value > 0.18:
                x_text = max(value - 0.035, 0.02)
                ha = "right"
                color = "white"
            else:
                x_text = value + 0.015
                ha = "left"
                color = "black"

            ax.text(
                x_text,
                yi,
                f"{value:.3f}",
                va="center",
                ha=ha,
                fontsize=7,
                color=color,
            )

    ax.legend(loc="lower right", framealpha=0.92)

    path = save_figure(fig, out_dir / "fig_06_agreement_between_methods.png")

    record_figure(
        index_rows,
        "fig_06",
        path.name,
        "Acuerdo entre técnicas",
        "Jaccard medio entre pares de técnicas (barra azul) y Cohen's kappa flat (barra roja fina centrada).",
        "agreement_between_methods_eval_set.csv",
    )

# ---------------------------------------------------------------------------
# FIGURA 07: Hamming loss
# ---------------------------------------------------------------------------
def plot_hamming_loss(overall_df, out_dir, index_rows):
    if overall_df is None or overall_df.empty:
        return

    if not {"method", "scope", "hamming_loss"}.issubset(overall_df.columns):
        return

    df = ensure_numeric(overall_df, ["hamming_loss"])
    df = df[df["scope"] == "eval_set"]
    df = df[df["method"].isin(EVAL_SET_METHOD_ORDER)]
    df = order_by_method(df, EVAL_SET_METHOD_ORDER)

    if df.empty:
        return

    labels = [wrap_label(method_display(m), 14) for m in df["method"]]
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(x, df["hamming_loss"], color="#C44E52")

    ax.set_title("Hamming loss sobre eval_set")
    ax.set_ylabel("Hamming loss")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    add_lower_is_better_note(ax)
    add_value_labels_bars(ax, bars, fmt="{:.4f}", fontsize=9)

    path = save_figure(fig, out_dir / "fig_07_hamming_loss_eval_set.png")

    record_figure(
        index_rows, "fig_07", path.name,
        "Hamming loss sobre eval_set",
        "Proporción de etiquetas mal asignadas en eval_set.",
        "overall_metrics.csv",
    )


# ---------------------------------------------------------------------------
# FIGURA 08: matriz de metricas (MEJORADA)
# ---------------------------------------------------------------------------
def plot_main_metrics_matrix(overall_df, out_dir, index_rows):
    """
    Heatmap con metricas principales por tecnica. Hamming loss se convierte
    a "1 - Hamming" para que TODO el heatmap se lea "mas oscuro = mejor".
    """
    if overall_df is None or overall_df.empty:
        return

    metrics = [
        "micro_precision", "micro_recall", "micro_f1",
        "macro_precision", "macro_recall", "macro_f1",
        "hamming_accuracy",   # = 1 - hamming_loss
        "jaccard_samples",
        "subset_accuracy",
        "cohen_kappa_flat",
        "cohen_kappa_label_mean",
    ]

    df = overall_df.copy()
    # Calcular hamming_accuracy
    if "hamming_loss" in df.columns:
        df["hamming_accuracy"] = 1.0 - pd.to_numeric(df["hamming_loss"],
                                                     errors="coerce")
    else:
        df["hamming_accuracy"] = np.nan

    df = ensure_numeric(df, metrics)
    df = df[df["method"].isin(MAIN_METHOD_ORDER)]
    df = order_by_method(df, MAIN_METHOD_ORDER)

    if df.empty:
        return

    methods = [method_display(m) for m in df["method"]]
    metric_names = [METRIC_LABELS.get(m, m) for m in metrics]
    values = df[metrics].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    im = ax.imshow(values, aspect="auto", cmap="viridis_r",
                   vmin=0, vmax=1)

    ax.set_title("Matriz de métricas principales por técnica\n"
                 "(todas las métricas están normalizadas: más oscuro = mejor)",
                 fontsize=12)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metric_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                text = "—"
                color = "gray"
            else:
                text = f"{value:.3f}"
                # texto blanco si fondo oscuro, negro si claro
                color = "white" if value > 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Valor (0-1, mayor mejor)", rotation=270, labelpad=15)

    path = save_figure(fig, out_dir / "fig_08_main_metrics_matrix.png")

    record_figure(
        index_rows, "fig_08", path.name,
        "Matriz de métricas principales",
        "Heatmap con 11 métricas para técnicas principales. Hamming se muestra como 1-Hamming; LLM se evalua sobre llm_eval_subset_srbh2020.",
        "overall_metrics.csv",
    )


# ---------------------------------------------------------------------------
# FIGURA 09: FPR y FNR por tecnica sobre eval_set
# ---------------------------------------------------------------------------
def plot_fpr_fnr_per_method(confusion_df, out_dir, index_rows):
    """
    FPR y FNR agregados por tecnica sobre eval_set.
    Calculamos FPR_micro y FNR_micro = sum(FP)/sum(FP+TN) y sum(FN)/sum(FN+TP)
    para tener un valor por tecnica.
    """
    if confusion_df is None or confusion_df.empty:
        return

    needed = {"method", "scope", "fp", "tn", "fn", "tp"}
    if not needed.issubset(confusion_df.columns):
        return

    df = confusion_df.copy()
    df = df[df["scope"] == "eval_set"]
    df = df[df["method"].isin(EVAL_SET_METHOD_ORDER)]
    if df.empty:
        return

    df = ensure_numeric(df, ["fp", "tn", "fn", "tp"])

    agg = df.groupby("method").agg(
        FP=("fp", "sum"),
        TN=("tn", "sum"),
        FN=("fn", "sum"),
        TP=("tp", "sum"),
    ).reset_index()

    agg["FPR_micro"] = agg["FP"] / (agg["FP"] + agg["TN"]).replace(0, np.nan)
    agg["FNR_micro"] = agg["FN"] / (agg["FN"] + agg["TP"]).replace(0, np.nan)
    agg = order_by_method(agg, EVAL_SET_METHOD_ORDER)

    x = np.arange(len(agg))
    width = 0.38
    labels = [wrap_label(method_display(m), 14) for m in agg["method"]]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - width / 2, agg["FPR_micro"], width,
                label="FPR micro (falsos positivos)", color="#C44E52")
    b2 = ax.bar(x + width / 2, agg["FNR_micro"], width,
                label="FNR micro (falsos negativos)", color="#8C8C8C")

    ax.set_title("Tasas de error por técnica sobre eval_set (agregadas)")
    ax.set_ylabel("Tasa (0-1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    add_lower_is_better_note(ax)
    add_value_labels_bars(ax, b1, fmt="{:.4f}", fontsize=8)
    add_value_labels_bars(ax, b2, fmt="{:.4f}", fontsize=8)

    path = save_figure(fig, out_dir / "fig_09_fpr_fnr_eval_set.png")

    record_figure(
        index_rows, "fig_09", path.name,
        "FPR y FNR por técnica",
        "Tasa de falsos positivos (FPR) y falsos negativos (FNR) agregadas por técnica sobre eval_set.",
        "confusion_matrices_per_label.csv",
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                        help="Carpeta de entrada con CSV de evaluacion")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="Carpeta de salida para graficas")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"Leyendo resultados desde: {input_dir}")
    print(f"Guardando graficas en: {out_dir}")

    main_df = read_csv_or_none(input_dir / "main_results_table.csv")
    overall_df = read_csv_or_none(input_dir / "overall_metrics.csv")
    eval_df = read_csv_or_none(input_dir / "eval_set_comparison_table.csv")
    llm_df = read_csv_or_none(input_dir / "llm_comparison_table.csv")
    hitl_df = read_csv_or_none(input_dir / "hitl_budget_comparison_table.csv")
    per_label_df = read_csv_or_none(input_dir / "per_label_metrics.csv")
    agreement_df = read_csv_or_none(input_dir / "agreement_between_methods_eval_set.csv")
    confusion_df = read_csv_or_none(input_dir / "confusion_matrices_per_label.csv")

    index_rows = []

    safe_plot_call(plot_main_results_pr_f1, main_df, overall_df, out_dir, index_rows)
    safe_plot_call(plot_eval_set_pr_f1, overall_df, out_dir, index_rows)
    safe_plot_call(plot_eval_set_other_metrics, overall_df, out_dir, index_rows)
    safe_plot_call(plot_llm_comparison, overall_df, out_dir, index_rows)
    safe_plot_call(plot_hitl_budget_curve, hitl_df, out_dir, index_rows)
    safe_plot_call(plot_per_label_f1_eval_set, per_label_df, out_dir, index_rows)
    safe_plot_call(plot_per_label_f1_llm, per_label_df, out_dir, index_rows)
    safe_plot_call(plot_agreement, agreement_df, out_dir, index_rows)
    safe_plot_call(plot_hamming_loss, overall_df, out_dir, index_rows)
    safe_plot_call(plot_main_metrics_matrix, overall_df, out_dir, index_rows)
    safe_plot_call(plot_fpr_fnr_per_method, confusion_df, out_dir, index_rows)

    figure_index_path = out_dir / "figure_index.csv"
    with figure_index_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["figure_id", "filename", "title", "description", "source_csv"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in index_rows:
            writer.writerow(row)

    summary_path = out_dir / "plot_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("RESUMEN DE GRAFICAS GENERADAS - SR-BH 2020 (v2)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Carpeta de entrada: {input_dir}\n")
        f.write(f"Carpeta de salida: {out_dir}\n")
        f.write(f"Figuras generadas: {len(index_rows)}\n\n")

        for row in index_rows:
            f.write(f"{row['figure_id']} - {row['title']}\n")
            f.write(f"  Archivo: {row['filename']}\n")
            f.write(f"  Fuente: {row['source_csv']}\n")
            f.write(f"  Descripcion: {row['description']}\n\n")

        f.write("--- Notas metodologicas ---\n")
        f.write(
            "Las graficas separan comparaciones por scope: eval_set para "
            "tecnicas comparables (reglas, embeddings, supervision debil, HITL), "
            "y llm_eval_subset_srbh2020 para zero-shot vs few-shot. Las figuras "
            "resumen que incluyen LLM deben interpretarse con esta diferencia de "
            "scope. En el heatmap de metricas, Hamming loss se muestra como "
            "1-Hamming para que valores mayores indiquen mejor rendimiento.\n"
        )

    print()
    print(f"Total de figuras generadas: {len(index_rows)}")
    for row in index_rows:
        print(f"  - {row['filename']}")

    print(f"Indice de figuras: {figure_index_path}")
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()
