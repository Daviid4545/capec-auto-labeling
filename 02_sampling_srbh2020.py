"""
02_sampling_srbh2020.py

Crea particiones estratificadas multietiqueta a partir del dataset SR-BH 2020,
sin modificar el archivo original.

Diseño:
    - eval_set: conjunto comun de evaluacion para reglas, embeddings y
      supervision debil/distante. Es el conjunto principal donde se calculan
      metricas comparables.
    - train_set: conjunto disjunto de eval_set. Se usa para calibracion,
      seleccion de ejemplos few-shot y apoyo a tecnicas que necesiten una
      particion auxiliar. No se usa para evaluar resultados finales.
    - llm_eval_subset_srbh2020: subconjunto reducido y estratificado de eval_set
      para las pruebas zero-shot/few-shot con LLM local. Se crea por coste y
      tiempo de inferencia, manteniendo representacion de Normal y CAPEC.

Estratificacion:
    Se usa MultilabelStratifiedShuffleSplit (iterative-stratification) para
    mantener la distribucion multietiqueta de las particiones principales.

Exclusiones:
    - CAPEC-248 (Command Injection) tiene una unica instancia en el dataset.
      Se descarta como columna operativa por insuficiencia muestral.
      Las filas no se eliminan por este motivo.

Filas sin informacion HTTP:
    - No se eliminan. Se conservan para mantener trazabilidad con el dataset
      original y evitar decisiones de limpieza no previstas en el alcance.
      El script solo informa de cuantas filas tienen vacias todas las features
      HTTP consideradas.

text_input:
    Se construye concatenando campos de la peticion del atacante, no de la
    respuesta del servidor. Esto evita que el etiquetado dependa de como
    responde el servidor.

Uso:
    pip install pandas iterative-stratification
    python 02_sampling_srbh2020.py data_capec_multilabel.csv

Salidas en ./sampling_srbh2020/:
    - eval_set.csv
    - train_set.csv
    - llm_eval_subset_srbh2020.csv
    - eval_set_label_distribution.csv
    - train_set_label_distribution.csv
    - llm_eval_subset_srbh2020_label_distribution.csv
    - sample_report.txt
"""

from pathlib import Path
import sys
import csv

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------
SEED = 42

EVAL_SIZE = 8000
TRAIN_SIZE = 12000

LLM_EVAL_SIZE = 100
LLM_EVAL_NORMAL_SIZE = 40
LLM_EVAL_PER_LABEL = 5

EXCLUDED_LABELS = ["248 - Command Injection"]

TEXT_FIELDS = [
    "request_http_method",
    "request_http_request",
    "request_user_agent",
    "request_referer",
    "request_host",
    "request_origin",
    "request_cookie",
    "request_content_type",
    "request_body",
]

OUT_DIR = Path("sampling_srbh2020")


# ---------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------------------------
def build_text_input(row, text_fields):
    """
    Concatena los campos de texto de la peticion en una sola cadena.
    Los campos vacios se omiten para no introducir ruido.
    """
    parts = []

    for field in text_fields:
        value = row.get(field, "")

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            parts.append(f"{field}: {value}")

    return " | ".join(parts)


def stratified_split(df, label_cols, size, seed):
    """
    Devuelve indices de una muestra estratificada multietiqueta de tamano `size`.
    """
    if size >= len(df):
        return df.index.to_numpy()

    y = df[label_cols].astype(int).values

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=size,
        random_state=seed,
    )

    _, take_idx = next(splitter.split(np.zeros(len(df)), y))
    return df.index.to_numpy()[take_idx]


def label_distribution(df, label_cols):
    counts = (df[label_cols].astype(int) == 1).sum().to_dict()
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def write_label_distribution(path, df, label_cols):
    counts = label_distribution(df, label_cols)
    total = len(df)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "count", "percentage_sample"])

        for label, count in counts.items():
            pct = (count / total * 100) if total else 0
            writer.writerow([label, int(count), round(pct, 4)])


def create_llm_eval_subset(eval_set, label_cols, size, normal_size, per_label, seed):
    """
    Crea un subconjunto reducido desde eval_set para las pruebas LLM.

    Criterio:
        - normales puros;
        - hasta `per_label` ejemplos por cada CAPEC de ataque;
        - preferencia por ejemplos monoetiqueta;
        - si faltan filas, se completa desde eval_set;
        - nunca usa train_set.
    """
    rng = np.random.default_rng(seed)

    eval_reset = eval_set.reset_index(drop=True).copy()
    eval_reset["__local_eval_id__"] = np.arange(len(eval_reset))

    normal_label = "000 - Normal"
    attack_labels = [label for label in label_cols if label != normal_label]

    selected_ids = set()

    # 1) Normal puro
    if normal_label in eval_reset.columns:
        normal_mask = (
            (eval_reset[normal_label].astype(int) == 1)
            & (eval_reset[attack_labels].astype(int).sum(axis=1) == 0)
        )
        normal_candidates = eval_reset[normal_mask]["__local_eval_id__"].to_numpy().copy()
        rng.shuffle(normal_candidates)

        for idx in normal_candidates[:normal_size]:
            selected_ids.add(int(idx))

    # 2) Ataques por etiqueta
    for label in attack_labels:
        mono_mask = (
            (eval_reset[label].astype(int) == 1)
            & (eval_reset[attack_labels].astype(int).sum(axis=1) == 1)
        )
        candidates = eval_reset[mono_mask]["__local_eval_id__"].to_numpy().copy()

        if len(candidates) == 0:
            candidates = eval_reset[
                eval_reset[label].astype(int) == 1
            ]["__local_eval_id__"].to_numpy().copy()

        rng.shuffle(candidates)

        taken = 0
        for idx in candidates:
            idx = int(idx)

            if idx in selected_ids:
                continue

            selected_ids.add(idx)
            taken += 1

            if taken >= per_label:
                break

    # 3) Completar hasta el tamano objetivo
    if len(selected_ids) < size:
        remaining = [
            int(idx)
            for idx in eval_reset["__local_eval_id__"].tolist()
            if int(idx) not in selected_ids
        ]
        rng.shuffle(remaining)

        for idx in remaining:
            selected_ids.add(idx)

            if len(selected_ids) >= size:
                break

    selected_ids = sorted(list(selected_ids))[:size]

    return eval_reset.loc[selected_ids].drop(columns=["__local_eval_id__"]).copy()


def drop_internal_columns(df):
    return df.drop(columns=["__row_id__"], errors="ignore")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(input_file):
    input_path = Path(input_file)
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Cargando {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)

    initial_rows = len(df)
    initial_cols = len(df.columns)

    feature_cols = df.columns[:24].tolist()
    label_cols_all = df.columns[24:].tolist()

    # Se informa de filas sin features HTTP, pero no se eliminan.
    feature_data = df[feature_cols].fillna("").astype(str).apply(lambda col: col.str.strip())
    empty_rows_mask = (feature_data == "").all(axis=1)
    n_empty_rows = int(empty_rows_mask.sum())

    # Identificador interno para verificar disjuncion.
    df = df.reset_index(drop=True).copy()
    df["__row_id__"] = np.arange(len(df))

    # Excluir solo columnas de etiquetas con insuficiencia muestral.
    excluded_present = [label for label in EXCLUDED_LABELS if label in df.columns]
    df = df.drop(columns=excluded_present)

    label_cols = [label for label in label_cols_all if label not in excluded_present]

    # Construir text_input.
    available_text_fields = [field for field in TEXT_FIELDS if field in df.columns]
    df["text_input"] = df.apply(
        lambda row: build_text_input(row, available_text_fields),
        axis=1,
    )

    # Split 1: eval_set.
    print(f"Construyendo eval_set ({EVAL_SIZE} instancias)...")
    eval_idx = stratified_split(df, label_cols, EVAL_SIZE, SEED)
    eval_set_internal = df.loc[eval_idx].copy()

    # Split 2: train_set disjunto.
    remaining = df.drop(index=eval_idx).reset_index(drop=True)

    print(f"Construyendo train_set ({TRAIN_SIZE} instancias, disjunto de eval_set)...")
    train_idx = stratified_split(remaining, label_cols, TRAIN_SIZE, SEED + 1)
    train_set_internal = remaining.loc[train_idx].copy()

    # Split 3: subconjunto reducido para LLM desde eval_set.
    print(f"Construyendo llm_eval_subset_srbh2020 ({LLM_EVAL_SIZE} instancias, subset de eval_set)...")
    llm_eval_subset_internal = create_llm_eval_subset(
        eval_set=eval_set_internal,
        label_cols=label_cols,
        size=LLM_EVAL_SIZE,
        normal_size=LLM_EVAL_NORMAL_SIZE,
        per_label=LLM_EVAL_PER_LABEL,
        seed=SEED + 2,
    )

    # Verificaciones.
    eval_ids = set(eval_set_internal["__row_id__"].tolist())
    train_ids = set(train_set_internal["__row_id__"].tolist())
    llm_eval_ids = set(llm_eval_subset_internal["__row_id__"].tolist())

    intersection_eval_train = eval_ids & train_ids
    llm_eval_subset_of_eval = llm_eval_ids.issubset(eval_ids)

    # Quitar columnas internas antes de guardar.
    eval_set = drop_internal_columns(eval_set_internal)
    train_set = drop_internal_columns(train_set_internal)
    llm_eval_subset = drop_internal_columns(llm_eval_subset_internal)

    # Guardar salidas.
    eval_set.to_csv(OUT_DIR / "eval_set.csv", index=False)
    train_set.to_csv(OUT_DIR / "train_set.csv", index=False)
    llm_eval_subset.to_csv(OUT_DIR / "llm_eval_subset_srbh2020.csv", index=False)

    write_label_distribution(
        OUT_DIR / "eval_set_label_distribution.csv",
        eval_set,
        label_cols,
    )
    write_label_distribution(
        OUT_DIR / "train_set_label_distribution.csv",
        train_set,
        label_cols,
    )
    write_label_distribution(
        OUT_DIR / "llm_eval_subset_srbh2020_label_distribution.csv",
        llm_eval_subset,
        label_cols,
    )

    # Reporte.
    with (OUT_DIR / "sample_report.txt").open("w", encoding="utf-8") as f:
        f.write("MUESTREO ESTRATIFICADO MULTIETIQUETA - SR-BH 2020\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Archivo de entrada: {input_path}\n")
        f.write(f"Semilla aleatoria: {SEED}\n\n")

        f.write("--- Limpieza inicial ---\n")
        f.write(f"Filas iniciales: {initial_rows}\n")
        f.write(f"Filas con todas las features HTTP vacias detectadas: {n_empty_rows}\n")
        f.write("Accion sobre esas filas: conservadas, no eliminadas\n")
        f.write(f"Filas usadas para el muestreo: {len(df)}\n")
        f.write(f"Columnas iniciales: {initial_cols}\n")
        f.write(f"Etiquetas excluidas por insuficiencia muestral: {excluded_present}\n")
        f.write(f"Etiquetas operativas: {len(label_cols)} -> {label_cols}\n\n")

        f.write("--- Construccion de text_input ---\n")
        f.write("Campos concatenados, solo request y no response:\n")
        for field in available_text_fields:
            f.write(f"  - {field}\n")
        f.write("\n")

        f.write("--- Tamanos de subsets ---\n")
        f.write(f"eval_set:                  {len(eval_set):>6} filas (objetivo {EVAL_SIZE})\n")
        f.write(f"train_set:                 {len(train_set):>6} filas (objetivo {TRAIN_SIZE})\n")
        f.write(f"llm_eval_subset_srbh2020:  {len(llm_eval_subset):>6} filas (objetivo {LLM_EVAL_SIZE})\n\n")

        f.write("--- Verificaciones ---\n")
        f.write(f"|eval_set ∩ train_set| = {len(intersection_eval_train)} (debe ser 0)\n")
        f.write(f"llm_eval_subset_srbh2020 ⊆ eval_set: {llm_eval_subset_of_eval} (debe ser True)\n\n")

        f.write("--- Distribucion de etiquetas: dataset operativo ---\n")
        for label, count in label_distribution(df, label_cols).items():
            f.write(f"  {label}: {int(count)}\n")

        f.write("\n--- Distribucion de etiquetas: eval_set ---\n")
        for label, count in label_distribution(eval_set, label_cols).items():
            f.write(f"  {label}: {int(count)}\n")

        f.write("\n--- Distribucion de etiquetas: train_set ---\n")
        for label, count in label_distribution(train_set, label_cols).items():
            f.write(f"  {label}: {int(count)}\n")

        f.write("\n--- Distribucion de etiquetas: llm_eval_subset_srbh2020 ---\n")
        for label, count in label_distribution(llm_eval_subset, label_cols).items():
            f.write(f"  {label}: {int(count)}\n")

    print()
    print("Muestreo completado.")
    print(f"  eval_set: {len(eval_set)} filas")
    print(f"  train_set: {len(train_set)} filas")
    print(f"  llm_eval_subset_srbh2020: {len(llm_eval_subset)} filas")
    print(f"  eval ∩ train = {len(intersection_eval_train)} (debe ser 0)")
    print(f"  llm_eval ⊆ eval = {llm_eval_subset_of_eval} (debe ser True)")
    print(f"Resultados en: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python 02_sampling_srbh2020.py data_capec_multilabel.csv")
        sys.exit(1)

    main(sys.argv[1])
