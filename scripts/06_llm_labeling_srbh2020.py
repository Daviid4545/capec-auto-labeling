"""
06_llm_labeling_srbh2020.py

Tecnica 4.5: Etiquetado zero-shot/few-shot con LLM local sobre SR-BH 2020.

Diseño:
    - Usa preferentemente llm_eval_subset_srbh2020.csv, generado por 02_sampling_srbh2020.py.
    - Ejecuta un modelo local mediante Ollama, sin API externa de pago.
    - Permite dos modos:
        1) zero_shot: lista cerrada de etiquetas + definiciones breves.
        2) few_shot: lo anterior + ejemplos etiquetados tomados de train_set.csv.
    - No entrena un clasificador final.
    - El modelo debe elegir etiquetas entre las 12 CAPEC operativas o "000 - Normal".
    - Guarda respuestas crudas, predicciones normalizadas y métricas rápidas.
    - Soporta reanudación mediante caché JSONL separada por modo/modelo/firma de prompt.
    - Usa temperatura baja y formato JSON de Ollama para favorecer reproducibilidad.
    - Las etiquetas originales solo se usan para métricas rápidas al final.

Uso zero-shot recomendado:
    python 06_llm_labeling_srbh2020.py sampling_srbh2020/llm_eval_subset_srbh2020.csv --model qwen2.5:7b-instruct --prompt-mode zero_shot

Uso few-shot recomendado:
    python 06_llm_labeling_srbh2020.py sampling_srbh2020/llm_eval_subset_srbh2020.csv --model qwen2.5:7b-instruct --prompt-mode few_shot --examples-file sampling_srbh2020/train_set.csv

Prueba rápida con 25 filas:
    python 06_llm_labeling_srbh2020.py sampling_srbh2020/llm_eval_subset_srbh2020.csv --model qwen2.5:7b-instruct --prompt-mode few_shot --examples-file sampling_srbh2020/train_set.csv --limit 25

Si se quiere ampliar a 100 filas, no hace falta --limit porque llm_eval_subset_srbh2020.csv ya tiene 100 instancias.

Dependencias:
    python -m pip install pandas numpy scikit-learn requests

Salidas en ./llm_labeling_srbh2020/:
    - predictions_llm_<modo>.csv
    - llm_raw_responses_<modo>_<modelo>.jsonl
    - llm_summary_<modo>.txt
    - llm_predicted_distribution_<modo>.csv
    - llm_quick_overall_metrics_<modo>.csv
    - llm_quick_per_label_metrics_<modo>.csv
    - llm_prompt_template_<modo>.txt
    - llm_few_shot_examples.csv  (solo si prompt-mode=few_shot)
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

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
OUT_DIR = Path("llm_labeling_srbh2020")
PRED_PREFIX = "pred__"

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

TEMPERATURE = 0.0
TOP_P = 0.9
SEED = 42
MAX_RETRIES = 2
REQUEST_TIMEOUT = 240
SLEEP_BETWEEN_REQUESTS = 0.05
NUM_PREDICT = 220

MAX_ATTACK_LABELS = 2

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

LABEL_GUIDE = {
    "272 - Protocol Manipulation": "manipulacion del protocolo o uso anomalo de HTTP/metadatos internos: .svn, .git, HTTP raro, backdoor, remote_syslog.",
    "242 - Code Injection": "inyeccion de codigo o script: <script>, javascript:, onerror, PHP, php://input, eval, base64_decode.",
    "88 - OS Command Injection": "comandos del sistema: ;cat, |wget, curl, chmod, bash, sh, whoami, uname, /bin/sh.",
    "126 - Path Traversal": "rutas o ficheros sensibles: ../, ..\\\\, /etc/passwd, wp-config.php, .git, .svn, WEB-INF.",
    "66 - SQL Injection": "payload SQL: UNION SELECT, information_schema, OR 1=1, SELECT FROM, sleep(), benchmark(), waitfor delay.",
    "16 - Dictionary-based Password Attack": "login/fuerza bruta/diccionario: wp-login.php, /login, user, username, pwd, password.",
    "310 - Scanning for Vulnerable Software": "escaneo/enumeracion de software vulnerable: nmap, nikto, wpscan, cgi-bin, cfide, rutas de plugins o pruebas conocidas.",
    "153 - Input Data Manipulation": "manipulacion generica de entrada: null byte, doble encoding, parametros malformados, template injection.",
    "274 - HTTP Verb Tampering": "metodos HTTP raros o abusivos: TRACE, TRACK, DEBUG, PROPFIND, CONNECT, PATCH, etc.",
    "194 - Fake the Source of Data": "origen falsificado o sospechoso: URL externa incrustada, referer/origin raro, localhost, file://, google externo.",
    "34 - HTTP Response Splitting": "CRLF/header injection: %0d%0a, %0a, set-cookie, location, content-length inyectado.",
    "33 - HTTP Request Smuggling": "smuggling/desincronizacion: CRLF en body POST, Transfer-Encoding chunked, Content-Length conflictivo.",
}

# Normalizacion de respuestas imperfectas del modelo.
LABEL_ALIASES = {
    "normal": NORMAL_LABEL,
    "000": NORMAL_LABEL,
    "no attack": NORMAL_LABEL,
    "benign": NORMAL_LABEL,

    "sql injection": "66 - SQL Injection",
    "sqli": "66 - SQL Injection",
    "66": "66 - SQL Injection",

    "path traversal": "126 - Path Traversal",
    "directory traversal": "126 - Path Traversal",
    "126": "126 - Path Traversal",

    "code injection": "242 - Code Injection",
    "xss": "242 - Code Injection",
    "cross site scripting": "242 - Code Injection",
    "242": "242 - Code Injection",

    "os command injection": "88 - OS Command Injection",
    "command injection": "88 - OS Command Injection",
    "command execution": "88 - OS Command Injection",
    "88": "88 - OS Command Injection",

    "dictionary-based password attack": "16 - Dictionary-based Password Attack",
    "dictionary password attack": "16 - Dictionary-based Password Attack",
    "brute force": "16 - Dictionary-based Password Attack",
    "password attack": "16 - Dictionary-based Password Attack",
    "16": "16 - Dictionary-based Password Attack",

    "scanning for vulnerable software": "310 - Scanning for Vulnerable Software",
    "vulnerability scanning": "310 - Scanning for Vulnerable Software",
    "scanner": "310 - Scanning for Vulnerable Software",
    "310": "310 - Scanning for Vulnerable Software",

    "input data manipulation": "153 - Input Data Manipulation",
    "input manipulation": "153 - Input Data Manipulation",
    "153": "153 - Input Data Manipulation",

    "http verb tampering": "274 - HTTP Verb Tampering",
    "verb tampering": "274 - HTTP Verb Tampering",
    "274": "274 - HTTP Verb Tampering",

    "fake the source of data": "194 - Fake the Source of Data",
    "fake source": "194 - Fake the Source of Data",
    "source spoofing": "194 - Fake the Source of Data",
    "194": "194 - Fake the Source of Data",

    "http response splitting": "34 - HTTP Response Splitting",
    "response splitting": "34 - HTTP Response Splitting",
    "crlf injection": "34 - HTTP Response Splitting",
    "34": "34 - HTTP Response Splitting",

    "http request smuggling": "33 - HTTP Request Smuggling",
    "request smuggling": "33 - HTTP Request Smuggling",
    "33": "33 - HTTP Request Smuggling",

    "protocol manipulation": "272 - Protocol Manipulation",
    "272": "272 - Protocol Manipulation",
}


# ---------------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------------
def build_labels_block():
    lines = [f"- {NORMAL_LABEL}: petición normal o sin indicios técnicos suficientes de ataque."]
    for label in ATTACK_LABELS:
        lines.append(f"- {label}: {LABEL_GUIDE[label]}")
    return "\n".join(lines)


def format_example(example):
    labels_json = json.dumps(example["labels"], ensure_ascii=False)
    text = example["text_input"]

    return f"""Ejemplo:
Petición HTTP:
\"\"\"{text}\"\"\"

Salida correcta:
{{"labels": {labels_json}, "reason": "señal técnica compatible con la etiqueta indicada"}}"""


def build_examples_block(few_shot_examples):
    if not few_shot_examples:
        return ""
    return "\n\n".join(format_example(ex) for ex in few_shot_examples)


def build_prompt(row_text, prompt_mode, few_shot_examples=None):
    labels_block = build_labels_block()
    examples_block = build_examples_block(few_shot_examples or [])

    few_shot_section = ""
    if prompt_mode == "few_shot":
        few_shot_section = f"""
Ejemplos etiquetados de referencia. Úsalos solo como guía de formato y criterio, no como lista cerrada de casos:
{examples_block}

Ahora etiqueta la nueva petición.
"""

    return f"""Eres un analista de ciberseguridad. Debes etiquetar UNA petición HTTP usando únicamente las etiquetas permitidas.

Etiquetas permitidas:
{labels_block}

Criterios obligatorios:
1. Devuelve SOLO JSON válido, sin Markdown y sin texto adicional.
2. Usa máximo {MAX_ATTACK_LABELS} etiquetas de ataque.
3. Si no hay una señal técnica clara, usa solo "{NORMAL_LABEL}".
4. No inventes etiquetas fuera de la lista.
5. Basa la decisión en la petición HTTP, no en suposiciones externas.
6. No clasifiques como ataque por una palabra aislada si no hay patrón técnico.
7. Si hay varias señales claras, puedes devolver dos etiquetas como máximo.
8. Si devuelves una etiqueta de ataque, no incluyas "{NORMAL_LABEL}".

{few_shot_section}
Petición HTTP a etiquetar:
\"\"\"{row_text}\"\"\"

Devuelve exactamente este esquema JSON:
{{
  "labels": ["000 - Normal"],
  "reason": "motivo breve basado en la señal técnica observada"
}}
"""


# ---------------------------------------------------------------------------
# FEW-SHOT EXAMPLES
# ---------------------------------------------------------------------------
def labels_from_row(row):
    active = [label for label in ALL_LABELS if int(row.get(label, 0)) == 1]
    attacks = [label for label in active if label != NORMAL_LABEL]
    if attacks:
        return attacks[:MAX_ATTACK_LABELS]
    return [NORMAL_LABEL]


def shorten_text(text, max_chars):
    text = str(text).strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def example_quality_score(text):
    """
    Priorizamos ejemplos cortos pero con señales técnicas visibles.
    """
    txt = str(text).lower()
    score = 0

    useful_tokens = [
        "union", "select", "information_schema", "../", "%2e%2e", "/etc/passwd",
        "wp-login", "password", "pwd=", "user=", "trace", "track", "%0d", "%0a",
        "script", "javascript", "php://", "eval", "wget", "curl", "chmod",
        "nmap", "nikto", "cgi-bin", "cfide", "http://", "https://",
        ".svn", ".git", "content-length", "transfer-encoding",
    ]

    for token in useful_tokens:
        if token in txt:
            score += 2

    # Penalizar textos demasiado largos.
    score -= min(len(txt) / 1000, 3)

    return score


def select_few_shot_examples(examples_file, examples_per_label, normal_examples, max_example_chars):
    rng = random.Random(SEED)
    examples_path = Path(examples_file)

    if not examples_path.exists():
        raise FileNotFoundError(f"No existe examples-file: {examples_path}")

    df = pd.read_csv(examples_path, low_memory=False)

    missing = [label for label in ALL_LABELS if label not in df.columns]
    if missing:
        raise ValueError(f"Faltan etiquetas en examples-file: {missing}")

    if "text_input" not in df.columns:
        raise ValueError("Falta text_input en examples-file")

    selected = []
    selected_indices = set()

    # Normales: preferir textos relativamente cortos y sin etiquetas de ataque.
    normal_candidates = df[
        (df[NORMAL_LABEL].astype(int) == 1)
        & (df[ATTACK_LABELS].astype(int).sum(axis=1) == 0)
    ].copy()

    normal_candidates["_score"] = normal_candidates["text_input"].astype(str).apply(
        lambda x: -abs(len(x) - 180)
    )

    normal_indices = normal_candidates.sort_values("_score", ascending=False).index.tolist()

    for idx in normal_indices[:normal_examples]:
        row = df.loc[idx]
        selected.append({
            "source_index": int(idx),
            "target_label": NORMAL_LABEL,
            "labels": [NORMAL_LABEL],
            "text_input": shorten_text(row["text_input"], max_example_chars),
        })
        selected_indices.add(idx)

    # Ataques: preferir ejemplos monoetiqueta y con señales visibles.
    for label in ATTACK_LABELS:
        mono = df[
            (df[label].astype(int) == 1)
            & (df[ATTACK_LABELS].astype(int).sum(axis=1) == 1)
        ].copy()

        if len(mono) == 0:
            candidates_df = df[df[label].astype(int) == 1].copy()
        else:
            candidates_df = mono

        if len(candidates_df) == 0:
            continue

        candidates_df["_score"] = candidates_df["text_input"].astype(str).apply(example_quality_score)
        candidate_indices = candidates_df.sort_values("_score", ascending=False).index.tolist()

        taken = 0
        for idx in candidate_indices:
            if idx in selected_indices:
                continue

            row = df.loc[idx]
            labs = labels_from_row(row)

            selected.append({
                "source_index": int(idx),
                "target_label": label,
                "labels": labs,
                "text_input": shorten_text(row["text_input"], max_example_chars),
            })
            selected_indices.add(idx)
            taken += 1

            if taken >= examples_per_label:
                break

    return selected


def write_few_shot_examples(path, examples):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_index", "target_label", "labels", "text_input"
        ])
        writer.writeheader()
        for ex in examples:
            writer.writerow({
                "source_index": ex["source_index"],
                "target_label": ex["target_label"],
                "labels": ";".join(ex["labels"]),
                "text_input": ex["text_input"],
            })


# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------
def safe_col(label):
    return PRED_PREFIX + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def validate_columns(df):
    missing = [label for label in ALL_LABELS if label not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de etiquetas en llm_subset: {missing}")
    if "text_input" not in df.columns:
        raise ValueError("Falta la columna text_input en llm_subset.csv")


def prompt_signature(prompt_mode, model, few_shot_examples, json_format):
    base = {
        "prompt_mode": prompt_mode,
        "model": model,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "max_attack_labels": MAX_ATTACK_LABELS,
        "few_shot_examples": few_shot_examples,
        "labels": ALL_LABELS,
        "json_format": json_format,
    }
    encoded = json.dumps(base, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def ollama_generate(prompt, model, url, use_json_format=True):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": SEED,
            "num_predict": NUM_PREDICT,
        },
    }

    if use_json_format:
        payload["format"] = "json"

    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


def extract_json(text):
    if text is None:
        raise ValueError("respuesta vacia")

    raw = str(text).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("no se encontro bloque JSON")

    candidate = match.group(0)
    candidate = candidate.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON no valido: {exc}") from exc


def normalize_one_label(item):
    item_raw = str(item).strip()
    item_lower = item_raw.lower().strip()

    if item_raw in ALL_LABELS:
        return item_raw

    if item_lower in LABEL_ALIASES:
        return LABEL_ALIASES[item_lower]

    # Permitir "CAPEC-66", "CAPEC 66", "66 SQL Injection"
    capec_match = re.search(r"(?:capec[-\s]*)?(\d{1,3})", item_lower)
    if capec_match:
        capec_id = capec_match.group(1)
        for label in ALL_LABELS:
            if label.startswith(capec_id + " -"):
                return label

    # Coincidencia por nombre parcial.
    for label in ATTACK_LABELS:
        label_name = label.split(" - ", 1)[1].lower()
        if label_name in item_lower or item_lower in label_name:
            return label

    return None


def normalize_labels(parsed):
    labels = parsed.get("labels", [])

    if isinstance(labels, str):
        labels = [labels]

    if not isinstance(labels, list):
        labels = []

    normalized = []

    for item in labels:
        label = normalize_one_label(item)
        if label is not None:
            normalized.append(label)

    clean = []
    for label in normalized:
        if label not in clean:
            clean.append(label)

    attacks = [label for label in clean if label != NORMAL_LABEL]

    if attacks:
        return attacks[:MAX_ATTACK_LABELS]

    return [NORMAL_LABEL]


def load_cache(cache_path, signature):
    cache = {}
    if not cache_path.exists():
        return cache

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("prompt_signature") != signature:
                    continue
                cache[int(obj["row_id"])] = obj
            except Exception:
                continue
    return cache


def append_cache(cache_path, obj):
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def compute_quick_metrics(df, y_pred_all):
    y_true = df[ALL_LABELS].astype(int).values
    overall = {
        "rows": len(df),
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
        writer.writerow(["label", "predicted_count", "percentage_llm_subset"])
        n = y_pred_all.shape[0]
        for i, label in enumerate(ALL_LABELS):
            count = int(y_pred_all[:, i].sum())
            pct = (count / n * 100) if n else 0
            writer.writerow([label, count, round(pct, 6)])


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("llm_subset_file", help="Ruta a sampling_srbh2020/llm_eval_subset_srbh2020.csv o llm_subset.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo local de Ollama")
    parser.add_argument("--url", default=OLLAMA_URL, help="URL local de Ollama")
    parser.add_argument(
        "--prompt-mode",
        choices=["zero_shot", "few_shot"],
        default="zero_shot",
        help="Modo de prompting",
    )
    parser.add_argument("--examples-file", default=None, help="train_set.csv para construir ejemplos few-shot")
    parser.add_argument("--examples-per-label", type=int, default=1, help="Ejemplos few-shot por etiqueta de ataque")
    parser.add_argument("--normal-examples", type=int, default=2, help="Ejemplos Normal en few-shot")
    parser.add_argument("--max-example-chars", type=int, default=280, help="Longitud maxima de cada ejemplo few-shot")
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo N filas")
    parser.add_argument("--start", type=int, default=0, help="Fila inicial")
    parser.add_argument("--no-json-format", action="store_true", help="No usar format=json de Ollama")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    if args.prompt_mode == "few_shot" and not args.examples_file:
        raise ValueError("Para --prompt-mode few_shot debes indicar --examples-file sampling_srbh2020/train_set.csv")

    input_path = Path(args.llm_subset_file)
    print(f"Cargando llm_subset: {input_path}")
    original_df = pd.read_csv(input_path, low_memory=False)
    validate_columns(original_df)

    original_df = original_df.copy()
    original_df["__row_id__"] = np.arange(len(original_df))

    if args.limit is not None:
        end = args.start + args.limit
        df = original_df.iloc[args.start:end].copy()
        print(f"Procesando subconjunto: start={args.start}, limit={args.limit}, filas={len(df)}")
    else:
        df = original_df.iloc[args.start:].copy()
        if args.start:
            print(f"Procesando desde start={args.start}, filas={len(df)}")
    df = df.reset_index(drop=True)

    few_shot_examples = []
    if args.prompt_mode == "few_shot":
        print(f"Seleccionando ejemplos few-shot desde: {args.examples_file}")
        few_shot_examples = select_few_shot_examples(
            args.examples_file,
            args.examples_per_label,
            args.normal_examples,
            args.max_example_chars,
        )
        write_few_shot_examples(OUT_DIR / "llm_few_shot_examples.csv", few_shot_examples)
        print(f"Ejemplos few-shot seleccionados: {len(few_shot_examples)}")

    use_json_format = not args.no_json_format
    signature = prompt_signature(args.prompt_mode, args.model, few_shot_examples, use_json_format)
    model_safe = safe_name(args.model)

    output_suffix = args.prompt_mode
    cache_path = OUT_DIR / f"llm_raw_responses_{output_suffix}_{model_safe}.jsonl"
    cache = load_cache(cache_path, signature)

    prompt_template_path = OUT_DIR / f"llm_prompt_template_{output_suffix}.txt"
    prompt_template_path.write_text(
        build_prompt("{text_input}", args.prompt_mode, few_shot_examples),
        encoding="utf-8",
    )

    predictions = []
    raw_status_counts = {"from_cache": 0, "ok": 0, "fallback_normal": 0, "error": 0}

    print(f"Modelo Ollama: {args.model}")
    print(f"Modo de prompt: {args.prompt_mode}")
    print(f"Formato JSON Ollama: {use_json_format}")
    print(f"Filas a etiquetar: {len(df)}")

    for local_idx, row in df.iterrows():
        row_id = int(row["__row_id__"])
        row_text = str(row.get("text_input", ""))

        if row_id in cache:
            cached = cache[row_id]
            labels = normalize_labels({"labels": cached.get("labels", [NORMAL_LABEL])})
            reason = cached.get("reason", "")
            status = "from_cache"
            raw_response = cached.get("raw_response", "")
            raw_status_counts["from_cache"] += 1
        else:
            prompt = build_prompt(row_text, args.prompt_mode, few_shot_examples)
            raw_response = ""
            labels = [NORMAL_LABEL]
            reason = ""
            status = "error"

            for attempt in range(MAX_RETRIES + 1):
                try:
                    raw_response = ollama_generate(prompt, args.model, args.url, use_json_format)
                    parsed = extract_json(raw_response)
                    labels = normalize_labels(parsed)
                    reason = str(parsed.get("reason", "")).strip()
                    status = "ok"
                    raw_status_counts["ok"] += 1
                    break
                except Exception as exc:
                    reason = f"parse_or_request_error: {exc}"
                    status = "error"
                    if attempt < MAX_RETRIES:
                        time.sleep(0.5)

            if status != "ok":
                labels = [NORMAL_LABEL]
                raw_status_counts["fallback_normal"] += 1
                raw_status_counts["error"] += 1

            cache_obj = {
                "row_id": row_id,
                "prompt_signature": signature,
                "prompt_mode": args.prompt_mode,
                "json_format": use_json_format,
                "model": args.model,
                "labels": labels,
                "reason": reason,
                "raw_response": raw_response,
                "status": status,
            }
            append_cache(cache_path, cache_obj)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        pred_record = {safe_col(label): 0 for label in ALL_LABELS}
        for label in labels:
            if label in ALL_LABELS:
                pred_record[safe_col(label)] = 1
        if sum(pred_record.values()) == 0:
            pred_record[safe_col(NORMAL_LABEL)] = 1

        predictions.append({
            **pred_record,
            "llm_labels": ";".join(labels),
            "llm_reason": reason,
            "llm_status": status,
            "llm_raw_response": raw_response,
        })

        if (local_idx + 1) % 25 == 0:
            print(f"Procesadas {local_idx + 1}/{len(df)} filas")

    pred_df = pd.DataFrame(predictions)
    out_df = pd.concat([df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

    y_pred_all = out_df[[safe_col(label) for label in ALL_LABELS]].astype(int).values
    overall, per_label = compute_quick_metrics(out_df, y_pred_all)

    predictions_path = OUT_DIR / f"predictions_llm_{output_suffix}.csv"
    summary_path = OUT_DIR / f"llm_summary_{output_suffix}.txt"
    distribution_path = OUT_DIR / f"llm_predicted_distribution_{output_suffix}.csv"
    overall_path = OUT_DIR / f"llm_quick_overall_metrics_{output_suffix}.csv"
    per_label_path = OUT_DIR / f"llm_quick_per_label_metrics_{output_suffix}.csv"

    out_df.to_csv(predictions_path, index=False)
    write_distribution(distribution_path, y_pred_all)
    write_overall_metrics(overall_path, overall)
    write_per_label_metrics(per_label_path, per_label)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("RESUMEN DE ETIQUETADO ZERO-SHOT / FEW-SHOT CON LLM LOCAL\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Archivo de entrada: {input_path}\n")
        f.write(f"Filas procesadas: {len(df)}\n")
        f.write(f"Modo de prompt: {args.prompt_mode}\n")
        f.write(f"Archivo de ejemplos few-shot: {args.examples_file if args.examples_file else 'no usado'}\n")
        f.write(f"Numero de ejemplos few-shot: {len(few_shot_examples)}\n")
        f.write(f"Modelo Ollama: {args.model}\n")
        f.write(f"URL Ollama: {args.url}\n")
        f.write(f"Formato JSON Ollama: {use_json_format}\n")
        f.write(f"Temperatura: {TEMPERATURE}\n")
        f.write(f"Top-p: {TOP_P}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"MAX_ATTACK_LABELS: {MAX_ATTACK_LABELS}\n")
        f.write(f"Prompt signature: {signature}\n\n")

        f.write("--- Estado de respuestas ---\n")
        for key, value in raw_status_counts.items():
            f.write(f"  - {key}: {value}\n")

        if few_shot_examples:
            f.write("\n--- Ejemplos few-shot usados ---\n")
            for ex in few_shot_examples:
                f.write(f"  - {ex['target_label']} -> {ex['labels']} (source_index={ex['source_index']})\n")

        f.write("\n--- Etiquetas evaluadas ---\n")
        for label in ALL_LABELS:
            f.write(f"  - {label}\n")

        f.write("\n--- Distribucion de etiquetas predichas ---\n")
        for i, label in enumerate(ALL_LABELS):
            f.write(f"  - {label}: {int(y_pred_all[:, i].sum())}\n")

        f.write("\n--- Metricas rapidas sobre subconjunto LLM ---\n")
        for key, value in overall.items():
            if isinstance(value, float):
                f.write(f"  - {key}: {value:.6f}\n")
            else:
                f.write(f"  - {key}: {value}\n")

        f.write("\n--- Nota metodologica ---\n")
        f.write(
            "Esta tecnica usa un modelo de lenguaje local mediante Ollama, sin API externa "
            "de pago. El modelo no se ajusta ni se entrena; se utiliza mediante prompting "
            "zero-shot o few-shot con una lista cerrada de etiquetas CAPEC. En few-shot, "
            "los ejemplos proceden exclusivamente de train_set.csv, no de llm_subset.csv, "
            "para evitar contaminacion de la evaluacion. Se usa temperatura 0 y formato JSON "
            "cuando Ollama lo permite, y se guardan las respuestas crudas para trazabilidad. "
            "La evaluacion se realiza sobre un subconjunto LLM extraido de eval_set.\n"
        )

    print()
    print("Etiquetado LLM completado.")
    print(f"Predicciones: {predictions_path}")
    print(f"Resumen: {summary_path}")
    print(f"Micro-F1 rapido: {overall['micro_f1']:.4f}")
    print(f"Macro-F1 rapido: {overall['macro_f1']:.4f}")
    print(f"Hamming loss rapido: {overall['hamming_loss']:.4f}")
    print(f"Jaccard rapido: {overall['jaccard_samples']:.4f}")
    print(f"Subset accuracy rapido: {overall['subset_accuracy']:.4f}")


if __name__ == "__main__":
    main()
