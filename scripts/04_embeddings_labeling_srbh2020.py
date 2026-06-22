"""
04_embeddings_labeling_srbh2020.py

Tecnica 4.3: Etiquetado automatico mediante embeddings y similitud coseno
sobre SR-BH 2020.

Diseno:
    - Carga eval_set.csv, generado por 02_sampling_srbh2020.py.
    - Opcionalmente usa train_set.csv para calibrar umbrales por etiqueta.
    - NO entrena un clasificador final.
    - Usa un modelo sentence-transformers congelado para representar peticiones
      HTTP y prototipos textuales de cada categoria CAPEC.
    - Calcula similitud coseno entre cada peticion y cada categoria.
    - Genera predicciones multietiqueta en formato comun pred__<etiqueta>.
    - Usa calibracion conservadora para evitar sobreetiquetado.
    - Si ninguna etiqueta de ataque supera su umbral, predice 000 - Normal.
    - El dataset original no se modifica.

Uso recomendado:
    python 04_embeddings_labeling_srbh2020.py sampling_srbh2020/eval_set.csv --calibration-file sampling_srbh2020/train_set.csv

Uso sin calibracion:
    python 04_embeddings_labeling_srbh2020.py sampling_srbh2020/eval_set.csv

Dependencias:
    python -m pip install pandas numpy scikit-learn sentence-transformers

Salidas en ./embeddings_labeling_srbh2020/:
    - predictions_embeddings.csv
    - embeddings_scores.csv
    - embeddings_thresholds.csv
    - embeddings_summary.txt
    - embeddings_predicted_distribution.csv
    - embeddings_quick_overall_metrics.csv
    - embeddings_quick_per_label_metrics.csv

Notas:
    - CAPEC-248 no se evalua porque fue excluida del universo operativo
      por insuficiencia muestral en el script 02.
    - Las metricas aqui son una comprobacion rapida. La evaluacion final comun
      se realizara posteriormente con 07_evaluation.py.
"""

from pathlib import Path
from urllib.parse import unquote_plus
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

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    print("ERROR: falta sentence-transformers.")
    print("Ejecuta: python -m pip install sentence-transformers")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CONFIGURACION GENERAL
# ---------------------------------------------------------------------------
OUT_DIR = Path("embeddings_labeling_srbh2020")
PRED_PREFIX = "pred__"
SCORE_PREFIX = "score__"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
TOP_K = 1
SEED = 42

# Rango de busqueda para umbrales por etiqueta cuando hay train_set.
MIN_THRESHOLD = 0.10
MAX_THRESHOLD = 0.80
THRESHOLD_STEPS = 151

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

# Solo campos de peticion. No se usa response_* para que el etiquetado dependa
# de lo que envia el atacante, no de como responde el servidor.
INPUT_FIELDS = [
    "request_http_method",
    "request_http_request",
    "request_http_protocol",
    "request_user_agent",
    "request_referer",
    "request_origin",
    "request_cookie",
    "request_content_type",
    "request_body",
]

# Prototipos cortos por etiqueta. Se usan varios por categoria para evitar que
# una sola descripcion genere sesgos fuertes por palabras genericas.
LABEL_PROTOTYPES = {
    "66 - SQL Injection": [
        "sql injection union select information schema database table",
        "or one equals one sql boolean condition select from where",
        "sql payload union all select table name database extraction",
        "waitfor delay sleep benchmark extractvalue updatexml sql injection",
    ],
    "126 - Path Traversal": [
        "path traversal dot dot slash parent directory",
        "directory traversal etc passwd sensitive file access",
        "svn entries wc db git config web inf password properties",
        "encoded traversal percent two e percent two e slash",
    ],
    "242 - Code Injection": [
        "php input auto prepend file php wrapper allow url include",
        "script alert javascript onerror onload xss tag",
        "eval assert base64 decode create function php code",
        "php payload executable code inside parameter or body",
    ],
    "88 - OS Command Injection": [
        "operating system command injection shell command execution",
        "semicolon pipe command wget curl chmod cat bash sh",
        "cat etc passwd wget malware chmod command payload",
        "whoami id uname shell command injection",
    ],
    "16 - Dictionary-based Password Attack": [
        "dictionary password attack login username password",
        "wordpress wp login repeated credentials pwd parameter",
        "brute force login admin root test password attempt",
        "authentication form user password credential guessing",
    ],
    "310 - Scanning for Vulnerable Software": [
        "scanner nmap nikto wpscan vulnerable software detection",
        "probe wordpress plugin theme cgi bin vulnerable path",
        "security scan known vulnerable endpoint enumeration",
        "cfide administrator nmap lower check vulnerability scan",
    ],
    "153 - Input Data Manipulation": [
        "input data manipulation malformed parameter value",
        "tampered content type unexpected value request field",
        "double encoding null byte manipulated input",
        "template injection malformed input data parameter",
    ],
    "274 - HTTP Verb Tampering": [
        "http verb tampering unusual method trace track debug",
        "unexpected http method not implemented method tampering",
        "propfind mkcol lock unlock copy move method abuse",
        "invalid http verb used to bypass access control",
    ],
    "194 - Fake the Source of Data": [
        "external url encoded inside request path",
        "spoofed origin referer source header",
        "google url embedded in local request path",
        "fake referer origin source manipulation",
    ],
    "34 - HTTP Response Splitting": [
        "http response splitting crlf injection response header",
        "carriage return line feed encoded newline header injection",
        "percent zero d percent zero a response splitting",
        "injected header set cookie location content length",
    ],
    "33 - HTTP Request Smuggling": [
        "http request smuggling crlf inside post body",
        "smuggled request transfer encoding chunked content length",
        "desynchronization request body encoded newline",
        "post body contains embedded second request line",
    ],
    "272 - Protocol Manipulation": [
        "protocol manipulation malformed http protocol request",
        "svn wc db git head protocol abuse internal metadata",
        "backdoor localhost remote syslog protocol manipulation",
        "unexpected http protocol version protocol abuse",
    ],
}

# Umbrales de reserva si no se proporciona train_set.
# Son conservadores para evitar que una etiqueta absorba demasiadas predicciones.
DEFAULT_THRESHOLDS = {
    "272 - Protocol Manipulation": 0.35,
    "242 - Code Injection": 0.35,
    "88 - OS Command Injection": 0.34,
    "126 - Path Traversal": 0.35,
    "66 - SQL Injection": 0.33,
    "16 - Dictionary-based Password Attack": 0.38,
    "310 - Scanning for Vulnerable Software": 0.38,
    "153 - Input Data Manipulation": 0.36,
    "274 - HTTP Verb Tampering": 0.38,
    "194 - Fake the Source of Data": 0.36,
    "34 - HTTP Response Splitting": 0.35,
    "33 - HTTP Request Smuggling": 0.37,
}


# ---------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------------------------
def safe_col(label):
    return PRED_PREFIX + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def safe_score_col(label):
    return SCORE_PREFIX + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value)


def decode_repeated(value, rounds=2):
    text = clean_value(value)
    for _ in range(rounds):
        try:
            decoded = unquote_plus(text)
            if decoded == text:
                break
            text = decoded
        except Exception:
            break
    return text


def normalize_text(value):
    text = decode_repeated(value).lower()
    text = text.replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_semantic_input(row):
    """
    Construye una representacion textual para embeddings.
    Se usan campos de peticion y se omiten campos vacios.
    """
    parts = []

    for field in INPUT_FIELDS:
        if field not in row.index:
            continue
        value = normalize_text(row.get(field, ""))
        if value:
            parts.append(f"{field}: {value}")

    return " ".join(parts)


def cosine_similarity_matrix(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)

    return np.dot(a_norm, b_norm.T)


def encode_texts(model, texts, label):
    print(f"Calculando embeddings: {label} ({len(texts)} textos)")
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def build_label_scores(model, rows_df):
    """
    Devuelve:
        semantic_inputs: textos normalizados
        label_scores: matriz n_filas x n_etiquetas_ataque
        prototype_texts: lista de prototipos usados
        prototype_to_label: etiqueta asociada a cada prototipo
    """
    semantic_inputs = [build_semantic_input(row) for _, row in rows_df.iterrows()]

    prototype_texts = []
    prototype_to_label = []

    for label in ATTACK_LABELS:
        for proto in LABEL_PROTOTYPES[label]:
            prototype_texts.append(proto)
            prototype_to_label.append(label)

    prototype_embeddings = encode_texts(model, prototype_texts, "prototipos CAPEC")
    text_embeddings = encode_texts(model, semantic_inputs, "peticiones HTTP")

    proto_scores = cosine_similarity_matrix(text_embeddings, prototype_embeddings)

    label_scores = np.zeros((len(rows_df), len(ATTACK_LABELS)), dtype=float)

    for j, label in enumerate(ATTACK_LABELS):
        proto_indices = [
            idx for idx, proto_label in enumerate(prototype_to_label)
            if proto_label == label
        ]
        label_scores[:, j] = proto_scores[:, proto_indices].max(axis=1)

    return semantic_inputs, label_scores, prototype_texts, prototype_to_label


def label_prediction_cap(prevalence):
    """
    Limite maximo de prediccion permitido durante la calibracion.

    Evita que una etiqueta semanticamente amplia absorba demasiadas instancias.
    """
    if prevalence <= 0:
        return 0.005

    return min(
        0.35,
        max(prevalence * 2.5, prevalence + 0.02, 0.01)
    )


def calibrate_thresholds(train_df, train_scores):
    """
    Calibra umbrales por etiqueta usando train_set.

    La seleccion del umbral busca buen F1, pero descarta umbrales que generan
    una proporcion excesiva de predicciones respecto a la prevalencia real.
    """
    thresholds = {}
    candidate_thresholds = np.linspace(MIN_THRESHOLD, MAX_THRESHOLD, THRESHOLD_STEPS)

    for j, label in enumerate(ATTACK_LABELS):
        y_true = train_df[label].astype(int).values
        scores = train_scores[:, j]

        support = int(y_true.sum())
        prevalence = support / len(y_true) if len(y_true) else 0.0
        max_pred_rate = label_prediction_cap(prevalence)

        best = None
        fallback = None

        for threshold in candidate_thresholds:
            y_pred = (scores >= threshold).astype(int)

            predicted = int(y_pred.sum())
            pred_rate = predicted / len(y_pred) if len(y_pred) else 0.0

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            candidate = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "predicted": predicted,
                "support": support,
                "prevalence": float(prevalence),
                "max_pred_rate": float(max_pred_rate),
                "pred_rate": float(pred_rate),
            }

            if fallback is None:
                fallback = candidate
            else:
                old_distance = abs(fallback["pred_rate"] - max_pred_rate)
                new_distance = abs(pred_rate - max_pred_rate)

                if (
                    new_distance < old_distance
                    or (new_distance == old_distance and f1 > fallback["f1"])
                    or (new_distance == old_distance and f1 == fallback["f1"] and precision > fallback["precision"])
                ):
                    fallback = candidate

            valid = pred_rate <= max_pred_rate and (predicted > 0 or support == 0)

            if not valid:
                continue

            if best is None:
                best = candidate
                continue

            if (
                candidate["f1"] > best["f1"]
                or (candidate["f1"] == best["f1"] and candidate["precision"] > best["precision"])
                or (
                    candidate["f1"] == best["f1"]
                    and candidate["precision"] == best["precision"]
                    and candidate["pred_rate"] < best["pred_rate"]
                )
                or (
                    candidate["f1"] == best["f1"]
                    and candidate["precision"] == best["precision"]
                    and candidate["pred_rate"] == best["pred_rate"]
                    and candidate["threshold"] > best["threshold"]
                )
            ):
                best = candidate

        thresholds[label] = best if best is not None else fallback

    return thresholds

def predict_from_scores(label_scores, thresholds):
    """
    Prediccion multietiqueta:
        - Una etiqueta se activa si score >= threshold propio.
        - Se limita a TOP_K etiquetas de ataque por instancia para evitar
          sobreetiquetado excesivo.
        - Normal se activa si no hay ninguna etiqueta de ataque.
    """
    y_pred_attack = np.zeros(label_scores.shape, dtype=int)

    for i in range(label_scores.shape[0]):
        candidates = []

        for j, label in enumerate(ATTACK_LABELS):
            threshold = thresholds[label]["threshold"]
            score = label_scores[i, j]

            if score >= threshold:
                candidates.append((j, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        for j, _score in candidates[:TOP_K]:
            y_pred_attack[i, j] = 1

    y_pred_normal = (y_pred_attack.sum(axis=1) == 0).astype(int).reshape(-1, 1)
    y_pred_all = np.hstack([y_pred_normal, y_pred_attack])

    return y_pred_all


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


def validate_columns(df, name):
    missing = [label for label in ALL_LABELS if label not in df.columns]
    if missing:
        raise ValueError(f"Faltan etiquetas en {name}: {missing}")

    missing_fields = [field for field in INPUT_FIELDS if field not in df.columns]
    if missing_fields:
        print(f"Aviso: faltan campos de entrada en {name}: {missing_fields}")


def write_thresholds(path, thresholds, source):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label",
            "threshold",
            "calibration_source",
            "calibration_precision",
            "calibration_recall",
            "calibration_f1",
            "calibration_support",
            "calibration_predicted",
            "calibration_prevalence",
            "max_pred_rate",
            "selected_pred_rate",
        ])

        for label in ATTACK_LABELS:
            values = thresholds[label]
            writer.writerow([
                label,
                round(values["threshold"], 6),
                source,
                round(values.get("precision", 0.0), 6),
                round(values.get("recall", 0.0), 6),
                round(values.get("f1", 0.0), 6),
                values.get("support", ""),
                values.get("predicted", ""),
                round(values.get("prevalence", 0.0), 6) if values.get("prevalence", "") != "" else "",
                round(values.get("max_pred_rate", 0.0), 6) if values.get("max_pred_rate", "") != "" else "",
                round(values.get("pred_rate", 0.0), 6) if values.get("pred_rate", "") != "" else "",
            ])


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_file", help="Ruta a sampling_srbh2020/eval_set.csv")
    parser.add_argument(
        "--calibration-file",
        default=None,
        help="Ruta a sampling_srbh2020/train_set.csv para calibrar umbrales",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Modelo sentence-transformers a utilizar",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    calibration_path = Path(args.calibration_file) if args.calibration_file else None

    OUT_DIR.mkdir(exist_ok=True)

    print(f"Cargando eval_set: {eval_path}")
    eval_df = pd.read_csv(eval_path, low_memory=False)
    validate_columns(eval_df, "eval_set")

    if "248 - Command Injection" in eval_df.columns:
        print("Aviso: CAPEC-248 aparece en eval_set, pero no se evalua en esta tecnica.")

    print(f"Filas en eval_set: {len(eval_df)}")
    print(f"Cargando modelo: {args.model}")
    model = SentenceTransformer(args.model)

    # Calibracion opcional con train_set
    calibration_source = "default_thresholds"

    if calibration_path is not None and calibration_path.exists():
        print(f"Cargando calibration/train_set: {calibration_path}")
        train_df = pd.read_csv(calibration_path, low_memory=False)
        validate_columns(train_df, "train_set")

        print(f"Filas en train_set: {len(train_df)}")
        _, train_scores, _, _ = build_label_scores(model, train_df)

        print("Calibrando umbrales por etiqueta con train_set...")
        thresholds = calibrate_thresholds(train_df, train_scores)
        calibration_source = "train_set"
    else:
        if calibration_path is not None:
            print(f"Aviso: no se ha encontrado {calibration_path}. Se usaran umbrales por defecto.")
        else:
            print("No se proporciono calibration-file. Se usaran umbrales por defecto.")

        thresholds = {
            label: {
                "threshold": DEFAULT_THRESHOLDS[label],
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": "",
                "predicted": "",
                "prevalence": "",
                "max_pred_rate": "",
                "pred_rate": "",
            }
            for label in ATTACK_LABELS
        }

    # Prediccion sobre eval_set
    print("Calculando puntuaciones sobre eval_set...")
    semantic_inputs, eval_scores, prototype_texts, prototype_to_label = build_label_scores(model, eval_df)

    print("Generando predicciones sobre eval_set...")
    y_pred_all = predict_from_scores(eval_scores, thresholds)

    overall, per_label = compute_quick_metrics(eval_df, y_pred_all)

    # -----------------------------------------------------------------------
    # Salidas
    # -----------------------------------------------------------------------
    pred_cols = [safe_col(label) for label in ALL_LABELS]
    score_cols = [safe_score_col(label) for label in ATTACK_LABELS]

    predictions_path = OUT_DIR / "predictions_embeddings.csv"
    scores_path = OUT_DIR / "embeddings_scores.csv"
    thresholds_path = OUT_DIR / "embeddings_thresholds.csv"
    predicted_distribution_path = OUT_DIR / "embeddings_predicted_distribution.csv"
    overall_path = OUT_DIR / "embeddings_quick_overall_metrics.csv"
    per_label_path = OUT_DIR / "embeddings_quick_per_label_metrics.csv"
    summary_path = OUT_DIR / "embeddings_summary.txt"

    pred_df = eval_df.copy()
    pred_df["semantic_input"] = semantic_inputs

    for i, label in enumerate(ALL_LABELS):
        pred_df[safe_col(label)] = y_pred_all[:, i].astype(int)

    pred_df.to_csv(predictions_path, index=False)

    scores_df = pd.DataFrame(eval_scores, columns=score_cols)
    scores_df.insert(0, "row_id", np.arange(len(scores_df)))
    scores_df.to_csv(scores_path, index=False)

    write_thresholds(thresholds_path, thresholds, calibration_source)

    with predicted_distribution_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "predicted_count", "percentage_eval_set"])
        for i, label in enumerate(ALL_LABELS):
            count = int(y_pred_all[:, i].sum())
            pct = (count / len(eval_df) * 100) if len(eval_df) else 0
            writer.writerow([label, count, round(pct, 6)])

    with overall_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in overall.items():
            writer.writerow([key, round(value, 6) if isinstance(value, float) else value])

    with per_label_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "label", "support", "predicted", "precision", "recall", "f1", "kappa"
        ])
        writer.writeheader()
        for row in per_label:
            writer.writerow({
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in row.items()
            })

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("RESUMEN DE ETIQUETADO MEDIANTE EMBEDDINGS Y SIMILITUD COSENO\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Archivo de evaluacion: {eval_path}\n")
        f.write(f"Archivo de calibracion: {calibration_path if calibration_path else 'no usado'}\n")
        f.write(f"Fuente de umbrales: {calibration_source}\n")
        f.write(f"Filas evaluadas: {len(eval_df)}\n")
        f.write(f"Modelo utilizado: {args.model}\n")
        f.write(f"TOP_K: {TOP_K}\n\n")

        f.write("--- Campos usados para semantic_input ---\n")
        for field in INPUT_FIELDS:
            f.write(f"  - {field}\n")

        f.write("\n--- Etiquetas evaluadas ---\n")
        for label in ALL_LABELS:
            f.write(f"  - {label}\n")

        f.write("\n--- Umbrales por etiqueta ---\n")
        for label in ATTACK_LABELS:
            values = thresholds[label]
            f.write(f"  - {label}: {values['threshold']:.6f}\n")

        f.write("\n--- Prototipos por etiqueta ---\n")
        for label in ATTACK_LABELS:
            f.write(f"\n[{label}]\n")
            for proto in LABEL_PROTOTYPES[label]:
                f.write(f"  - {proto}\n")

        f.write("\n--- Distribucion de etiquetas predichas ---\n")
        for i, label in enumerate(ALL_LABELS):
            count = int(y_pred_all[:, i].sum())
            f.write(f"  - {label}: {count}\n")

        f.write("\n--- Metricas rapidas sobre eval_set ---\n")
        for key, value in overall.items():
            if isinstance(value, float):
                f.write(f"  - {key}: {value:.6f}\n")
            else:
                f.write(f"  - {key}: {value}\n")

        f.write("\n--- Nota metodologica ---\n")
        f.write(
            "Esta tecnica compara embeddings de cada peticion HTTP con prototipos "
            "textuales asociados a cada categoria CAPEC. El modelo se usa congelado "
            "y no se entrena un clasificador final. Si se proporciona train_set, este "
            "se usa solo para calibrar umbrales por etiqueta, manteniendo eval_set como "
            "conjunto comun de evaluacion para la comparativa. La calibracion incorpora "
            "un limite de predicciones por etiqueta para reducir sobreetiquetado en "
            "categorias semanticamente amplias.\n"
        )

    print()
    print("Etiquetado por embeddings completado.")
    print(f"Predicciones: {predictions_path}")
    print(f"Resumen: {summary_path}")
    print(f"Micro-F1 rapido: {overall['micro_f1']:.4f}")
    print(f"Macro-F1 rapido: {overall['macro_f1']:.4f}")
    print(f"Hamming loss rapido: {overall['hamming_loss']:.4f}")
    print(f"Jaccard rapido: {overall['jaccard_samples']:.4f}")
    print(f"Subset accuracy rapido: {overall['subset_accuracy']:.4f}")


if __name__ == "__main__":
    main()
