"""
03_rules_labeling_srbh2020.py

Tecnica 4.2: Etiquetado automatico por reglas y heuristicas sobre SR-BH 2020.

Diseno:
    - Carga eval_set.csv, generado por 02_sampling_srbh2020.py.
    - Aplica reglas explicitas por etiqueta CAPEC.
    - Enfoque multietiqueta: una instancia puede activar varias etiquetas.
    - Estrategia agresiva moderada: se prioriza recall, pero evitando reglas
      demasiado generales que generen falsos positivos masivos.
    - Si ninguna regla de ataque dispara, se predice 000 - Normal.
    - El dataset original no se modifica.

Uso:
    python 03_rules_labeling_srbh2020.py sampling_srbh2020/eval_set.csv

Salidas en ./rules_labeling_srbh2020/
    - predictions_rules.csv
    - rules_summary.txt
    - rules_dictionary.csv
    - rules_predicted_distribution.csv
    - rules_quick_overall_metrics.csv
    - rules_quick_per_label_metrics.csv

Notas:
    - CAPEC-248 no se evalua porque fue excluida del universo operativo
      por insuficiencia muestral en el script 02.
    - Las metricas aqui son una comprobacion rapida de la tecnica.
      La evaluacion final comun se realizara posteriormente con 08_evaluation_srbh2020.py.
"""

from pathlib import Path
from urllib.parse import unquote_plus
import sys
import re
import csv
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


OUT_DIR = Path("rules_labeling_srbh2020")
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

REQUEST_FIELDS = [
    "request_http_method",
    "request_http_request",
    "request_http_protocol",
    "request_user_agent",
    "request_referer",
    "request_host",
    "request_origin",
    "request_cookie",
    "request_content_type",
    "request_accept",
    "request_accept_language",
    "request_accept_encoding",
    "request_do_not_track",
    "request_connection",
    "request_body",
]


RULES = [
    # ------------------------------------------------------------------
    # CAPEC-66 SQL Injection
    # ------------------------------------------------------------------
    {
        "id": "SQLI_01",
        "label": "66 - SQL Injection",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"\bunion\s+(all\s+)?select\b",
        "description": "Uso de UNION SELECT, patron clasico de SQL Injection.",
    },
    {
        "id": "SQLI_02",
        "label": "66 - SQL Injection",
        "field": "all",
        "type": "keyword_decoded",
        "pattern": "information_schema",
        "description": "Referencia a information_schema, frecuente en extraccion de metadatos SQL.",
    },
    {
        "id": "SQLI_03",
        "label": "66 - SQL Injection",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"(\bor\b|\band\b)\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+",
        "description": "Condiciones booleanas del tipo OR 1=1 o AND 1=1.",
    },
    {
        "id": "SQLI_04",
        "label": "66 - SQL Injection",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"\b(select|insert|update|delete|drop)\b.{0,80}\b(from|into|set|table|database)\b",
        "description": "Combinaciones de palabras clave SQL en una misma carga.",
    },
    {
        "id": "SQLI_05",
        "label": "66 - SQL Injection",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"(\b(select|union|insert|update|delete|drop|where|from)\b.{0,80}(--|#|/\*)|(--|#|/\*).{0,80}\b(select|union|insert|update|delete|drop|where|from)\b)",
        "description": "Comentarios SQL, pero solo cuando aparecen cerca de palabras clave SQL para evitar falsos positivos.",
    },
    {
        "id": "SQLI_06",
        "label": "66 - SQL Injection",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"\b(sleep|benchmark|extractvalue|updatexml|load_file)\s*\(|\bwaitfor\s+delay\b",
        "description": "Funciones SQL habituales en inyecciones o pruebas temporales.",
    },

    # ------------------------------------------------------------------
    # CAPEC-126 Path Traversal
    # ------------------------------------------------------------------
    {
        "id": "PT_01",
        "label": "126 - Path Traversal",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"\.\.[\\/]",
        "description": "Secuencias ../ o ..\\ para salto de directorios.",
    },
    {
        "id": "PT_02",
        "label": "126 - Path Traversal",
        "field": "all",
        "type": "keyword_decoded",
        "pattern": "/etc/passwd",
        "description": "Acceso a fichero sensible de sistemas Unix.",
    },
    {
        "id": "PT_03",
        "label": "126 - Path Traversal",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"(\.svn/|\.git/|web-inf|wp-config\.php|boot\.ini|win\.ini|password\.properties)",
        "description": "Acceso a ficheros internos o de configuracion mediante ruta.",
    },
    {
        "id": "PT_04",
        "label": "126 - Path Traversal",
        "field": "all_raw",
        "type": "regex_raw",
        "pattern": r"(%2e%2e(%2f|%5c)|%252e%252e(%252f|%255c)|\.\.%2f|\.\.%5c)",
        "description": "Traversal URL-encoded. Se evita disparar por cualquier %2f aislado porque aparece en URLs normales.",
    },

    # ------------------------------------------------------------------
    # CAPEC-242 Code Injection
    # ------------------------------------------------------------------
    {
        "id": "CODE_01",
        "label": "242 - Code Injection",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"<\s*script\b|javascript\s*:|onerror\s*=|onload\s*=",
        "description": "Patrones de inyeccion de script/XSS reflejados en la peticion.",
    },
    {
        "id": "CODE_02",
        "label": "242 - Code Injection",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"<\?php|php://input|php://filter|allow_url_include|auto_prepend_file",
        "description": "Carga PHP o wrappers PHP usados para inyectar codigo.",
    },
    {
        "id": "CODE_03",
        "label": "242 - Code Injection",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"\b(eval|assert|base64_decode|create_function)\s*\(",
        "description": "Funciones asociadas a ejecucion u ofuscacion de codigo.",
    },

    # ------------------------------------------------------------------
    # CAPEC-88 OS Command Injection
    # ------------------------------------------------------------------
    {
        "id": "OSCI_01",
        "label": "88 - OS Command Injection",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"(;|\||&&|\$\(|`)\s*(whoami|id|uname|cat|ls|dir|wget|curl|chmod|nc|netcat|ping|bash|sh)\b",
        "description": "Encadenamiento de comandos del sistema operativo.",
    },
    {
        "id": "OSCI_02",
        "label": "88 - OS Command Injection",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"(/bin/sh|/bin/bash|cmd\.exe|powershell)",
        "description": "Invocacion directa de shells o interpretes del sistema.",
    },
    {
        "id": "OSCI_03",
        "label": "88 - OS Command Injection",
        "field": "request_body",
        "type": "regex_decoded",
        "pattern": r"\b(wget|curl|chmod|cat)\b",
        "description": "Comandos del sistema dentro del cuerpo de la peticion.",
    },

    # ------------------------------------------------------------------
    # CAPEC-16 Dictionary-based Password Attack
    # ------------------------------------------------------------------
    {
        "id": "DICT_01",
        "label": "16 - Dictionary-based Password Attack",
        "field": "request_http_request",
        "type": "regex_decoded",
        "pattern": r"(wp-login\.php|/login|/signin|/admin|/administrator)",
        "description": "Peticion dirigida a endpoint de autenticacion.",
    },
    {
        "id": "DICT_02",
        "label": "16 - Dictionary-based Password Attack",
        "field": "request_body",
        "type": "regex_decoded",
        "pattern": r"(log|user|username|pwd|pass|password)=",
        "description": "Envio de parametros de usuario y contrasena en la peticion.",
    },
    {
        "id": "DICT_03",
        "label": "16 - Dictionary-based Password Attack",
        "field": "request_body",
        "type": "regex_decoded",
        "pattern": r"(admin|root|test|password|123456)",
        "description": "Credenciales comunes en cuerpo de autenticacion.",
    },

    # ------------------------------------------------------------------
    # CAPEC-310 Scanning for Vulnerable Software
    # ------------------------------------------------------------------
    {
        "id": "SCAN_01",
        "label": "310 - Scanning for Vulnerable Software",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"\b(nmap|nikto|sqlmap|wpscan|nessus|openvas|acunetix|masscan|dirbuster|gobuster|ffuf|wfuzz|zaproxy)\b",
        "description": "Herramientas conocidas de escaneo o enumeracion.",
    },
    {
        "id": "SCAN_02",
        "label": "310 - Scanning for Vulnerable Software",
        "field": "request_http_request",
        "type": "regex_decoded",
        "pattern": r"(nmaplowercheck|/cfide/administrator|/readme\.html|/license\.txt|/cgi-bin/)",
        "description": "Rutas tipicas de enumeracion o comprobacion de software vulnerable.",
    },

    # ------------------------------------------------------------------
    # CAPEC-153 Input Data Manipulation
    # ------------------------------------------------------------------
    {
        "id": "IDM_01",
        "label": "153 - Input Data Manipulation",
        "field": "request_content_type",
        "type": "regex_decoded",
        "pattern": r"(\.\.[\\/]|/etc/passwd|%00|wp-config\.php)",
        "description": "Manipulacion de Content-Type con rutas o valores no esperados.",
    },
    {
        "id": "IDM_02",
        "label": "153 - Input Data Manipulation",
        "field": "all_raw",
        "type": "regex_raw",
        "pattern": r"(%00|%25(00|2e|2f|5c))",
        "description": "Null byte o doble encoding orientado a manipulacion de entradas.",
    },
    {
        "id": "IDM_03",
        "label": "153 - Input Data Manipulation",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"(\$\{[^}]+\}|\{\{[^}]+\}\}|0x[0-9a-f]{12,})",
        "description": "Patrones de manipulacion generica: template injection o valores hexadecimales largos.",
    },

    # ------------------------------------------------------------------
    # CAPEC-274 HTTP Verb Tampering
    # ------------------------------------------------------------------
    {
        "id": "VERB_01",
        "label": "274 - HTTP Verb Tampering",
        "field": "request_http_method",
        "type": "method_set",
        "pattern": "TRACE|TRACK|DEBUG|PROPFIND|MKCOL|COPY|MOVE|LOCK|UNLOCK|PATCH|CONNECT",
        "description": "Uso de metodos HTTP poco habituales o potencialmente peligrosos.",
    },
    {
        "id": "VERB_02",
        "label": "274 - HTTP Verb Tampering",
        "field": "request_http_method",
        "type": "method_uncommon",
        "pattern": "not in GET|POST|HEAD|OPTIONS",
        "description": "Metodo HTTP no comun observado en la peticion.",
    },

    # ------------------------------------------------------------------
    # CAPEC-194 Fake the Source of Data
    # ------------------------------------------------------------------
    {
        "id": "FAKE_01",
        "label": "194 - Fake the Source of Data",
        "field": "request_http_request",
        "type": "regex_decoded",
        "pattern": r"(https?://|www\.google\.com|file://)",
        "description": "URL externa incrustada dentro de la ruta solicitada.",
    },
    {
        "id": "FAKE_02",
        "label": "194 - Fake the Source of Data",
        "field": "request_origin",
        "type": "regex_decoded",
        "pattern": r"(null|localhost|127\.0\.0\.1|file://|javascript:)",
        "description": "Origin sospechoso o no esperable.",
    },
    {
        "id": "FAKE_03",
        "label": "194 - Fake the Source of Data",
        "field": "request_referer",
        "type": "regex_decoded",
        "pattern": r"(localhost|127\.0\.0\.1|file://|javascript:)",
        "description": "Referer sospechoso o falsificado.",
    },

    # ------------------------------------------------------------------
    # CAPEC-34 HTTP Response Splitting
    # ------------------------------------------------------------------
    {
        "id": "HRS_01",
        "label": "34 - HTTP Response Splitting",
        "field": "all_raw",
        "type": "regex_raw",
        "pattern": r"%0d%0a|%0a%0d|%0d|%0a",
        "description": "Secuencias CR/LF codificadas en la peticion.",
    },
    {
        "id": "HRS_02",
        "label": "34 - HTTP Response Splitting",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"(set-cookie:|location:|content-length:|content-type:)",
        "description": "Cabeceras inyectadas tras decodificacion.",
    },

    # ------------------------------------------------------------------
    # CAPEC-33 HTTP Request Smuggling
    # ------------------------------------------------------------------
    {
        "id": "HSMUG_01",
        "label": "33 - HTTP Request Smuggling",
        "field": "request_body",
        "type": "regex_raw",
        "pattern": r"%0d%0a|%0a|%0d",
        "description": "CR/LF codificado dentro del cuerpo de una peticion POST.",
    },
    {
        "id": "HSMUG_02",
        "label": "33 - HTTP Request Smuggling",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"transfer-encoding\s*:\s*chunked|content-length\s*:",
        "description": "Cabeceras asociadas a desincronizacion HTTP.",
    },

    # ------------------------------------------------------------------
    # CAPEC-272 Protocol Manipulation
    # ------------------------------------------------------------------
    {
        "id": "PROTO_01",
        "label": "272 - Protocol Manipulation",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"(\.svn/wc\.db|\.svn/entries|\.git/head|\.git/config)",
        "description": "Acceso a metadatos internos que alteran el uso esperado del protocolo/aplicacion.",
    },
    {
        "id": "PROTO_02",
        "label": "272 - Protocol Manipulation",
        "field": "all",
        "type": "regex_decoded",
        "pattern": r"(b4ckdoor|backdoor|127\.0\.0\.1|remote_syslog|remotesyslogsupported)",
        "description": "Indicadores de manipulacion de protocolo o configuracion remota.",
    },
    {
        "id": "PROTO_03",
        "label": "272 - Protocol Manipulation",
        "field": "request_http_protocol",
        "type": "regex_decoded",
        "pattern": r"http/(0\.9|1\.2|2\.0)",
        "description": "Versiones HTTP no esperadas en el contexto del dataset.",
    },
]


def safe_col(label):
    return PRED_PREFIX + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


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


def row_text(row, decoded=False):
    parts = []
    for field in REQUEST_FIELDS:
        if field in row.index:
            value = clean_value(row.get(field, ""))
            if value.strip():
                parts.append(value)
    text = " ".join(parts)
    return decode_repeated(text) if decoded else text


def field_text(row, field, decoded=False):
    value = clean_value(row.get(field, ""))
    return decode_repeated(value) if decoded else value


def is_rule_fired(rule, row):
    field = rule["field"]
    pattern = rule["pattern"]
    rule_type = rule["type"]

    if field == "all":
        target_raw = row_text(row, decoded=False)
        target_decoded = row_text(row, decoded=True)
    elif field == "all_raw":
        target_raw = row_text(row, decoded=False)
        target_decoded = target_raw
    else:
        target_raw = field_text(row, field, decoded=False)
        target_decoded = field_text(row, field, decoded=True)

    if rule_type == "keyword_decoded":
        return pattern.lower() in target_decoded.lower()

    if rule_type == "keyword_raw":
        return pattern.lower() in target_raw.lower()

    if rule_type == "regex_decoded":
        return re.search(pattern, target_decoded, flags=re.IGNORECASE | re.DOTALL) is not None

    if rule_type == "regex_raw":
        return re.search(pattern, target_raw, flags=re.IGNORECASE | re.DOTALL) is not None

    if rule_type == "method_set":
        method = field_text(row, "request_http_method", decoded=True).strip().upper()
        allowed = {m.strip().upper() for m in pattern.split("|")}
        return method in allowed

    if rule_type == "method_uncommon":
        method = field_text(row, "request_http_method", decoded=True).strip().upper()
        common = {"GET", "POST", "HEAD", "OPTIONS"}
        return bool(method) and method not in common

    return False


def predict_row(row):
    fired_by_label = {label: [] for label in ATTACK_LABELS}

    for rule in RULES:
        label = rule["label"]
        if label not in fired_by_label:
            continue
        if is_rule_fired(rule, row):
            fired_by_label[label].append(rule["id"])

    predictions = {label: 1 if fired_by_label[label] else 0 for label in ATTACK_LABELS}

    # Restricciones lógicas para reducir falsos positivos en reglas agresivas.
    # Dictionary attack: endpoint de login + parametros/credenciales.
    if predictions["16 - Dictionary-based Password Attack"]:
        dict_hits = set(fired_by_label["16 - Dictionary-based Password Attack"])
        predictions["16 - Dictionary-based Password Attack"] = 1 if (
            "DICT_01" in dict_hits and ("DICT_02" in dict_hits or "DICT_03" in dict_hits)
        ) else 0
        if not predictions["16 - Dictionary-based Password Attack"]:
            fired_by_label["16 - Dictionary-based Password Attack"] = []

    # Request smuggling: CRLF en body debe ir en una peticion con metodo POST.
    if predictions["33 - HTTP Request Smuggling"]:
        method = field_text(row, "request_http_method", decoded=True).strip().upper()
        predictions["33 - HTTP Request Smuggling"] = 1 if method == "POST" else 0
        if not predictions["33 - HTTP Request Smuggling"]:
            fired_by_label["33 - HTTP Request Smuggling"] = []

    predictions[NORMAL_LABEL] = 0 if any(predictions.values()) else 1

    fired_rule_ids = []
    for label in ATTACK_LABELS:
        if predictions[label]:
            fired_rule_ids.extend(fired_by_label[label])

    return predictions, fired_rule_ids


def compute_quick_metrics(df, pred_cols):
    y_true = df[ALL_LABELS].astype(int).values
    y_pred = df[[pred_cols[label] for label in ALL_LABELS]].astype(int).values

    overall = {
        "rows": len(df),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "jaccard_samples": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
        "subset_accuracy": accuracy_score(y_true, y_pred),
    }

    per_label = []
    for i, label in enumerate(ALL_LABELS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
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


def main(input_file):
    input_path = Path(input_file)
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Cargando {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Filas a procesar: {len(df)}")

    missing_labels = [label for label in ALL_LABELS if label not in df.columns]
    if missing_labels:
        raise ValueError(f"Faltan columnas de etiquetas en el CSV: {missing_labels}")

    if "248 - Command Injection" in df.columns:
        print("Aviso: CAPEC-248 aparece en el archivo, pero no se evalua en esta tecnica.")

    pred_records = []
    rule_hit_counts = {rule["id"]: 0 for rule in RULES}
    label_hit_counts = {label: 0 for label in ALL_LABELS}

    print("Aplicando reglas...")
    for _, row in df.iterrows():
        preds, fired_rule_ids = predict_row(row)

        for rule_id in fired_rule_ids:
            rule_hit_counts[rule_id] += 1

        for label, value in preds.items():
            if value == 1:
                label_hit_counts[label] += 1

        pred_record = {safe_col(label): preds[label] for label in ALL_LABELS}
        pred_record["rules_fired"] = ";".join(fired_rule_ids)
        pred_records.append(pred_record)

    pred_df = pd.DataFrame(pred_records)
    out_df = pd.concat([df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

    predictions_path = OUT_DIR / "predictions_rules.csv"
    out_df.to_csv(predictions_path, index=False)

    pred_cols = {label: safe_col(label) for label in ALL_LABELS}
    overall, per_label = compute_quick_metrics(out_df, pred_cols)

    predicted_distribution_path = OUT_DIR / "rules_predicted_distribution.csv"
    with predicted_distribution_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "predicted_count", "percentage_eval_set"])
        for label, count in sorted(label_hit_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(df) * 100) if len(df) else 0
            writer.writerow([label, int(count), round(pct, 6)])

    dictionary_path = OUT_DIR / "rules_dictionary.csv"
    with dictionary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rule_id", "label", "field", "type", "pattern", "description"])
        for rule in RULES:
            writer.writerow([
                rule["id"],
                rule["label"],
                rule["field"],
                rule["type"],
                rule["pattern"],
                rule["description"],
            ])

    overall_path = OUT_DIR / "rules_quick_overall_metrics.csv"
    with overall_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in overall.items():
            writer.writerow([key, round(value, 6) if isinstance(value, float) else value])

    per_label_path = OUT_DIR / "rules_quick_per_label_metrics.csv"
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

    summary_path = OUT_DIR / "rules_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("RESUMEN DE ETIQUETADO POR REGLAS Y HEURISTICAS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Archivo de entrada: {input_path}\n")
        f.write(f"Filas procesadas: {len(df)}\n")
        f.write("Estrategia: agresiva moderada, orientada a priorizar recall sin reglas excesivamente generales.\n")
        f.write("Formato: predicciones multietiqueta independientes por CAPEC.\n")
        f.write("Normal se predice solo si no se activa ninguna regla de ataque.\n\n")

        f.write("--- Distribucion de etiquetas predichas ---\n")
        for label, count in sorted(label_hit_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  - {label}: {count}\n")

        f.write("\n--- Activacion por regla ---\n")
        for rule in RULES:
            f.write(f"  - {rule['id']} ({rule['label']}): {rule_hit_counts[rule['id']]}\n")

        f.write("\n--- Metricas rapidas sobre eval_set ---\n")
        for key, value in overall.items():
            if isinstance(value, float):
                f.write(f"  - {key}: {value:.6f}\n")
            else:
                f.write(f"  - {key}: {value}\n")

        f.write("\n--- Nota metodologica ---\n")
        f.write(
            "Esta tecnica se usa como baseline interpretable. Las reglas son explicitas "
            "y trazables. Se adopta una estrategia agresiva moderada: no se busca "
            "maximizar un clasificador final, sino disponer de un punto de comparacion "
            "reproducible frente a las demas tecnicas. La evaluacion final se hara "
            "de forma comun junto con el resto de tecnicas mediante 07_evaluation.py.\n"
        )

    print()
    print("Etiquetado por reglas completado.")
    print(f"Predicciones: {predictions_path}")
    print(f"Resumen: {summary_path}")
    print(f"Micro-F1 rapido: {overall['micro_f1']:.4f}")
    print(f"Macro-F1 rapido: {overall['macro_f1']:.4f}")
    print(f"Hamming loss rapido: {overall['hamming_loss']:.4f}")
    print(f"Jaccard rapido: {overall['jaccard_samples']:.4f}")
    print(f"Subset accuracy rapido: {overall['subset_accuracy']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python 03_rules_labeling_srbh2020.py sampling_srbh2020/eval_set.csv")
        sys.exit(1)
    main(sys.argv[1])
