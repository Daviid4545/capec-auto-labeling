"""
Análisis exploratorio inicial del dataset SR-BH 2020.

Uso:
    python 01_eda_srbh2020.py data_capec_multilabel.csv

El script no modifica el dataset original.
Genera:
    - label_distribution.csv
    - feature_completeness.csv
    - eda_summary.txt
    - label_distribution.png
"""

from pathlib import Path
import csv
import collections
import sys
import matplotlib.pyplot as plt


def main(input_file):
    input_path = Path(input_file)
    out_dir = Path("eda_srbh2020")
    out_dir.mkdir(exist_ok=True)

    with input_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)

    features = header[:24]
    labels = header[24:]

    total_rows = 0
    label_counts = {label: 0 for label in labels}
    feature_non_empty = {feature: 0 for feature in features}
    rows_without_features = 0
    total_label_count_dist = collections.Counter()
    attack_label_count_dist = collections.Counter()
    normal_plus_attack = 0

    with input_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1

            has_any_feature = False
            for feature in features:
                value = row.get(feature, "")
                if value is not None and str(value).strip() != "":
                    feature_non_empty[feature] += 1
                    has_any_feature = True

            if not has_any_feature:
                rows_without_features += 1

            total_label_sum = 0
            attack_label_sum = 0

            for label in labels:
                value = row.get(label, "").strip()
                is_active = value == "1" or value == "1.0"
                if is_active:
                    label_counts[label] += 1
                    total_label_sum += 1
                    if not label.startswith("000"):
                        attack_label_sum += 1

            if row.get("000 - Normal", "").strip() in {"1", "1.0"} and attack_label_sum > 0:
                normal_plus_attack += 1

            total_label_count_dist[total_label_sum] += 1
            attack_label_count_dist[attack_label_sum] += 1

    with (out_dir / "label_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "count", "percentage_total"])
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([label, count, round((count / total_rows) * 100, 6)])

    with (out_dir / "feature_completeness.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "non_empty_count", "non_empty_percentage"])
        for feature in features:
            count = feature_non_empty[feature]
            writer.writerow([feature, count, round((count / total_rows) * 100, 6)])

    with (out_dir / "eda_summary.txt").open("w", encoding="utf-8") as f:
        f.write("ANÁLISIS EXPLORATORIO INICIAL - SR-BH 2020\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Filas totales del CSV: {total_rows}\n")
        f.write(f"Columnas totales: {len(header)}\n")
        f.write(f"Características: {len(features)}\n")
        f.write(f"Columnas de etiquetas: {len(labels)}\n")
        f.write(f"Filas sin ninguna característica HTTP rellena: {rows_without_features}\n")
        f.write(f"Filas con Normal y alguna etiqueta de ataque simultánea: {normal_plus_attack}\n\n")

        f.write("Distribución por número total de etiquetas activas:\n")
        for k in sorted(total_label_count_dist):
            f.write(f"  {k} etiqueta(s): {total_label_count_dist[k]}\n")

        f.write("\nDistribución por número de etiquetas de ataque activas:\n")
        for k in sorted(attack_label_count_dist):
            f.write(f"  {k} etiqueta(s) de ataque: {attack_label_count_dist[k]}\n")

        f.write("\nDistribución de etiquetas:\n")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {label}: {count} ({(count / total_rows) * 100:.4f}%)\n")

    labels_sorted = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    plot_labels = [x[0] for x in labels_sorted]
    plot_counts = [x[1] for x in labels_sorted]

    plt.figure(figsize=(12, 7))
    plt.barh(plot_labels[::-1], plot_counts[::-1])
    plt.xlabel("Número de instancias")
    plt.ylabel("Etiqueta")
    plt.title("Distribución de etiquetas en SR-BH 2020")
    plt.tight_layout()
    plt.savefig(out_dir / "label_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Análisis terminado. Resultados guardados en:", out_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python 01_eda_srbh2020.py data_capec_multilabel.csv")
        sys.exit(1)
    main(sys.argv[1])
