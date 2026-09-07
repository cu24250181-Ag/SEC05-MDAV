from argparse import ArgumentParser
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


DEFAULT_WHOLESALE_ARCHIVE = Path.home() / "Downloads" / "archive (2).zip"


def read_csv_from_zip(archive_path):
    with ZipFile(archive_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError(f"No CSV file found in {archive_path}")
        with archive.open(csv_names[0]) as csv_file:
            return pd.read_csv(csv_file)


def preprocess(data, features):
    cleaned = data.drop_duplicates().copy()
    cleaned[features] = cleaned[features].apply(pd.to_numeric, errors="coerce")
    return cleaned.dropna(subset=features)


def select_k(values):
    upper_k = min(10, len(values) - 1)
    k_values = list(range(2, upper_k + 1))
    inertias = []
    silhouettes = []
    for cluster_count in k_values:
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        labels = model.fit_predict(values)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(values, labels))
    selected_k = k_values[int(np.argmax(silhouettes))]
    return k_values, inertias, silhouettes, selected_k


def cluster_dataset(data, features):
    cleaned = preprocess(data, features)
    scaled = StandardScaler().fit_transform(cleaned[features])
    k_values, inertias, silhouettes, selected_k = select_k(scaled)
    model = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    cleaned["Cluster"] = model.fit_predict(scaled)
    score = silhouette_score(scaled, cleaned["Cluster"])
    return cleaned, scaled, model, k_values, inertias, silhouettes, score


def plot_elbow(k_values, inertias, silhouettes, title):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(k_values, inertias, marker="o")
    axes[0].set_title(f"{title}: Elbow Method")
    axes[0].set_xlabel("Number of clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(k_values, silhouettes, marker="o", color="darkorange")
    axes[1].set_title(f"{title}: Silhouette Validation")
    axes[1].set_xlabel("Number of clusters (K)")
    axes[1].set_ylabel("Silhouette score")
    figure.tight_layout()
    plt.show()


def plot_clusters(scaled, labels, centroids, title, x_label, y_label):
    plt.figure(figsize=(8, 6))
    plt.scatter(scaled[:, 0], scaled[:, 1], c=labels, s=45, alpha=0.75)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=220, color="black")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def describe_segments(data, features, dataset_name):
    profile = data.groupby("Cluster")[features].mean().round(2)
    print(f"\n{dataset_name} segment profiles")
    print(profile)
    print("\nSuggested marketing strategies")
    for cluster, row in profile.iterrows():
        highest = row.idxmax()
        lowest = row.idxmin()
        print(
            f"Cluster {cluster}: emphasize {highest}; improve engagement around {lowest} "
            "with targeted offers and recommendations."
        )


def run_dataset(data, features, name):
    clustered, scaled, model, k_values, inertias, silhouettes, score = cluster_dataset(
        data, features
    )
    print(f"\n{name}")
    print(f"Rows after cleaning: {len(clustered)}")
    print(f"Selected K: {model.n_clusters}")
    print(f"Silhouette score: {score:.3f}")
    describe_segments(clustered, features, name)
    plot_elbow(k_values, inertias, silhouettes, name)
    plot_clusters(
        scaled,
        clustered["Cluster"],
        model.cluster_centers_,
        f"{name} Customer Segmentation",
        f"{features[0]} (standardized)",
        f"{features[1]} (standardized)",
    )
    return clustered


def main():
    parser = ArgumentParser(
        description="Experiment 7: Wholesale Customer Segmentation using K-Means"
    )
    parser.add_argument("--wholesale-archive", type=Path, default=DEFAULT_WHOLESALE_ARCHIVE)
    args = parser.parse_args()

    wholesale = read_csv_from_zip(args.wholesale_archive)
    run_dataset(
        wholesale,
        ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"],
        "Wholesale Customers",
    )


if __name__ == "__main__":
    main()
