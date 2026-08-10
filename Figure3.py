#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Aug 10 12:42:25 2026

@author: jaanajurvansuu

Temporal distance-decay analysis of Homo-associated viral communities.

The script:
1. filters Homo-associated viral records;
2. aggregates normalised RPKMF at species level per sample;
3. extracts sampling month, city and location from sample_ID;
4. defines dominant/prevalent species separately within each city × location
   as species cumulatively accounting for 90% of mean normalised RPKMF;
5. calculates temporal distance-decay for:
   - the complete community;
   - dominant/prevalent species;
   - remaining species;
6. calculates Bray–Curtis and Jaccard dissimilarities;
7. fits descriptive linear regressions;
8. performs two-sided Pearson Mantel permutation tests;
9. applies Holm correction within each analysis family;
10. exports pairwise data, species classifications and statistical tables;
11. saves PNG figures only.

Pairwise observations are not independent. Linear regressions are therefore
descriptive, whereas Mantel tests provide the inferential results.
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import braycurtis, jaccard
from scipy.stats import linregress, pearsonr
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent
input_file = ROOT / "EsVirutu_abundance_metadata.tsv"
output_dir = ROOT

output_prefix = "Homo_species_temporal_distance_decay"

output_classification_tsv = (
    output_dir / "Figure3_species_classification.tsv"
)
output_pairwise_tsv = (
    output_dir / "Figure3_temporal_pairwise_dissimilarities.tsv"
)
output_statistics_tsv = (
    output_dir / "Figure3_temporal_distance_decay_statistics.tsv"
)

output_complete_png = {
    "bray_curtis": (
        output_dir / "Figure3_A-D_temporal_Bray_Curtis.png"
    ),
    "jaccard": (
        output_dir / "FigureS2_complete_Jaccard.png"
    ),
}

output_partitioned_png = {
    "bray_curtis": (
        output_dir
        / "FigureS2_dominant_remaining_Bray_Curtis.png"
    ),
    "jaccard": (
        output_dir
        / "FigureS2_dominant_remaining_Jaccard.png"
    ),
}

cities = ["TRE", "KUO"]
locations = ["UH", "WWTP", "SA1", "SA2"]

groups = [
    f"{city}_{location}"
    for city in cities
    for location in locations
]

dominant_abundance_fraction = 0.90
presence_threshold = 0.0

n_permutations = 9999
random_seed = 42
alpha_threshold = 0.05

distance_metrics = {
    "bray_curtis": "Bray–Curtis dissimilarity",
    "jaccard": "Jaccard dissimilarity",
}

palette = {
    "UH": "#E67E22",
    "WWTP": "#2E86C1",
    "SA1": "#52BE80",
    "SA2": "#239B56",
}

city_markers = {
    "TRE": "o",
    "KUO": "s",
}

city_linestyles = {
    "TRE": "-",
    "KUO": "--",
}

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 13
mpl.rcParams["ytick.labelsize"] = 13
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["axes.linewidth"] = 1.2
mpl.rcParams["xtick.major.width"] = 1.2
mpl.rcParams["ytick.major.width"] = 1.2
mpl.rcParams["xtick.major.size"] = 5
mpl.rcParams["ytick.major.size"] = 5

sns.set_theme(style="ticks", context="notebook")

def validate_columns(data: pd.DataFrame, required_columns: set[str]) -> None:
    """Check that all required columns are present."""
    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

def load_species_abundance(file_path: str | Path) -> pd.DataFrame:
    """Read and aggregate Homo-associated normalised RPKMF."""
    data = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    validate_columns(
        data,
        {"sample_ID", "host_genus", "species", "RPKMF_norm"},
    )

    data["RPKMF_norm"] = pd.to_numeric(
        data["RPKMF_norm"],
        errors="coerce",
    )

    invalid_count = data["RPKMF_norm"].isna().sum()

    if invalid_count > 0:
        warnings.warn(
            f"{invalid_count} non-numeric RPKMF_norm values "
            "were converted to zero."
        )

    data["RPKMF_norm"] = data["RPKMF_norm"].fillna(0.0)

    if (data["RPKMF_norm"] < 0).any():
        raise ValueError(
            "Negative normalised RPKMF values were detected."
        )

    data = data[
        data["host_genus"]
        .str.strip()
        .str.casefold()
        .eq("homo")
    ].copy()

    data["species"] = data["species"].str.strip()
    data = data[data["species"].ne("")].copy()

    sample_parts = data["sample_ID"].str.split("_", expand=True)

    if sample_parts.shape[1] < 4:
        raise ValueError(
            "sample_ID must contain at least year_month_city_location."
        )

    data["year"] = pd.to_numeric(
        sample_parts[0],
        errors="raise",
    ).astype(int)

    data["month"] = pd.to_numeric(
        sample_parts[1],
        errors="raise",
    ).astype(int)

    if not data["month"].between(1, 12).all():
        raise ValueError(
            "An invalid sampling month was detected in sample_ID."
        )

    data["city"] = sample_parts[2].str.strip()
    data["location"] = sample_parts[3].str.strip()
    data["group"] = data["city"] + "_" + data["location"]

    data = data[data["group"].isin(groups)].copy()

    if data.empty:
        raise ValueError(
            "No observations matched the expected city-location groups."
        )

    data["month_index"] = data["year"] * 12 + data["month"]
    data["year_month"] = (
        data["year"].astype(str)
        + "-"
        + data["month"].astype(str).str.zfill(2)
    )

    return (
        data.groupby(
            [
                "sample_ID",
                "city",
                "location",
                "group",
                "month_index",
                "year_month",
                "species",
            ],
            as_index=False,
            observed=True,
        )
        .agg(RPKMF_norm=("RPKMF_norm", "sum"))
    )

def build_group_matrix(
    sample_species: pd.DataFrame,
    group_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build metadata and a sample-by-species abundance matrix."""
    subset = sample_species[
        sample_species["group"] == group_name
    ].copy()

    abundance_matrix = subset.pivot_table(
        index="sample_ID",
        columns="species",
        values="RPKMF_norm",
        aggfunc="sum",
        fill_value=0.0,
    )

    if abundance_matrix.empty:
        return pd.DataFrame(), pd.DataFrame()

    metadata = (
        subset[
            [
                "sample_ID",
                "city",
                "location",
                "group",
                "month_index",
                "year_month",
            ]
        ]
        .drop_duplicates(subset="sample_ID")
        .set_index("sample_ID")
        .reindex(abundance_matrix.index)
        .sort_values(["month_index", "year_month"])
    )

    abundance_matrix = abundance_matrix.loc[metadata.index].copy()
    abundance_matrix = abundance_matrix.reindex(
        sorted(abundance_matrix.columns),
        axis=1,
    )

    return metadata, abundance_matrix

def classify_species(
    abundance_matrix: pd.DataFrame,
    cumulative_fraction: float,
) -> pd.DataFrame:
    """Classify dominant/prevalent species by cumulative mean abundance."""
    mean_abundance = (
        abundance_matrix.mean(axis=0).sort_values(ascending=False)
    )

    total_mean_abundance = mean_abundance.sum()

    if total_mean_abundance <= 0:
        return pd.DataFrame(
            {
                "species": mean_abundance.index,
                "abundance_rank": np.arange(1, len(mean_abundance) + 1),
                "mean_normalised_RPKMF": mean_abundance.values,
                "fraction_of_total_mean_abundance": np.nan,
                "cumulative_abundance_fraction": np.nan,
                "dominant_prevalent": False,
            }
        )

    abundance_fraction = mean_abundance / total_mean_abundance
    cumulative_fraction_values = abundance_fraction.cumsum()

    selected_count = int(
        np.searchsorted(
            cumulative_fraction_values.to_numpy(),
            cumulative_fraction,
            side="left",
        )
        + 1
    )

    selected_species = set(mean_abundance.index[:selected_count])

    classification = pd.DataFrame(
        {
            "species": mean_abundance.index,
            "abundance_rank": np.arange(1, len(mean_abundance) + 1),
            "mean_normalised_RPKMF": mean_abundance.values,
            "fraction_of_total_mean_abundance": abundance_fraction.values,
            "cumulative_abundance_fraction": (
                cumulative_fraction_values.values
            ),
        }
    )

    classification["dominant_prevalent"] = (
        classification["species"].isin(selected_species)
    )

    return classification

def temporal_distance_matrix(month_index: np.ndarray) -> np.ndarray:
    """Calculate absolute temporal separation in months."""
    return np.abs(
        month_index[:, None] - month_index[None, :]
    ).astype(float)

def bray_curtis_value(
    vector_1: np.ndarray,
    vector_2: np.ndarray,
) -> float:
    """Calculate Bray–Curtis dissimilarity safely."""
    if np.sum(vector_1) == 0 and np.sum(vector_2) == 0:
        return 0.0

    return float(braycurtis(vector_1, vector_2))

def jaccard_value(
    vector_1: np.ndarray,
    vector_2: np.ndarray,
    threshold: float,
) -> float:
    """Calculate Jaccard dissimilarity safely."""
    presence_1 = vector_1 > threshold
    presence_2 = vector_2 > threshold

    if not np.any(presence_1 | presence_2):
        return 0.0

    return float(jaccard(presence_1, presence_2))

def community_distance_matrix(
    abundance: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Construct a symmetric community dissimilarity matrix."""
    sample_count = abundance.shape[0]
    matrix = np.zeros((sample_count, sample_count), dtype=float)

    for index_1 in range(sample_count - 1):
        for index_2 in range(index_1 + 1, sample_count):
            if metric == "bray_curtis":
                value = bray_curtis_value(
                    abundance[index_1],
                    abundance[index_2],
                )
            elif metric == "jaccard":
                value = jaccard_value(
                    abundance[index_1],
                    abundance[index_2],
                    threshold=presence_threshold,
                )
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            matrix[index_1, index_2] = value
            matrix[index_2, index_1] = value

    return matrix

def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    """Extract the upper triangle excluding the diagonal."""
    triangle = np.triu_indices_from(matrix, k=1)
    return matrix[triangle]

def mantel_test(
    community_matrix: np.ndarray,
    time_matrix: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Perform a two-sided Pearson Mantel permutation test."""
    if community_matrix.shape != time_matrix.shape:
        raise ValueError(
            "Mantel matrices must have identical dimensions."
        )

    community_values = upper_triangle_values(community_matrix)
    time_values = upper_triangle_values(time_matrix)

    valid = np.isfinite(community_values) & np.isfinite(time_values)
    community_values = community_values[valid]
    time_values = time_values[valid]

    if (
        len(community_values) < 3
        or np.std(community_values) == 0
        or np.std(time_values) == 0
    ):
        return np.nan, np.nan

    observed_r = float(
        pearsonr(time_values, community_values).statistic
    )

    sample_count = community_matrix.shape[0]
    permuted_r = np.empty(permutations, dtype=float)

    for permutation_index in range(permutations):
        permutation = rng.permutation(sample_count)
        permuted_matrix = community_matrix[
            np.ix_(permutation, permutation)
        ]
        permuted_values = upper_triangle_values(
            permuted_matrix
        )[valid]

        permuted_r[permutation_index] = pearsonr(
            time_values,
            permuted_values,
        ).statistic

    permuted_r = permuted_r[np.isfinite(permuted_r)]

    if len(permuted_r) == 0:
        return observed_r, np.nan

    p_value = (
        np.sum(np.abs(permuted_r) >= abs(observed_r)) + 1
    ) / (len(permuted_r) + 1)

    return observed_r, float(p_value)

def regression_statistics(
    time_matrix: np.ndarray,
    community_matrix: np.ndarray,
) -> dict[str, float]:
    """Calculate a descriptive linear regression."""
    x_values = upper_triangle_values(time_matrix)
    y_values = upper_triangle_values(community_matrix)

    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]

    if len(x_values) < 3 or np.unique(x_values).size < 2:
        return {
            "intercept": np.nan,
            "slope_per_month": np.nan,
            "r_value": np.nan,
            "r_squared": np.nan,
            "regression_p_value": np.nan,
            "slope_standard_error": np.nan,
        }

    result = linregress(x_values, y_values)

    return {
        "intercept": float(result.intercept),
        "slope_per_month": float(result.slope),
        "r_value": float(result.rvalue),
        "r_squared": float(result.rvalue**2),
        "regression_p_value": float(result.pvalue),
        "slope_standard_error": float(result.stderr),
    }

def apply_mantel_correction(
    statistics: pd.DataFrame,
) -> pd.DataFrame:
    """Apply Holm correction within subset × metric families."""
    statistics = statistics.copy()
    statistics["mantel_p_Holm"] = np.nan
    statistics["mantel_significant_Holm"] = False

    for _, row_indices in statistics.groupby(
        ["community_subset", "metric"],
        dropna=False,
    ).groups.items():
        row_indices = list(row_indices)

        valid_indices = [
            index
            for index in row_indices
            if pd.notna(
                statistics.loc[index, "mantel_p_two_sided"]
            )
        ]

        if not valid_indices:
            continue

        rejected, adjusted_p, _, _ = multipletests(
            statistics.loc[
                valid_indices,
                "mantel_p_two_sided",
            ],
            alpha=alpha_threshold,
            method="holm",
        )

        statistics.loc[
            valid_indices,
            "mantel_p_Holm",
        ] = adjusted_p

        statistics.loc[
            valid_indices,
            "mantel_significant_Holm",
        ] = rejected

    return statistics

def calculate_all_results(
    sample_species: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate classifications, pairwise values and statistics."""
    rng = np.random.default_rng(random_seed)

    classification_tables = []
    pairwise_rows = []
    statistics_rows = []

    for group_name in groups:
        metadata, abundance_matrix = build_group_matrix(
            sample_species,
            group_name,
        )

        if len(metadata) < 2:
            warnings.warn(
                f"Skipping {group_name}: fewer than two samples."
            )
            continue

        classification = classify_species(
            abundance_matrix,
            cumulative_fraction=dominant_abundance_fraction,
        )

        city, location = group_name.split("_", 1)

        classification["group"] = group_name
        classification["city"] = city
        classification["location"] = location
        classification["available_samples"] = (
            abundance_matrix.shape[0]
        )
        classification_tables.append(classification)

        dominant_species = classification.loc[
            classification["dominant_prevalent"],
            "species",
        ].tolist()

        remaining_species = classification.loc[
            ~classification["dominant_prevalent"],
            "species",
        ].tolist()

        subset_columns = {
            "Complete": abundance_matrix.columns.tolist(),
            "Dominant/prevalent": dominant_species,
            "Remaining": remaining_species,
        }

        month_index = metadata["month_index"].to_numpy(dtype=int)
        time_matrix = temporal_distance_matrix(month_index)
        sample_ids = metadata.index.tolist()

        for community_subset, columns in subset_columns.items():
            if len(columns) == 0:
                warnings.warn(
                    f"Skipping {group_name}, {community_subset}: "
                    "no species were available."
                )
                continue

            abundance = abundance_matrix.loc[:, columns].to_numpy(
                dtype=float
            )

            for metric in distance_metrics:
                community_matrix = community_distance_matrix(
                    abundance=abundance,
                    metric=metric,
                )

                regression = regression_statistics(
                    time_matrix=time_matrix,
                    community_matrix=community_matrix,
                )

                mantel_r, mantel_p = mantel_test(
                    community_matrix=community_matrix,
                    time_matrix=time_matrix,
                    permutations=n_permutations,
                    rng=rng,
                )

                time_values = upper_triangle_values(time_matrix)

                statistics_rows.append(
                    {
                        "group": group_name,
                        "city": city,
                        "location": location,
                        "community_subset": community_subset,
                        "metric": metric,
                        "n_samples": len(metadata),
                        "n_species": len(columns),
                        "n_pairs": (
                            len(metadata) * (len(metadata) - 1) // 2
                        ),
                        "n_unique_time_lags": len(
                            np.unique(time_values)
                        ),
                        "minimum_time_lag_months": float(
                            np.min(time_values)
                        ),
                        "maximum_time_lag_months": float(
                            np.max(time_values)
                        ),
                        **regression,
                        "mantel_r": mantel_r,
                        "mantel_p_two_sided": mantel_p,
                        "mantel_permutations": n_permutations,
                    }
                )

                for index_1, index_2 in itertools.combinations(
                    range(len(metadata)),
                    2,
                ):
                    pairwise_rows.append(
                        {
                            "group": group_name,
                            "city": city,
                            "location": location,
                            "community_subset": community_subset,
                            "metric": metric,
                            "sample_1": sample_ids[index_1],
                            "sample_2": sample_ids[index_2],
                            "year_month_1": metadata.iloc[index_1][
                                "year_month"
                            ],
                            "year_month_2": metadata.iloc[index_2][
                                "year_month"
                            ],
                            "months_apart": float(
                                time_matrix[index_1, index_2]
                            ),
                            "dissimilarity": float(
                                community_matrix[index_1, index_2]
                            ),
                            "n_species": len(columns),
                        }
                    )

    if not classification_tables:
        raise ValueError(
            "No species-classification results were generated."
        )

    if not pairwise_rows or not statistics_rows:
        raise ValueError(
            "No temporal distance-decay results were generated."
        )

    classifications = pd.concat(
        classification_tables,
        ignore_index=True,
    )
    pairwise = pd.DataFrame(pairwise_rows)
    statistics = apply_mantel_correction(
        pd.DataFrame(statistics_rows)
    )

    return classifications, pairwise, statistics

def city_legend_handles() -> list[Line2D]:
    """Create city marker and line-style legend handles."""
    return [
        Line2D(
            [0],
            [0],
            marker=city_markers[city],
            linestyle=city_linestyles[city],
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            linewidth=1.5,
            markersize=7,
            label=city,
        )
        for city in cities
    ]

def add_city_to_axis(
    ax: plt.Axes,
    pairwise: pd.DataFrame,
    statistics: pd.DataFrame,
    city: str,
    location: str,
    metric: str,
    community_subset: str,
) -> dict[str, float | str] | None:
    """
    Add one city-specific point cloud and regression line.

    Return the city, slope and Pearson R value for the
    lower-right panel annotation.
    """
    subset = pairwise[
        (pairwise["city"] == city)
        & (pairwise["location"] == location)
        & (pairwise["metric"] == metric)
        & (pairwise["community_subset"] == community_subset)
    ].copy()

    statistic = statistics[
        (statistics["city"] == city)
        & (statistics["location"] == location)
        & (statistics["metric"] == metric)
        & (statistics["community_subset"] == community_subset)
    ]

    if subset.empty or statistic.empty:
        return None

    ax.scatter(
        subset["months_apart"],
        subset["dissimilarity"],
        s=25,
        color=palette[location],
        marker=city_markers[city],
        alpha=0.38,
        edgecolor="black",
        linewidth=0.25,
    )

    intercept = float(statistic.iloc[0]["intercept"])
    slope = float(statistic.iloc[0]["slope_per_month"])
    r_value = float(statistic.iloc[0]["r_value"])

    if np.isfinite(intercept) and np.isfinite(slope):
        x_values = np.linspace(
            subset["months_apart"].min(),
            subset["months_apart"].max(),
            200,
        )

        ax.plot(
            x_values,
            intercept + slope * x_values,
            color=palette[location],
            linestyle=city_linestyles[city],
            linewidth=2,
        )

    return {
        "city": city,
        "slope": slope,
        "r_value": r_value,
    }

def format_temporal_axis(
    ax: plt.Axes,
    location: str,
    show_location_label: bool,
) -> None:
    """Apply consistent panel formatting."""
    if show_location_label:
        ax.set_title(location)

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="both", direction="out")
    ax.grid(False)
    sns.despine(ax=ax)

def plot_complete_community(
    pairwise: pd.DataFrame,
    statistics: pd.DataFrame,
    metric: str,
    output_file: Path,
) -> None:
    """Plot complete-community temporal distance-decay."""
    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(16, 4.8),
        sharex=True,
        sharey=True,
    )

    for ax, location in zip(axes, locations):
        annotation_rows = []

        for city in cities:
            result = add_city_to_axis(
                ax=ax,
                pairwise=pairwise,
                statistics=statistics,
                city=city,
                location=location,
                metric=metric,
                community_subset="Complete",
            )

            if result is not None:
                annotation_rows.append(
                    f"{result['city']}: "
                    f"slope = {result['slope']:.4f}, "
                    f"R = {result['r_value']:.2f}"
                )

        if annotation_rows:
            ax.text(
                0.98,
                0.03,
                "\n".join(annotation_rows),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
            )

        format_temporal_axis(
            ax=ax,
            location=location,
            show_location_label=True,
        )
        ax.set_xlabel("Temporal lag between samples (months)")

    y_label = distance_metrics[metric]
    axes[0].set_ylabel(y_label)

    fig.legend(
        handles=city_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(cities),
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.5,
    )

    fig.subplots_adjust(
        top=0.80,
        bottom=0.18,
        left=0.07,
        right=0.99,
        wspace=0.32,
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

def plot_partitioned_community(
    pairwise: pd.DataFrame,
    statistics: pd.DataFrame,
    metric: str,
    output_file: Path,
) -> None:
    """Plot dominant/prevalent and remaining species in separate rows."""
    row_subsets = ["Dominant/prevalent", "Remaining"]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(16, 8.2),
        sharex=True,
        sharey=True,
    )

    for row_index, community_subset in enumerate(row_subsets):
        for column_index, location in enumerate(locations):
            ax = axes[row_index, column_index]

            annotation_rows = []

            for city in cities:
                result = add_city_to_axis(
                    ax=ax,
                    pairwise=pairwise,
                    statistics=statistics,
                    city=city,
                    location=location,
                    metric=metric,
                    community_subset=community_subset,
                )

                if result is not None:
                    annotation_rows.append(
                        f"{result['city']}: "
                        f"slope = {result['slope']:.4f}, "
                        f"R = {result['r_value']:.2f}"
                    )

            if annotation_rows:
                ax.text(
                    0.98,
                    0.03,
                    "\n".join(annotation_rows),
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=10,
                )

            format_temporal_axis(
                ax=ax,
                location=location,
                show_location_label=(row_index == 0),
            )

            if column_index == 0:
                ax.set_ylabel(
                    f"{community_subset} species\n"
                    f"{distance_metrics[metric]}"
                )
            else:
                ax.set_ylabel("")

            if row_index == 1:
                ax.set_xlabel(
                    "Temporal lag between samples (months)"
                )
            else:
                ax.set_xlabel("")

    fig.legend(
        handles=city_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(cities),
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.5,
    )

    fig.subplots_adjust(
        top=0.88,
        bottom=0.10,
        left=0.08,
        right=0.99,
        hspace=0.30,
        wspace=0.32,
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

sample_species = load_species_abundance(input_file)

classifications, pairwise_results, statistics_results = (
    calculate_all_results(sample_species)
)

classifications.to_csv(
    output_classification_tsv,
    sep="\t",
    index=False,
)

pairwise_results.to_csv(
    output_pairwise_tsv,
    sep="\t",
    index=False,
)

statistics_results.to_csv(
    output_statistics_tsv,
    sep="\t",
    index=False,
)

for metric in distance_metrics:
    plot_complete_community(
        pairwise=pairwise_results,
        statistics=statistics_results,
        metric=metric,
        output_file=output_complete_png[metric],
    )

    plot_partitioned_community(
        pairwise=pairwise_results,
        statistics=statistics_results,
        metric=metric,
        output_file=output_partitioned_png[metric],
    )

print("\nTemporal distance-decay analysis completed.")
print(
    "\nDominant/prevalent abundance threshold: "
    f"{dominant_abundance_fraction:.0%}"
)

print("\nSaved tables:")
print(f"  {output_classification_tsv}")
print(f"  {output_pairwise_tsv}")
print(f"  {output_statistics_tsv}")

print("\nSaved figures:")
for metric in distance_metrics:
    print(f"  {output_complete_png[metric]}")
    print(f"  {output_partitioned_png[metric]}")


weighted_figure_file = (
    output_dir / "Figure3_E-H_temporal_stability.png"
)
weighted_species_table_file = (
    output_dir / "Figure3_temporal_stability_species_metrics.tsv"
)
weighted_summary_table_file = (
    output_dir / "Figure3_temporal_stability_location_summary.tsv"
)
weighted_bootstrap_table_file = (
    output_dir / "Figure3_temporal_stability_bootstrap.tsv"
)
weighted_permutation_table_file = (
    output_dir / "Figure3_temporal_stability_permutation_tests.tsv"
)

number_of_bootstrap_iterations = 5000
number_of_weighted_permutations = 9999
weighted_random_seed = 12345

change_column = "median_absolute_monthly_change_log10"
directionality_column = "absolute_spearman_rho_month"
weight_column = "mean_BC_numerator_contribution"

def select_cumulative_abundance_species(
    abundance_matrix: pd.DataFrame,
    threshold: float,
) -> tuple[list[str], pd.Series]:
    """Select species cumulatively accounting for a mean-abundance fraction."""
    mean_abundance = (
        abundance_matrix.mean(axis=0)
        .sort_values(ascending=False)
    )

    total_abundance = mean_abundance.sum()

    if total_abundance <= 0:
        return [], mean_abundance

    cumulative_fraction = (
        mean_abundance.cumsum() / total_abundance
    )

    number_selected = int(
        np.searchsorted(
            cumulative_fraction.to_numpy(),
            threshold,
            side="left",
        )
        + 1
    )

    selected_species = mean_abundance.index[
        :number_selected
    ].tolist()

    return selected_species, mean_abundance

def mean_bray_numerator_contribution(
    values: np.ndarray,
) -> float:
    """Calculate mean absolute abundance difference across sample pairs."""
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return np.nan

    differences = [
        abs(values[index_1] - values[index_2])
        for index_1, index_2 in itertools.combinations(
            range(len(values)),
            2,
        )
    ]

    return float(np.mean(differences))

def weighted_median(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Calculate a weighted median."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    values = values[valid]
    weights = weights[valid]

    if len(values) == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative_weights = np.cumsum(weights)
    cutoff = weights.sum() / 2.0

    weighted_median_index = np.searchsorted(
        cumulative_weights,
        cutoff,
        side="left",
    )

    return float(values[weighted_median_index])

def scale_point_sizes(
    values: np.ndarray,
    minimum: float = 25,
    maximum: float = 220,
) -> np.ndarray:
    """Scale Bray–Curtis contribution values to marker areas."""
    values = np.asarray(values, dtype=float)
    transformed = np.log10(values + 1.0)

    minimum_value = np.nanmin(transformed)
    maximum_value = np.nanmax(transformed)

    if np.isclose(minimum_value, maximum_value):
        return np.full(
            len(values),
            (minimum + maximum) / 2,
        )

    return (
        minimum
        + (transformed - minimum_value)
        / (maximum_value - minimum_value)
        * (maximum - minimum)
    )

def holm_adjust(
    p_values: np.ndarray,
) -> np.ndarray:
    """Adjust p-values using the Holm procedure."""
    p_values = np.asarray(p_values, dtype=float)
    number_of_tests = len(p_values)
    order = np.argsort(p_values)

    adjusted = np.empty(
        number_of_tests,
        dtype=float,
    )

    running_maximum = 0.0

    for rank, index in enumerate(order):
        adjusted_value = (
            number_of_tests - rank
        ) * p_values[index]

        adjusted_value = max(
            adjusted_value,
            running_maximum,
        )

        adjusted[index] = min(
            adjusted_value,
            1.0,
        )

        running_maximum = adjusted[index]

    return adjusted

def calculate_weight_diagnostics(
    weights: np.ndarray,
) -> dict[str, float]:
    """Calculate Bray–Curtis weight-concentration diagnostics."""
    weights = np.asarray(weights, dtype=float)

    weights = weights[
        np.isfinite(weights)
        & (weights > 0)
    ]

    if len(weights) == 0:
        return {
            "maximum_weight_fraction": np.nan,
            "effective_number_of_species": np.nan,
        }

    normalised_weights = weights / weights.sum()

    return {
        "maximum_weight_fraction": float(
            normalised_weights.max()
        ),
        "effective_number_of_species": float(
            1.0 / np.sum(normalised_weights**2)
        ),
    }

def city_stratified_bootstrap(
    location_data: pd.DataFrame,
    iterations: int,
    random_generator: np.random.Generator,
) -> pd.DataFrame:
    """Bootstrap species within city and calculate pooled weighted medians."""
    bootstrap_rows = []

    available_cities = [
        city
        for city in cities
        if not location_data[
            location_data["city"] == city
        ].empty
    ]

    for iteration in range(iterations):
        resampled_tables = []

        for city in available_cities:
            city_data = location_data[
                location_data["city"] == city
            ]

            selected_indices = random_generator.choice(
                city_data.index.to_numpy(),
                size=len(city_data),
                replace=True,
            )

            resampled_tables.append(
                city_data.loc[selected_indices]
            )

        resampled_data = pd.concat(
            resampled_tables,
            ignore_index=True,
        )

        bootstrap_rows.append(
            {
                "iteration": iteration + 1,
                "weighted_median_change": weighted_median(
                    resampled_data[change_column],
                    resampled_data[weight_column],
                ),
                "weighted_median_directionality": weighted_median(
                    resampled_data[directionality_column],
                    resampled_data[weight_column],
                ),
            }
        )

    return pd.DataFrame(bootstrap_rows)

def pairwise_weighted_median_permutation_test(
    data: pd.DataFrame,
    location_1: str,
    location_2: str,
    metric: str,
    permutations: int,
    random_generator: np.random.Generator,
) -> dict[str, float]:
    """Compare two location-specific weighted medians."""
    subset = data[
        data["location"].isin(
            [location_1, location_2]
        )
    ].dropna(
        subset=[
            metric,
            weight_column,
            "city",
            "location",
        ]
    ).copy()

    subset = subset[
        subset[weight_column] > 0
    ].copy()

    observed_1 = weighted_median(
        subset.loc[
            subset["location"] == location_1,
            metric,
        ],
        subset.loc[
            subset["location"] == location_1,
            weight_column,
        ],
    )

    observed_2 = weighted_median(
        subset.loc[
            subset["location"] == location_2,
            metric,
        ],
        subset.loc[
            subset["location"] == location_2,
            weight_column,
        ],
    )

    observed_difference = observed_1 - observed_2

    original_locations = subset["location"].to_numpy()
    metric_values = subset[metric].to_numpy(dtype=float)
    weights = subset[weight_column].to_numpy(dtype=float)
    city_values = subset["city"].to_numpy()

    city_indices = {
        city: np.where(city_values == city)[0]
        for city in np.unique(city_values)
    }

    permuted_differences = np.empty(
        permutations,
        dtype=float,
    )

    for permutation_index in range(permutations):
        permuted_locations = original_locations.copy()

        for indices in city_indices.values():
            permuted_locations[indices] = (
                random_generator.permutation(
                    permuted_locations[indices]
                )
            )

        group_1 = permuted_locations == location_1
        group_2 = permuted_locations == location_2

        permuted_differences[
            permutation_index
        ] = (
            weighted_median(
                metric_values[group_1],
                weights[group_1],
            )
            - weighted_median(
                metric_values[group_2],
                weights[group_2],
            )
        )

    permuted_differences = permuted_differences[
        np.isfinite(permuted_differences)
    ]

    p_value = (
        1
        + np.sum(
            np.abs(permuted_differences)
            >= abs(observed_difference)
        )
    ) / (len(permuted_differences) + 1)

    return {
        "weighted_median_1": observed_1,
        "weighted_median_2": observed_2,
        "difference_1_minus_2": observed_difference,
        "p_value": float(p_value),
    }

def pairwise_joint_permutation_test(
    data: pd.DataFrame,
    location_1: str,
    location_2: str,
    permutations: int,
    random_generator: np.random.Generator,
    global_change_scale: float,
    global_directionality_scale: float,
) -> dict[str, float]:
    """Test joint separation of two weighted-median positions."""
    subset = data[
        data["location"].isin(
            [location_1, location_2]
        )
    ].dropna(
        subset=[
            change_column,
            directionality_column,
            weight_column,
            "city",
            "location",
        ]
    ).copy()

    subset = subset[
        subset[weight_column] > 0
    ].copy()

    def weighted_location_centre(
        labels: np.ndarray,
        target_location: str,
    ) -> np.ndarray:
        group_mask = labels == target_location

        return np.array(
            [
                weighted_median(
                    subset.loc[group_mask, change_column],
                    subset.loc[group_mask, weight_column],
                ),
                weighted_median(
                    subset.loc[
                        group_mask,
                        directionality_column,
                    ],
                    subset.loc[group_mask, weight_column],
                ),
            ],
            dtype=float,
        )

    original_locations = subset["location"].to_numpy()

    observed_centre_1 = weighted_location_centre(
        original_locations,
        location_1,
    )
    observed_centre_2 = weighted_location_centre(
        original_locations,
        location_2,
    )

    observed_difference = (
        observed_centre_1 - observed_centre_2
    )

    observed_scaled_distance = np.sqrt(
        (
            observed_difference[0]
            / global_change_scale
        ) ** 2
        + (
            observed_difference[1]
            / global_directionality_scale
        ) ** 2
    )

    city_values = subset["city"].to_numpy()

    city_indices = {
        city: np.where(city_values == city)[0]
        for city in np.unique(city_values)
    }

    permuted_distances = np.empty(
        permutations,
        dtype=float,
    )

    for permutation_index in range(permutations):
        permuted_locations = original_locations.copy()

        for indices in city_indices.values():
            permuted_locations[indices] = (
                random_generator.permutation(
                    permuted_locations[indices]
                )
            )

        centre_1 = weighted_location_centre(
            permuted_locations,
            location_1,
        )
        centre_2 = weighted_location_centre(
            permuted_locations,
            location_2,
        )

        difference = centre_1 - centre_2

        permuted_distances[
            permutation_index
        ] = np.sqrt(
            (
                difference[0]
                / global_change_scale
            ) ** 2
            + (
                difference[1]
                / global_directionality_scale
            ) ** 2
        )

    permuted_distances = permuted_distances[
        np.isfinite(permuted_distances)
    ]

    p_value = (
        1
        + np.sum(
            permuted_distances
            >= observed_scaled_distance
        )
    ) / (len(permuted_distances) + 1)

    return {
        "weighted_median_change_1": observed_centre_1[0],
        "weighted_median_change_2": observed_centre_2[0],
        "weighted_median_directionality_1": observed_centre_1[1],
        "weighted_median_directionality_2": observed_centre_2[1],
        "scaled_2D_distance": float(
            observed_scaled_distance
        ),
        "p_value": float(p_value),
    }

weighted_sample_species = sample_species.copy()

weighted_sample_species["sampling_date"] = pd.to_datetime(
    weighted_sample_species["year_month"] + "-01",
    errors="raise",
)

weighted_species_rows = []

for city in cities:
    for location in locations:
        group_data = weighted_sample_species[
            (weighted_sample_species["city"] == city)
            & (
                weighted_sample_species["location"]
                == location
            )
        ].copy()

        if group_data.empty:
            warnings.warn(
                f"No data were available for {city}, {location}."
            )
            continue

        abundance_matrix = (
            group_data.pivot_table(
                index="sampling_date",
                columns="species",
                values="RPKMF_norm",
                aggfunc="sum",
                fill_value=0.0,
            )
            .sort_index()
        )

        number_of_samples = abundance_matrix.shape[0]

        positive_samples = (
            abundance_matrix.gt(0).sum(axis=0)
        )
        occupancy = (
            positive_samples / number_of_samples
        )

        selected_species, mean_abundance = (
            select_cumulative_abundance_species(
                abundance_matrix,
                dominant_abundance_fraction,
            )
        )

        if not selected_species:
            continue

        total_mean_abundance = mean_abundance.sum()
        cumulative_fraction = (
            mean_abundance.cumsum()
            / total_mean_abundance
        )

        month_numbers = (
            abundance_matrix.index.year * 12
            + abundance_matrix.index.month
        ).to_numpy(dtype=float)

        month_numbers = (
            month_numbers - month_numbers.min()
        )

        for species in selected_species:
            raw_values = abundance_matrix[
                species
            ].to_numpy(dtype=float)

            log_values = np.log10(
                raw_values + 1.0
            )

            absolute_monthly_changes = np.abs(
                np.diff(log_values)
            )

            if np.std(log_values) > 0:
                spearman_rho, spearman_p_value = (
                    spearmanr(
                        month_numbers,
                        log_values,
                    )
                )
            else:
                spearman_rho = np.nan
                spearman_p_value = np.nan

            weighted_species_rows.append(
                {
                    "city": city,
                    "location": location,
                    "species": species,
                    "available_samples": number_of_samples,
                    "positive_samples": int(
                        positive_samples.loc[species]
                    ),
                    "occupancy": float(
                        occupancy.loc[species]
                    ),
                    "mean_normalised_RPKMF": float(
                        mean_abundance.loc[species]
                    ),
                    "fraction_of_total_mean_abundance": float(
                        mean_abundance.loc[species]
                        / total_mean_abundance
                    ),
                    "cumulative_abundance_fraction": float(
                        cumulative_fraction.loc[species]
                    ),
                    change_column: float(
                        np.median(
                            absolute_monthly_changes
                        )
                    ),
                    "mean_absolute_monthly_change_log10": float(
                        np.mean(
                            absolute_monthly_changes
                        )
                    ),
                    "spearman_rho_month": float(
                        spearman_rho
                    ),
                    directionality_column: (
                        float(abs(spearman_rho))
                        if np.isfinite(spearman_rho)
                        else np.nan
                    ),
                    "spearman_p_value": float(
                        spearman_p_value
                    ),
                    weight_column: float(
                        mean_bray_numerator_contribution(
                            raw_values
                        )
                    ),
                }
            )

weighted_species_results = pd.DataFrame(
    weighted_species_rows
)

weighted_species_results = (
    weighted_species_results
    .replace([np.inf, -np.inf], np.nan)
    .dropna(
        subset=[
            change_column,
            directionality_column,
            weight_column,
        ]
    )
)

weighted_species_results = weighted_species_results[
    weighted_species_results[weight_column] > 0
].copy()

if weighted_species_results.empty:
    raise ValueError(
        "No valid Bray–Curtis-weighted species results "
        "were generated."
    )

weighted_rng = np.random.default_rng(
    weighted_random_seed
)

weighted_summary_rows = []
weighted_bootstrap_tables = []

for location in locations:
    location_data = weighted_species_results[
        weighted_species_results["location"]
        == location
    ].copy()

    if location_data.empty:
        continue

    weighted_change = weighted_median(
        location_data[change_column],
        location_data[weight_column],
    )

    weighted_directionality = weighted_median(
        location_data[directionality_column],
        location_data[weight_column],
    )

    location_bootstrap = city_stratified_bootstrap(
        location_data=location_data,
        iterations=number_of_bootstrap_iterations,
        random_generator=weighted_rng,
    )

    location_bootstrap.insert(
        0,
        "location",
        location,
    )

    weighted_bootstrap_tables.append(
        location_bootstrap
    )

    diagnostics = calculate_weight_diagnostics(
        location_data[weight_column]
    )

    weighted_summary_rows.append(
        {
            "location": location,
            "number_of_selected_species": len(
                location_data
            ),
            "number_of_TRE_species": int(
                (
                    location_data["city"] == "TRE"
                ).sum()
            ),
            "number_of_KUO_species": int(
                (
                    location_data["city"] == "KUO"
                ).sum()
            ),
            "weighted_median_change": weighted_change,
            "change_CI_lower_95": float(
                np.nanpercentile(
                    location_bootstrap[
                        "weighted_median_change"
                    ],
                    2.5,
                )
            ),
            "change_CI_upper_95": float(
                np.nanpercentile(
                    location_bootstrap[
                        "weighted_median_change"
                    ],
                    97.5,
                )
            ),
            "weighted_median_directionality": (
                weighted_directionality
            ),
            "directionality_CI_lower_95": float(
                np.nanpercentile(
                    location_bootstrap[
                        "weighted_median_directionality"
                    ],
                    2.5,
                )
            ),
            "directionality_CI_upper_95": float(
                np.nanpercentile(
                    location_bootstrap[
                        "weighted_median_directionality"
                    ],
                    97.5,
                )
            ),
            **diagnostics,
        }
    )

weighted_location_summary = pd.DataFrame(
    weighted_summary_rows
)

weighted_bootstrap_results = pd.concat(
    weighted_bootstrap_tables,
    ignore_index=True,
)

# Pairwise city-stratified permutation tests

global_change_scale = weighted_species_results[
    change_column
].std(ddof=0)

global_directionality_scale = weighted_species_results[
    directionality_column
].std(ddof=0)

if (
    not np.isfinite(global_change_scale)
    or global_change_scale == 0
):
    global_change_scale = 1.0

if (
    not np.isfinite(global_directionality_scale)
    or global_directionality_scale == 0
):
    global_directionality_scale = 1.0

weighted_permutation_rows = []

for location_1, location_2 in itertools.combinations(
    locations,
    2,
):
    change_test = (
        pairwise_weighted_median_permutation_test(
            data=weighted_species_results,
            location_1=location_1,
            location_2=location_2,
            metric=change_column,
            permutations=number_of_weighted_permutations,
            random_generator=weighted_rng,
        )
    )

    directionality_test = (
        pairwise_weighted_median_permutation_test(
            data=weighted_species_results,
            location_1=location_1,
            location_2=location_2,
            metric=directionality_column,
            permutations=number_of_weighted_permutations,
            random_generator=weighted_rng,
        )
    )

    joint_test = pairwise_joint_permutation_test(
        data=weighted_species_results,
        location_1=location_1,
        location_2=location_2,
        permutations=number_of_weighted_permutations,
        random_generator=weighted_rng,
        global_change_scale=global_change_scale,
        global_directionality_scale=(
            global_directionality_scale
        ),
    )

    weighted_permutation_rows.extend(
        [
            {
                "metric": "Weighted median monthly change",
                "location_1": location_1,
                "location_2": location_2,
                "estimate_1": change_test[
                    "weighted_median_1"
                ],
                "estimate_2": change_test[
                    "weighted_median_2"
                ],
                "difference_1_minus_2": change_test[
                    "difference_1_minus_2"
                ],
                "joint_scaled_distance": np.nan,
                "number_of_permutations": (
                    number_of_weighted_permutations
                ),
                "p_value": change_test["p_value"],
            },
            {
                "metric": "Weighted median directionality",
                "location_1": location_1,
                "location_2": location_2,
                "estimate_1": directionality_test[
                    "weighted_median_1"
                ],
                "estimate_2": directionality_test[
                    "weighted_median_2"
                ],
                "difference_1_minus_2": (
                    directionality_test[
                        "difference_1_minus_2"
                    ]
                ),
                "joint_scaled_distance": np.nan,
                "number_of_permutations": (
                    number_of_weighted_permutations
                ),
                "p_value": directionality_test[
                    "p_value"
                ],
            },
            {
                "metric": "Joint 2D weighted-median position",
                "location_1": location_1,
                "location_2": location_2,
                "estimate_1": np.nan,
                "estimate_2": np.nan,
                "difference_1_minus_2": np.nan,
                "joint_scaled_distance": joint_test[
                    "scaled_2D_distance"
                ],
                "number_of_permutations": (
                    number_of_weighted_permutations
                ),
                "p_value": joint_test["p_value"],
            },
        ]
    )

weighted_permutation_results = pd.DataFrame(
    weighted_permutation_rows
)

weighted_permutation_results["p_value_Holm"] = np.nan
weighted_permutation_results[
    "significant_Holm_0.05"
] = False

for metric_name in weighted_permutation_results[
    "metric"
].unique():
    metric_mask = (
        weighted_permutation_results["metric"]
        == metric_name
    )

    valid_mask = (
        metric_mask
        & weighted_permutation_results[
            "p_value"
        ].notna()
    )

    adjusted_p_values = holm_adjust(
        weighted_permutation_results.loc[
            valid_mask,
            "p_value",
        ].to_numpy()
    )

    weighted_permutation_results.loc[
        valid_mask,
        "p_value_Holm",
    ] = adjusted_p_values

    weighted_permutation_results.loc[
        valid_mask,
        "significant_Holm_0.05",
    ] = adjusted_p_values < alpha_threshold

weighted_species_results.sort_values(
    [
        "location",
        "city",
        weight_column,
    ],
    ascending=[
        True,
        True,
        False,
    ],
).to_csv(
    weighted_species_table_file,
    sep="\t",
    index=False,
)

weighted_location_summary.to_csv(
    weighted_summary_table_file,
    sep="\t",
    index=False,
)

weighted_bootstrap_results.to_csv(
    weighted_bootstrap_table_file,
    sep="\t",
    index=False,
)

weighted_permutation_results.to_csv(
    weighted_permutation_table_file,
    sep="\t",
    index=False,
)

weighted_species_results["point_size"] = (
    scale_point_sizes(
        weighted_species_results[
            weight_column
        ].to_numpy()
    )
)

global_x_max = weighted_species_results[
    change_column
].max()

if (
    not np.isfinite(global_x_max)
    or global_x_max <= 0
):
    global_x_max = 1.0

fig, axes = plt.subplots(
    nrows=1,
    ncols=4,
    figsize=(16, 4.8),
    sharex=True,
    sharey=True,
)

for ax, location in zip(
    axes,
    locations,
):
    panel = weighted_species_results[
        weighted_species_results["location"]
        == location
    ].copy()

    summary = weighted_location_summary[
        weighted_location_summary["location"]
        == location
    ]

    if panel.empty or summary.empty:
        ax.set_visible(False)
        continue

    for city in cities:
        city_panel = panel[
            panel["city"] == city
        ]

        if city_panel.empty:
            continue

        ax.scatter(
            city_panel[change_column],
            city_panel[directionality_column],
            s=city_panel["point_size"],
            marker=city_markers[city],
            color=palette[location],
            alpha=0.60,
            edgecolor="black",
            linewidth=0.35,
        )

    weighted_change = float(
        summary.iloc[0]["weighted_median_change"]
    )
    weighted_directionality = float(
        summary.iloc[0][
            "weighted_median_directionality"
        ]
    )

    horizontal_errors = np.array(
        [
            [
                weighted_change
                - float(
                    summary.iloc[0][
                        "change_CI_lower_95"
                    ]
                )
            ],
            [
                float(
                    summary.iloc[0][
                        "change_CI_upper_95"
                    ]
                )
                - weighted_change
            ],
        ]
    )

    vertical_errors = np.array(
        [
            [
                weighted_directionality
                - float(
                    summary.iloc[0][
                        "directionality_CI_lower_95"
                    ]
                )
            ],
            [
                float(
                    summary.iloc[0][
                        "directionality_CI_upper_95"
                    ]
                )
                - weighted_directionality
            ],
        ]
    )

    ax.errorbar(
        weighted_change,
        weighted_directionality,
        xerr=horizontal_errors,
        yerr=vertical_errors,
        fmt="X",
        color="black",
        markersize=13,
        markeredgewidth=1.2,
        capsize=4,
        elinewidth=1.8,
        zorder=6,
    )

    ax.set_title(location)
    ax.set_xlim(
        -0.02,
        global_x_max * 1.05,
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(
        "Median absolute monthly change\n"
        "in log10(normalised RPKMF + 1)"
    )
    ax.tick_params(
        axis="both",
        direction="out",
    )
    ax.grid(False)
    sns.despine(ax=ax)

axes[0].set_ylabel(
    "Absolute temporal directionality\n"
    "|Spearman correlation with month|"
)

weighted_city_handles = [
    Line2D(
        [0],
        [0],
        marker=city_markers[city],
        linestyle="",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=8,
        label=city,
    )
    for city in cities
]

weighted_median_handle = Line2D(
    [0],
    [0],
    marker="X",
    linestyle="",
    markerfacecolor="black",
    markeredgecolor="black",
    markersize=9,
    label="BC-weighted median ± 95% CI",
)

fig.legend(
    handles=weighted_city_handles
    + [weighted_median_handle],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    handletextpad=0.5,
    columnspacing=1.5,
)

fig.subplots_adjust(
    top=0.80,
    bottom=0.20,
    left=0.07,
    right=0.99,
    wspace=0.32,
)

fig.savefig(
    weighted_figure_file,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print(
    "\nBray–Curtis-weighted temporal behaviour "
    "analysis completed."
)

print("\nSaved weighted-temporal-behaviour outputs:")
print(f"  {weighted_figure_file}")
print(f"  {weighted_species_table_file}")
print(f"  {weighted_summary_table_file}")
print(f"  {weighted_bootstrap_table_file}")
print(f"  {weighted_permutation_table_file}")

species_results = weighted_species_results
sample_species = weighted_sample_species
abundance_column = "RPKMF_norm"
dpi = 300


target_locations = [
    "UH",
    "WWTP",
]

target_strata = [
    ("TRE", "UH"),
    ("KUO", "UH"),
    ("TRE", "WWTP"),
    ("KUO", "WWTP"),
]

stratum_labels = [
    "TRE_UH",
    "KUO_UH",
    "TRE_WWTP",
    "KUO_WWTP",
]

number_of_trajectory_species = 4

contribution_figure_file = (
    output_dir / "Figure3_I_Bray_Curtis_species_contributions.png"
)

trajectory_figure_file = (
    output_dir / "UH_WWTP_top4_monthly_trajectories.png"
)

plotting_table_file = (
    output_dir / "Figure3_IJ_plotting_data.tsv"
)

additional_figure_dpi = dpi

def shorten_species_name(
    name,
    maximum_length=48,
):
    """Shorten long species labels for plotting."""

    text = str(name)

    if text.startswith("s__"):
        text = text[3:]

    if len(text) <= maximum_length:
        return text

    left_length = (
        maximum_length // 2
        - 2
    )

    right_length = (
        maximum_length
        - left_length
        - 3
    )

    return (
        text[:left_length]
        + "..."
        + text[-right_length:]
    )

def build_complete_stratum_table(
    sample_species_table,
    city,
    location,
    species_list,
):
    """
    Build a complete sample × species table for one city-location stratum.

    Missing species-sample combinations are filled with zero, but only for
    species explicitly requested in species_list.
    """

    metadata = (
        sample_species_table.loc[
            sample_species_table[
                "city"
            ].eq(city)
            & sample_species_table[
                "location"
            ].eq(location),
            [
                "sample_ID",
                "sampling_date",
            ],
        ]
        .drop_duplicates()
    )

    sample_ids = (
        metadata[
            "sample_ID"
        ]
        .tolist()
    )

    if len(sample_ids) == 0:

        return pd.DataFrame(
            columns=[
                "sample_ID",
                "species",
                abundance_column,
                "sampling_date",
                "city",
                "location",
                "stratum",
            ]
        )

    full_index = pd.MultiIndex.from_product(
        [
            sample_ids,
            species_list,
        ],
        names=[
            "sample_ID",
            "species",
        ],
    )

    observed = (
        sample_species_table.loc[
            sample_species_table[
                "city"
            ].eq(city)
            & sample_species_table[
                "location"
            ].eq(location)
            & sample_species_table[
                "species"
            ].isin(
                species_list
            )
        ]
        .groupby(
            [
                "sample_ID",
                "species",
            ],
            observed=True,
        )[abundance_column]
        .sum()
        .reindex(
            full_index,
            fill_value=0.0,
        )
        .rename(
            abundance_column
        )
        .reset_index()
    )

    observed = observed.merge(
        metadata,
        on="sample_ID",
        how="left",
    )

    observed[
        "city"
    ] = city

    observed[
        "location"
    ] = location

    observed[
        "stratum"
    ] = (
        f"{city}_{location}"
    )

    return observed

uh_wwtp_species_results = species_results.loc[
    species_results[
        "location"
    ].isin(
        target_locations
    )
].copy()

all_species = (
    uh_wwtp_species_results[
        "species"
    ]
    .drop_duplicates()
    .tolist()
)

if len(all_species) == 0:

    raise ValueError(
        "No viruses were retained for UH or WWTP."
    )

uh_wwtp_species_results[
    "stratum"
] = (
    uh_wwtp_species_results[
        "city"
    ]
    + "_"
    + uh_wwtp_species_results[
        "location"
    ]
)

print(
    f"\nUnique viruses retained across UH and WWTP: "
    f"{len(all_species)}"
)

bc_matrix = (
    uh_wwtp_species_results.pivot_table(
        index="species",
        columns="stratum",
        values=weight_column,
        aggfunc="first",
        fill_value=0.0,
    )
    .reindex(
        index=all_species,
        columns=stratum_labels,
        fill_value=0.0,
    )
)

bc_matrix[
    "UH_total"
] = (
    bc_matrix[
        "TRE_UH"
    ]
    + bc_matrix[
        "KUO_UH"
    ]
)

bc_matrix[
    "WWTP_total"
] = (
    bc_matrix[
        "TRE_WWTP"
    ]
    + bc_matrix[
        "KUO_WWTP"
    ]
)

bc_matrix[
    "overall_total"
] = (
    bc_matrix[
        "UH_total"
    ]
    + bc_matrix[
        "WWTP_total"
    ]
)

species_order = (
    bc_matrix[
        "overall_total"
    ]
    .sort_values(
        ascending=False
    )
    .index
    .tolist()
)

bc_matrix = bc_matrix.reindex(
    species_order
)

top_trajectory_species = (
    bc_matrix[
        "overall_total"
    ]
    .nlargest(
        number_of_trajectory_species
    )
    .index
    .tolist()
)

retention_mask = (
    uh_wwtp_species_results.assign(
        retained=True
    )
    .pivot_table(
        index="species",
        columns="stratum",
        values="retained",
        aggfunc="max",
        fill_value=False,
    )
    .reindex(
        index=species_order,
        columns=stratum_labels,
        fill_value=False,
    )
    .astype(bool)
)

complete_blocks = []

for city, location in target_strata:

    complete_blocks.append(
        build_complete_stratum_table(
            sample_species_table=sample_species,
            city=city,
            location=location,
            species_list=species_order,
        )
    )

complete_abundance = pd.concat(
    complete_blocks,
    ignore_index=True,
)

complete_abundance[
    "month"
] = (
    complete_abundance[
        "sampling_date"
    ]
    .dt.to_period("M")
    .dt.to_timestamp()
)

mean_abundance_matrix = (
    complete_abundance.groupby(
        [
            "species",
            "stratum",
        ],
        observed=True,
    )[abundance_column]
    .mean()
    .unstack(
        "stratum",
        fill_value=0.0,
    )
    .reindex(
        index=species_order,
        columns=stratum_labels,
        fill_value=0.0,
    )
)

masked_mean_abundance = (
    mean_abundance_matrix.where(
        retention_mask
    )
)

plotting_table = (
    bc_matrix[
        [
            "TRE_UH",
            "KUO_UH",
            "TRE_WWTP",
            "KUO_WWTP",
            "UH_total",
            "WWTP_total",
            "overall_total",
        ]
    ]
    .add_prefix(
        "BC_contribution_"
    )
    .join(
        masked_mean_abundance.add_prefix(
            "mean_RPKMF_norm_"
        )
    )
    .join(
        retention_mask.astype(int).add_prefix(
            "retained_"
        )
    )
    .reset_index()
)

plotting_table.to_csv(
    plotting_table_file,
    sep="\t",
    index=False,
)

number_of_species = len(species_order)

figure_height = max(
    7.0,
    0.40 * number_of_species + 2.0,
)

fig, axis = plt.subplots(
    figsize=(12.5, figure_height),
)

y_positions = np.arange(number_of_species)

tre_uh = bc_matrix["TRE_UH"].to_numpy()
kuo_uh = bc_matrix["KUO_UH"].to_numpy()
tre_wwtp = bc_matrix["TRE_WWTP"].to_numpy()
kuo_wwtp = bc_matrix["KUO_WWTP"].to_numpy()

axis.barh(
    y_positions,
    -tre_uh,
    height=0.68,
    color=palette["UH"],
    edgecolor="black",
    linewidth=0.5,
    label="TRE",
)

axis.barh(
    y_positions,
    -kuo_uh,
    left=-tre_uh,
    height=0.68,
    color=palette["UH"],
    edgecolor="black",
    linewidth=0.5,
    hatch="//",
    label="KUO",
)

axis.barh(
    y_positions,
    tre_wwtp,
    height=0.68,
    color=palette["WWTP"],
    edgecolor="black",
    linewidth=0.5,
)

axis.barh(
    y_positions,
    kuo_wwtp,
    left=tre_wwtp,
    height=0.68,
    color=palette["WWTP"],
    edgecolor="black",
    linewidth=0.5,
    hatch="//",
)

axis.axvline(
    0,
    color="black",
    linewidth=1.0,
)

axis.set_yticks(y_positions)
axis.set_yticklabels(
    [
        shorten_species_name(
            species,
            maximum_length=54,
        )
        for species in species_order
    ],
    fontsize=16,
    fontweight="bold",
)
axis.invert_yaxis()

maximum_total = max(
    bc_matrix["UH_total"].max(),
    bc_matrix["WWTP_total"].max(),
)

if (
    not np.isfinite(maximum_total)
    or maximum_total <= 0
):
    maximum_total = 1.0

axis.set_xlim(
    -maximum_total * 1.12,
    maximum_total * 1.12,
)

tick_locations = axis.get_xticks()
axis.set_xticks(tick_locations)
axis.set_xticklabels(
    [
        f"{abs(value):.3g}"
        for value in tick_locations
    ],
    fontsize=14,
)

axis.set_xlabel(
    "Summed mean Bray–Curtis numerator contribution"
)
axis.tick_params(
    axis="both",
    direction="out",
)
axis.grid(False)
sns.despine(
    ax=axis,
    left=False,
    bottom=False,
)

location_handles = [
    Line2D(
        [0],
        [0],
        color=palette["UH"],
        linewidth=10,
        label="UH",
    ),
    Line2D(
        [0],
        [0],
        color=palette["WWTP"],
        linewidth=8,
        label="WWTP",
    ),
]

city_handles = [
    mpl.patches.Patch(
        facecolor="white",
        edgecolor="black",
        label="TRE",
    ),
    mpl.patches.Patch(
        facecolor="white",
        edgecolor="black",
        hatch="//",
        label="KUO",
    ),
]

location_legend = fig.legend(
    handles=location_handles,
    loc="upper center",
    bbox_to_anchor=(0.40, 1.005),
    ncol=2,
    frameon=False,
    handlelength=1.2,
    columnspacing=1.5,
    fontsize=14,
    title_fontsize=14,
)

fig.add_artist(location_legend)

fig.legend(
    handles=city_handles,
    loc="upper center",
    bbox_to_anchor=(0.72, 1.005),
    ncol=2,
    frameon=False,
    handlelength=1.2,
    columnspacing=1.5,
    fontsize=14,
    title_fontsize=14,
)

fig.subplots_adjust(
    top=0.94,
    bottom=0.08,
    left=0.34,
    right=0.98,
)

fig.savefig(
    contribution_figure_file,
    dpi=additional_figure_dpi,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

ribbon_figure_file = (
    output_dir
    / "Figure3_J_monthly_abundance_ribbons.png"
)

ribbon_table_file = (
    output_dir
    / "Figure3_J_monthly_abundance_ribbon_data.tsv"
)

ribbon_strata = [
    ("TRE", "UH"),
    ("KUO", "UH"),
    ("TRE", "WWTP"),
    ("KUO", "WWTP"),
]

ribbon_labels = [
    "TRE–UH",
    "KUO–UH",
    "TRE–WWTP",
    "KUO–WWTP",
]

ribbon_monthly = (
    complete_abundance.loc[
        complete_abundance["species"].isin(
            top_trajectory_species
        )
    ]
    .groupby(
        [
            "species",
            "city",
            "location",
            "month",
        ],
        observed=True,
        as_index=False,
    )[abundance_column]
    .mean()
)

if ribbon_monthly.empty:
    raise ValueError(
        "No monthly abundance data were available for "
        "the selected trajectory species."
    )

all_months = pd.date_range(
    start=ribbon_monthly["month"].min(),
    end=ribbon_monthly["month"].max(),
    freq="MS",
)

ribbon_rows = []

for species in top_trajectory_species:
    species_data = ribbon_monthly.loc[
        ribbon_monthly["species"].eq(species)
    ].copy()

    for city, location in ribbon_strata:
        stratum_values = (
            species_data.loc[
                species_data["city"].eq(city)
                & species_data["location"].eq(location)
            ]
            .set_index("month")[abundance_column]
            .reindex(
                all_months,
                fill_value=0.0,
            )
        )

        for month, abundance_value in stratum_values.items():
            ribbon_rows.append(
                {
                    "species": species,
                    "city": city,
                    "location": location,
                    "stratum": f"{city}_{location}",
                    "month": month,
                    "mean_normalised_RPKMF": float(
                        abundance_value
                    ),
                    "log10_mean_normalised_RPKMF_plus_1": float(
                        np.log10(
                            abundance_value + 1.0
                        )
                    ),
                }
            )

ribbon_data = pd.DataFrame(ribbon_rows)

if ribbon_data.empty:
    raise ValueError(
        "The ribbon plotting table is empty."
    )

ribbon_data.to_csv(
    ribbon_table_file,
    sep="\t",
    index=False,
)

global_colour_maximum = ribbon_data[
    "log10_mean_normalised_RPKMF_plus_1"
].max()

if (
    not np.isfinite(global_colour_maximum)
    or global_colour_maximum <= 0
):
    global_colour_maximum = 1.0

colour_normalisation = Normalize(
    vmin=0.0,
    vmax=float(global_colour_maximum),
)

uh_cmap = LinearSegmentedColormap.from_list(
    "UH_orange",
    [
        "#FFFFFF",
        palette["UH"],
    ],
)

wwtp_cmap = LinearSegmentedColormap.from_list(
    "WWTP_blue",
    [
        "#FFFFFF",
        palette["WWTP"],
    ],
)

legend_cmap = plt.get_cmap("Greys")

number_of_species = len(
    top_trajectory_species
)

fig, axes = plt.subplots(
    nrows=number_of_species,
    ncols=1,
    figsize=(
        6,
        1 * number_of_species + 2,
    ),
    sharex=True,
    squeeze=False,
)

axes = axes.ravel()

for axis, species in zip(
    axes,
    top_trajectory_species,
):
    species_data = ribbon_data.loc[
        ribbon_data["species"].eq(species)
    ].copy()

    matrix = np.zeros(
        (
            len(ribbon_strata),
            len(all_months),
        ),
        dtype=float,
    )

    for row_index, (city, location) in enumerate(
        ribbon_strata
    ):
        matrix[row_index, :] = (
            species_data.loc[
                species_data["city"].eq(city)
                & species_data["location"].eq(location)
            ]
            .set_index("month")[
                "log10_mean_normalised_RPKMF_plus_1"
            ]
            .reindex(
                all_months,
                fill_value=0.0,
            )
            .to_numpy(dtype=float)
        )

    for row_index, (_, location) in enumerate(
        ribbon_strata
    ):
        colour_map = (
            uh_cmap
            if location == "UH"
            else wwtp_cmap
        )

        axis.imshow(
            matrix[
                row_index : row_index + 1,
                :,
            ],
            aspect="auto",
            interpolation="nearest",
            cmap=colour_map,
            norm=colour_normalisation,
            origin="upper",
            extent=[
                -0.5,
                len(all_months) - 0.5,
                row_index - 0.5,
                row_index + 0.5,
            ],
        )

    for boundary in [
        0.5,
        1.5,
        2.5,
    ]:
        axis.axhline(
            boundary,
            color="white",
            linewidth=0.8,
        )

    axis.axhline(
        1.5,
        color="black",
        linewidth=0.8,
    )

    axis.set_ylim(
        len(ribbon_strata) - 0.5,
        -0.5,
    )

    axis.set_yticks(
        np.arange(len(ribbon_labels))
    )

    axis.set_yticklabels(
        ribbon_labels,
        fontsize=12,
    )

    axis.set_title(
        shorten_species_name(
            species,
            maximum_length=70,
        ),
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=2,
    )

    axis.tick_params(
        axis="y",
        length=0,
    )

    axis.grid(False)

    for spine in axis.spines.values():
        spine.set_visible(False)

month_tick_indices = np.arange(
    0,
    len(all_months),
    2,
)

axes[-1].set_xticks(
    month_tick_indices
)

axes[-1].set_xticklabels(
    [
        all_months[index].strftime("%b %Y")
        for index in month_tick_indices
    ],
    ha="right",
    fontsize=12,
)

axes[-1].set_xlabel(
    "Sampling month"
)

for axis in axes[:-1]:
    axis.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )

colourbar_axis = fig.add_axes(
    [
        0.30,
        0.055,
        0.40,
        0.025,
    ]
)

colourbar = fig.colorbar(
    ScalarMappable(
        norm=colour_normalisation,
        cmap=legend_cmap,
    ),
    cax=colourbar_axis,
    orientation="horizontal",
)

colourbar.set_label(
    "log10(mean normalised RPKMF + 1)",
    fontsize=12,
)

colourbar.ax.tick_params(
    labelsize=10,
)

location_legend_handles = [
    mpl.patches.Patch(
        facecolor=palette["UH"],
        edgecolor="black",
        linewidth=0.5,
        label="UH",
    ),
    mpl.patches.Patch(
        facecolor=palette["WWTP"],
        edgecolor="black",
        linewidth=0.5,
        label="WWTP",
    ),
]

fig.legend(
    handles=location_legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=2,
    frameon=False,
    handletextpad=0.5,
    columnspacing=1.5,
)

fig.subplots_adjust(
    top=0.92,
    bottom=0.18,
    left=0.12,
    right=0.99,
    hspace=0.34,
)

fig.savefig(
    ribbon_figure_file,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print(
    "\\nSaved compact coloured trajectory ribbon outputs:"
)

print(f"  {ribbon_figure_file}")
print(f"  {ribbon_table_file}")
# Integrated from the separate dominant-virus peak-analysis script.
# No additional figures are generated in this section.
# Dominant species are defined independently within TRE-UH, KUO-UH,
# UH and WWTP are compared using two-sided city-stratified permutation tests.
# Holm correction is applied across the three tested temporal metrics.

peak_locations = ["UH", "WWTP"]
peak_relative_floor_fraction = 0.01
peak_n_permutations = 9999
peak_random_seed = 42

def peak_local_prominence(
    abundance: np.ndarray,
    relative_floor: float = peak_relative_floor_fraction,
) -> float:
    """Maximum fold elevation of an interior observation above its neighbours."""
    abundance = np.asarray(abundance, dtype=float)

    if abundance.size < 3 or abundance.max() <= 0:
        return np.nan

    floor = relative_floor * abundance.max()

    scores = [
        abundance[index]
        / max(
            (abundance[index - 1] + abundance[index + 1]) / 2.0,
            floor,
        )
        for index in range(1, abundance.size - 1)
    ]

    return float(max(scores))

def calculate_peak_metrics(
    metadata: pd.DataFrame,
    abundance_matrix: pd.DataFrame,
    selected_species: list[str],
) -> pd.DataFrame:
    """Calculate temporal peak/change metrics for dominant viral species."""
    rows = []

    month_index = metadata["month_index"].to_numpy(dtype=int)
    consecutive_months = np.diff(month_index) == 1

    city = str(metadata["city"].iloc[0])
    location = str(metadata["location"].iloc[0])
    group = str(metadata["group"].iloc[0])

    for species in selected_species:
        abundance = abundance_matrix[species].to_numpy(dtype=float)

        monthly_changes = np.abs(
            np.diff(abundance)[consecutive_months]
        )

        median_abundance = float(np.median(abundance))
        maximum_abundance = float(np.max(abundance))

        if median_abundance > 0:
            peak_to_median = maximum_abundance / median_abundance
        elif maximum_abundance > 0:
            peak_to_median = np.inf
        else:
            peak_to_median = np.nan

        rows.append(
            {
                "city": city,
                "location": location,
                "group": group,
                "species": species,
                "median_absolute_monthly_change": (
                    float(np.median(monthly_changes))
                    if monthly_changes.size
                    else np.nan
                ),
                "peak_to_median_ratio": float(peak_to_median),
                "maximum_local_peak_prominence": peak_local_prominence(
                    abundance
                ),
            }
        )

    return pd.DataFrame(rows)

def city_stratified_uh_wwtp_test(
    data: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
) -> dict[str, float | str]:
    """Compare UH and WWTP medians by permuting location within each city."""
    subset = (
        data[["city", "location", "species", metric]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    uh = subset.loc[
        subset["location"].eq("UH"), metric
    ].to_numpy(dtype=float)

    wwtp = subset.loc[
        subset["location"].eq("WWTP"), metric
    ].to_numpy(dtype=float)

    if uh.size == 0 or wwtp.size == 0:
        return {
            "metric": metric,
            "median_UH": np.nan,
            "median_WWTP": np.nan,
            "difference_UH_minus_WWTP": np.nan,
            "permutation_p_two_sided": np.nan,
        }

    observed = float(np.median(uh) - np.median(wwtp))
    permuted = np.empty(peak_n_permutations, dtype=float)

    for permutation_index in range(peak_n_permutations):
        shuffled_parts = []

        for _, city_data in subset.groupby("city", sort=False):
            city_data = city_data.copy()
            city_data["location"] = rng.permutation(
                city_data["location"].to_numpy()
            )
            shuffled_parts.append(city_data)

        shuffled = pd.concat(shuffled_parts, ignore_index=True)

        permuted[permutation_index] = (
            shuffled.loc[
                shuffled["location"].eq("UH"), metric
            ].median()
            - shuffled.loc[
                shuffled["location"].eq("WWTP"), metric
            ].median()
        )

    p_value = (
        np.sum(np.abs(permuted) >= abs(observed)) + 1
    ) / (peak_n_permutations + 1)

    return {
        "metric": metric,
        "median_UH": float(np.median(uh)),
        "median_WWTP": float(np.median(wwtp)),
        "difference_UH_minus_WWTP": observed,
        "permutation_p_two_sided": float(p_value),
    }

def run_peak_statistics(
    sample_species_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate dominant-virus temporal metrics and UH-versus-WWTP tests."""
    metric_tables = []

    for city in cities:
        for location in peak_locations:
            group_name = f"{city}_{location}"

            metadata, abundance_matrix = build_group_matrix(
                sample_species_table,
                group_name,
            )

            if abundance_matrix.empty:
                warnings.warn(
                    f"Skipping peak analysis for {group_name}: no samples."
                )
                continue

            classification = classify_species(
                abundance_matrix,
                cumulative_fraction=dominant_abundance_fraction,
            )

            selected_species = classification.loc[
                classification["dominant_prevalent"],
                "species",
            ].tolist()

            if not selected_species:
                warnings.warn(
                    f"Skipping peak analysis for {group_name}: "
                    "no dominant species."
                )
                continue

            metric_tables.append(
                calculate_peak_metrics(
                    metadata,
                    abundance_matrix,
                    selected_species,
                )
            )

    if not metric_tables:
        raise ValueError(
            "No dominant-virus temporal peak metrics were generated."
        )

    metrics = pd.concat(metric_tables, ignore_index=True)

    rng = np.random.default_rng(peak_random_seed)
    tested_metrics = [
        "median_absolute_monthly_change",
        "peak_to_median_ratio",
        "maximum_local_peak_prominence",
    ]

    tests = pd.DataFrame(
        [
            city_stratified_uh_wwtp_test(metrics, metric, rng)
            for metric in tested_metrics
        ]
    )

    tests["p_Holm"] = np.nan
    tests["significant_Holm"] = False

    valid = tests["permutation_p_two_sided"].notna()

    if valid.any():
        rejected, adjusted, _, _ = multipletests(
            tests.loc[valid, "permutation_p_two_sided"],
            alpha=alpha_threshold,
            method="holm",
        )

        tests.loc[valid, "p_Holm"] = adjusted
        tests.loc[valid, "significant_Holm"] = rejected

    return metrics, tests

peak_metrics, peak_tests = run_peak_statistics(sample_species)

peak_metrics_file = output_dir / "Figure3_dominant_virus_peak_metrics.tsv"
peak_tests_file = output_dir / "Figure3_UH_vs_WWTP_peak_metric_tests.tsv"

peak_metrics.to_csv(
    peak_metrics_file,
    sep="\t",
    index=False,
)

peak_tests.to_csv(
    peak_tests_file,
    sep="\t",
    index=False,
)

print("\nAdditional dominant-virus temporal tests completed.")
print(f"  {peak_metrics_file}")
print(f"  {peak_tests_file}")
