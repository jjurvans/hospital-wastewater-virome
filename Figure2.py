#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:08:38 2026

@author: jaanajurvansuu

Figure 2. Abundance, alpha diversity, and beta diversity of human-associated viral species.

Panels A–D: Kuopio; E–H: Tampere.
Metrics: total normalised RPKMF, observed viral species richness,
Shannon diversity, and Gini–Simpson diversity.

Panels I–J: PCoA using Bray–Curtis and Jaccard dissimilarities.
Panels K–N: multivariate dispersion shown as distance to the
sampling-location spatial median.

Post-hoc analyses are performed only when the corresponding
Bonferroni-corrected omnibus test is significant.
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal
from skbio import DistanceMatrix
from skbio.stats.distance import permanova, permdisp
from skbio.stats.ordination import pcoa
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Settings
# =============================================================================

# Repository paths
# The script and input data are stored directly in the repository root.
ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "EsViritu_abundance_metadata.tsv"
OUTPUT_DIR = ROOT

ALPHA = 0.05
N_PERMUTATIONS = 9999
RANDOM_SEED = 42
PRESENCE_THRESHOLD = 0.0

CITIES = ["KUO", "TRE"]
LOCATIONS = ["UH", "WWTP", "SA1", "SA2"]

CITY_LABELS = {
    "KUO": "Kuopio",
    "TRE": "Tampere",
}

PALETTE = {
    "UH": "#E67E22",
    "WWTP": "#2E86C1",
    "SA1": "#52BE80",
    "SA2": "#239B56",
}

CITY_MARKERS = {
    "KUO": "s",
    "TRE": "o",
}

METRICS = {
    "total_normalised_RPKMF": "Total normalised RPKMF",
    "observed_viral_species": "Observed viral species",
    "shannon": "Shannon diversity",
    "gini_simpson": "Gini–Simpson diversity",
}

DISTANCES = {
    "Bray–Curtis": "braycurtis",
    "Jaccard": "jaccard",
}


# =============================================================================
# General helpers
# =============================================================================

def bonferroni_adjust(
    table: pd.DataFrame,
    family_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Add Bonferroni-adjusted p-values within defined test families."""
    table = table.copy()
    table["p_Bonferroni"] = np.nan
    table["significant_Bonferroni"] = False

    if table.empty:
        return table

    if family_columns:
        groups = table.groupby(
            family_columns,
            dropna=False,
            observed=False,
        ).groups.values()
    else:
        groups = [table.index]

    for indices in groups:
        indices = list(indices)
        valid = table.loc[indices, "p_raw"].notna()
        valid_indices = table.loc[indices].index[valid]

        if len(valid_indices) == 0:
            continue

        rejected, adjusted, _, _ = multipletests(
            table.loc[valid_indices, "p_raw"],
            alpha=ALPHA,
            method="bonferroni",
        )

        table.loc[valid_indices, "p_Bonferroni"] = adjusted
        table.loc[valid_indices, "significant_Bonferroni"] = rejected

    return table


def configure_plotting() -> None:
    """Apply common plotting settings."""
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 14,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
        }
    )
    sns.set_theme(style="ticks", context="notebook")


# =============================================================================
# Data preparation
# =============================================================================

def read_data(path: Path) -> pd.DataFrame:
    """Read, validate, filter, and annotate the input data."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {"sample_ID", "host_genus", "species", "RPKMF_norm"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    df["RPKMF_norm"] = pd.to_numeric(df["RPKMF_norm"], errors="coerce")

    invalid = df["RPKMF_norm"].isna().sum()
    if invalid:
        warnings.warn(
            f"{invalid} non-numeric RPKMF_norm values were converted to zero."
        )
        df["RPKMF_norm"] = df["RPKMF_norm"].fillna(0.0)

    negative = (df["RPKMF_norm"] < 0).sum()
    if negative:
        raise ValueError(f"Found {negative} negative RPKMF_norm values.")

    df = df[
        df["host_genus"].str.strip().str.casefold() == "homo"
    ].copy()

    if df.empty:
        raise ValueError("No records remained after host_genus == 'Homo'.")

    parts = df["sample_ID"].str.split("_", expand=True)
    if parts.shape[1] < 4:
        raise ValueError(
            "sample_ID must contain at least four underscore-separated "
            "components, with city in component 3 and location in component 4."
        )

    df["city"] = parts[2].str.strip()
    df["location"] = parts[3].str.strip()

    df = df[
        df["city"].isin(CITIES) & df["location"].isin(LOCATIONS)
    ].copy()

    blank_species = df["species"].str.strip().eq("")
    if blank_species.any():
        warnings.warn(
            f"Removing {blank_species.sum()} rows with blank species identifiers."
        )
        df = df.loc[~blank_species].copy()

    if df.empty:
        raise ValueError("No observations remained after filtering.")

    return df


def aggregate_species(df: pd.DataFrame) -> pd.DataFrame:
    """Sum normalized RPKMF for each species within each sample."""
    return (
        df.groupby(
            ["sample_ID", "city", "location", "species"],
            as_index=False,
            observed=True,
        )
        .agg(RPKMF_norm=("RPKMF_norm", "sum"))
    )


# =============================================================================
# Alpha diversity
# =============================================================================

def positive_abundances(values) -> np.ndarray:
    """Return finite positive abundance values."""
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x) & (x > 0)]


def shannon(values) -> float:
    """Shannon diversity: -sum(p_i * ln(p_i))."""
    x = positive_abundances(values)
    if x.size == 0:
        return 0.0
    p = x / x.sum()
    return float(-np.sum(p * np.log(p)))


def gini_simpson(values) -> float:
    """Gini–Simpson diversity: 1 - sum(p_i^2)."""
    x = positive_abundances(values)
    if x.size == 0:
        return 0.0
    p = x / x.sum()
    return float(1.0 - np.sum(p**2))


def calculate_alpha_metrics(abund: pd.DataFrame) -> pd.DataFrame:
    """Calculate abundance and alpha-diversity metrics per sample."""

    def summarize(group: pd.DataFrame) -> pd.Series:
        values = group["RPKMF_norm"].to_numpy(dtype=float)
        return pd.Series(
            {
                "total_normalised_RPKMF": float(values.sum()),
                "observed_viral_species": int((values > 0).sum()),
                "shannon": shannon(values),
                "gini_simpson": gini_simpson(values),
            }
        )

    alpha = (
        abund.groupby(
            ["sample_ID", "city", "location"],
            observed=True,
        )
        .apply(summarize, include_groups=False)
        .reset_index()
    )

    alpha["city"] = pd.Categorical(
        alpha["city"], categories=CITIES, ordered=True
    )
    alpha["location"] = pd.Categorical(
        alpha["location"], categories=LOCATIONS, ordered=True
    )

    return alpha.sort_values(
        ["city", "location", "sample_ID"]
    ).reset_index(drop=True)


def epsilon_squared_kruskal(h: float, n: int, k: int) -> float:
    """Kruskal–Wallis epsilon-squared effect size."""
    if n <= k:
        return np.nan
    return float(max(0.0, (h - k + 1) / (n - k)))


def run_alpha_statistics(
    alpha: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Kruskal–Wallis tests within each city.

    Dunn tests are performed only when the corresponding omnibus test is
    significant after Bonferroni correction across all 8 omnibus tests.
    """
    omnibus_rows = []

    for city, metric in itertools.product(CITIES, METRICS):
        subset = alpha.loc[alpha["city"] == city]

        arrays = [
            subset.loc[subset["location"] == location, metric]
            .dropna()
            .to_numpy()
            for location in LOCATIONS
            if subset.loc[
                subset["location"] == location, metric
            ].dropna().shape[0] > 0
        ]

        if len(arrays) < 2:
            h, p = np.nan, np.nan
            n_total, n_groups = sum(map(len, arrays)), len(arrays)
        else:
            n_total = sum(map(len, arrays))
            n_groups = len(arrays)
            try:
                result = kruskal(*arrays, nan_policy="omit")
                h, p = float(result.statistic), float(result.pvalue)
            except ValueError as exc:
                if "All numbers are identical" in str(exc):
                    h, p = 0.0, 1.0
                else:
                    raise

        omnibus_rows.append(
            {
                "city": city,
                "metric": metric,
                "metric_label": METRICS[metric],
                "n_total": n_total,
                "n_groups": n_groups,
                "H": h,
                "df": n_groups - 1 if n_groups else np.nan,
                "p_raw": p,
                "epsilon_squared": (
                    epsilon_squared_kruskal(h, n_total, n_groups)
                    if np.isfinite(h)
                    else np.nan
                ),
            }
        )

    omnibus = bonferroni_adjust(pd.DataFrame(omnibus_rows))

    posthoc_rows = []

    for _, row in omnibus.iterrows():
        if not bool(row["significant_Bonferroni"]):
            continue

        city = row["city"]
        metric = row["metric"]

        subset = alpha.loc[
            (alpha["city"] == city) & alpha[metric].notna(),
            ["location", metric],
        ].copy()

        dunn = sp.posthoc_dunn(
            subset,
            val_col=metric,
            group_col="location",
            p_adjust="bonferroni",
        ).reindex(index=LOCATIONS, columns=LOCATIONS)

        for loc1, loc2 in itertools.combinations(LOCATIONS, 2):
            if loc1 not in dunn.index or loc2 not in dunn.columns:
                continue
            value = dunn.loc[loc1, loc2]
            if pd.isna(value):
                continue

            posthoc_rows.append(
                {
                    "city": city,
                    "metric": metric,
                    "location_1": loc1,
                    "location_2": loc2,
                    "p_Bonferroni": float(value),
                    "significant_Bonferroni": bool(value < ALPHA),
                }
            )

    return omnibus, pd.DataFrame(posthoc_rows)


def alpha_descriptives(alpha: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR by city, location, and metric."""
    rows = []

    for city, location, metric in itertools.product(
        CITIES, LOCATIONS, METRICS
    ):
        values = alpha.loc[
            (alpha["city"] == city) & (alpha["location"] == location),
            metric,
        ].dropna()

        rows.append(
            {
                "city": city,
                "location": location,
                "metric": metric,
                "n": len(values),
                "median": values.median() if len(values) else np.nan,
                "q1": values.quantile(0.25) if len(values) else np.nan,
                "q3": values.quantile(0.75) if len(values) else np.nan,
            }
        )

    result = pd.DataFrame(rows)
    result["IQR"] = result["q3"] - result["q1"]
    return result


# =============================================================================
# Beta diversity
# =============================================================================

def build_species_matrix(
    abund: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample-by-species matrix and matching metadata."""
    matrix = abund.pivot_table(
        index="sample_ID",
        columns="species",
        values="RPKMF_norm",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    matrix = matrix.reindex(sorted(matrix.columns), axis=1)

    metadata = (
        abund[["sample_ID", "city", "location"]]
        .drop_duplicates("sample_ID")
        .set_index("sample_ID")
        .reindex(matrix.index)
    )

    if metadata.isna().any().any():
        raise ValueError("Metadata could not be matched to every sample.")

    keep_samples = matrix.sum(axis=1) > 0
    if (~keep_samples).any():
        removed = matrix.index[~keep_samples].tolist()
        warnings.warn(
            "Removing samples with zero total species abundance: "
            + ", ".join(removed)
        )
        matrix = matrix.loc[keep_samples]
        metadata = metadata.loc[matrix.index]

    matrix = matrix.loc[:, matrix.sum(axis=0) > 0]

    return matrix, metadata


def make_distance_matrix(
    values: np.ndarray,
    sample_ids: list[str],
    metric: str,
) -> DistanceMatrix:
    """Calculate a validated scikit-bio distance matrix."""
    array = squareform(pdist(values, metric=metric))
    np.fill_diagonal(array, 0.0)

    if not np.isfinite(array).all():
        raise ValueError(f"{metric} distance matrix contains invalid values.")

    return DistanceMatrix(array, ids=sample_ids)


def subset_distance_matrix(
    distance_matrix: DistanceMatrix,
    sample_ids: list[str],
) -> DistanceMatrix:
    """Return a distance matrix containing selected samples."""
    return distance_matrix.filter(sample_ids, strict=True)


def calculate_pcoa(
    distance_matrix: DistanceMatrix,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return the first two PCoA axes and explained proportions."""
    ordination = pcoa(
        distance_matrix,
        method="eigh",
        dimensions=2,
        warn_neg_eigval=0.01,
    )

    coords = ordination.samples.iloc[:, :2].copy()
    coords.columns = ["PCoA1", "PCoA2"]

    variance = {
        "PCoA1": float(ordination.proportion_explained.iloc[0]),
        "PCoA2": float(ordination.proportion_explained.iloc[1]),
    }
    return coords, variance


def run_permutation_test(
    test_function,
    distance_matrix: DistanceMatrix,
    grouping: pd.Series,
) -> dict:
    """Run PERMANOVA or median-based PERMDISP."""
    kwargs = {
        "distmat": distance_matrix,
        "grouping": grouping,
        "permutations": N_PERMUTATIONS,
        "seed": RANDOM_SEED,
    }

    if test_function is permdisp:
        kwargs["test"] = "median"

    result = test_function(**kwargs)

    return {
        "statistic": float(result["test statistic"]),
        "p_raw": float(result["p-value"]),
        "permutations": int(result["number of permutations"]),
        "sample_size": int(result["sample size"]),
        "number_of_groups": int(result["number of groups"]),
    }


def run_beta_statistics(
    distance_matrices: dict[str, DistanceMatrix],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run omnibus and pairwise PERMANOVA/PERMDISP.

    Omnibus p-values are Bonferroni-corrected separately for PERMANOVA and
    PERMDISP across 2 distances × 3 scopes = 6 tests per test type.

    Pairwise tests are run only when the corresponding Bonferroni-corrected
    omnibus test is significant. Pairwise p-values are Bonferroni-corrected
    within each scope × distance family.
    """
    scopes = {"all_samples": metadata}
    scopes.update(
        {
            city: metadata.loc[
                metadata["city"].astype(str) == city
            ].copy()
            for city in CITIES
        }
    )

    omnibus_rows = []

    for scope, scope_metadata in scopes.items():
        ids = scope_metadata.index.astype(str).tolist()
        grouping = scope_metadata.loc[ids, "location"].astype(str)

        if grouping.nunique() < 2:
            continue

        for distance_name, full_dm in distance_matrices.items():
            dm = (
                full_dm
                if scope == "all_samples"
                else subset_distance_matrix(full_dm, ids)
            )

            for test_name, test_function in {
                "PERMANOVA": permanova,
                "PERMDISP": permdisp,
            }.items():
                omnibus_rows.append(
                    {
                        "test": test_name,
                        "scope": scope,
                        "distance": distance_name,
                        "grouping_variable": "location",
                        **run_permutation_test(
                            test_function,
                            dm,
                            grouping,
                        ),
                    }
                )

    omnibus = bonferroni_adjust(
        pd.DataFrame(omnibus_rows),
        family_columns=["test"],
    )

    pairwise = {"PERMANOVA": [], "PERMDISP": []}

    for _, omnibus_row in omnibus.iterrows():
        # Corrected gating: no post-hoc testing unless the adjusted
        # omnibus result is significant.
        if not bool(omnibus_row["significant_Bonferroni"]):
            continue

        test_name = omnibus_row["test"]
        test_function = (
            permanova if test_name == "PERMANOVA" else permdisp
        )
        scope = omnibus_row["scope"]
        distance_name = omnibus_row["distance"]

        scope_metadata = scopes[scope]
        ids = scope_metadata.index.astype(str).tolist()
        full_dm = distance_matrices[distance_name]
        dm = (
            full_dm
            if scope == "all_samples"
            else subset_distance_matrix(full_dm, ids)
        )

        available_locations = [
            location
            for location in LOCATIONS
            if location in scope_metadata["location"].astype(str).unique()
        ]

        for loc1, loc2 in itertools.combinations(available_locations, 2):
            pair_metadata = scope_metadata.loc[
                scope_metadata["location"].astype(str).isin([loc1, loc2])
            ].copy()

            counts = pair_metadata["location"].astype(str).value_counts()
            if counts.get(loc1, 0) < 2 or counts.get(loc2, 0) < 2:
                warnings.warn(
                    f"Skipping {test_name}: {scope}, {distance_name}, "
                    f"{loc1} vs {loc2}; fewer than two samples in one group."
                )
                continue

            pair_ids = pair_metadata.index.astype(str).tolist()
            pair_dm = subset_distance_matrix(dm, pair_ids)
            grouping = pair_metadata.loc[pair_ids, "location"].astype(str)

            pairwise[test_name].append(
                {
                    "test": test_name,
                    "scope": scope,
                    "distance": distance_name,
                    "location_1": loc1,
                    "location_2": loc2,
                    "n_location_1": int(counts[loc1]),
                    "n_location_2": int(counts[loc2]),
                    **run_permutation_test(
                        test_function,
                        pair_dm,
                        grouping,
                    ),
                }
            )

    permanova_pairwise = bonferroni_adjust(
        pd.DataFrame(pairwise["PERMANOVA"]),
        family_columns=["scope", "distance"],
    )

    permdisp_pairwise = bonferroni_adjust(
        pd.DataFrame(pairwise["PERMDISP"]),
        family_columns=["scope", "distance"],
    )

    return omnibus, permanova_pairwise, permdisp_pairwise


def geometric_median(
    points: np.ndarray,
    tolerance: float = 1e-7,
    max_iterations: int = 1000,
) -> np.ndarray:
    """Calculate the spatial median using Weiszfeld's algorithm."""
    points = np.asarray(points, dtype=float)
    estimate = points.mean(axis=0)

    for _ in range(max_iterations):
        distances = np.linalg.norm(points - estimate, axis=1)

        if np.any(distances < tolerance):
            return points[np.argmin(distances)]

        updated = np.average(
            points,
            axis=0,
            weights=1.0 / distances,
        )

        if np.linalg.norm(updated - estimate) < tolerance:
            return updated

        estimate = updated

    warnings.warn("Spatial-median calculation did not converge.")
    return estimate


def calculate_dispersion(
    distance_matrix: DistanceMatrix,
    metadata: pd.DataFrame,
    scope: str,
    distance_name: str,
) -> pd.DataFrame:
    """Calculate distance of each sample to its location spatial median."""
    ordination = pcoa(
        distance_matrix,
        method="eigh",
        warn_neg_eigval=0.01,
    )

    positive_axes = ordination.eigvals[
        ordination.eigvals > 0
    ].index

    if len(positive_axes) == 0:
        raise ValueError(
            f"No positive PCoA axes for {scope}, {distance_name}."
        )

    coords = ordination.samples.loc[:, positive_axes]
    rows = []

    for location in LOCATIONS:
        ids = metadata.index[
            metadata["location"].astype(str) == location
        ].astype(str).tolist()

        if not ids:
            continue

        points = coords.loc[ids].to_numpy(dtype=float)
        median = geometric_median(points)
        distances = np.linalg.norm(points - median, axis=1)

        for sample_id, value in zip(ids, distances):
            rows.append(
                {
                    "sample_ID": sample_id,
                    "scope": scope,
                    "city": scope if scope in CITIES else "all",
                    "location": location,
                    "distance": distance_name,
                    "distance_to_spatial_median": float(value),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================

def plot_alpha(alpha: pd.DataFrame) -> None:
    """Save one four-panel alpha-diversity figure per city."""
    legend_handles = [
        Patch(
            facecolor=PALETTE[location],
            edgecolor="black",
            label=location,
        )
        for location in LOCATIONS
    ]

    for city in CITIES:
        city_data = alpha.loc[alpha["city"] == city].copy()
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

        for ax, (metric, label) in zip(axes, METRICS.items()):
            sns.boxplot(
                data=city_data,
                x="location",
                y=metric,
                order=LOCATIONS,
                hue="location",
                hue_order=LOCATIONS,
                palette=PALETTE,
                showfliers=False,
                width=0.65,
                linewidth=1.2,
                legend=False,
                ax=ax,
            )
            sns.stripplot(
                data=city_data,
                x="location",
                y=metric,
                order=LOCATIONS,
                color="black",
                size=4,
                jitter=0.18,
                alpha=0.65,
                ax=ax,
            )
            ax.set(xlabel="", ylabel=label)
            ax.grid(False)
            sns.despine(ax=ax)

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(LOCATIONS),
            frameon=False,
        )
        fig.subplots_adjust(
            top=0.82,
            bottom=0.18,
            left=0.06,
            right=0.99,
            wspace=0.42,
        )
        fig.savefig(
            OUTPUT_DIR / f"Figure2_{city}_alpha.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def plot_pcoa(
    coordinates: pd.DataFrame,
    variance: dict[str, dict[str, float]],
) -> None:
    """Save the two-panel all-sample PCoA figure."""
    location_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PALETTE[location],
            markeredgecolor="black",
            markersize=8,
            label=location,
        )
        for location in LOCATIONS
    ]

    city_handles = [
        Line2D(
            [0],
            [0],
            marker=CITY_MARKERS[city],
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            label=CITY_LABELS[city],
        )
        for city in CITIES
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))

    for ax, distance_name in zip(axes, DISTANCES):
        plot_data = coordinates.loc[
            coordinates["distance"] == distance_name
        ]

        for city, location in itertools.product(CITIES, LOCATIONS):
            subset = plot_data.loc[
                (plot_data["city"].astype(str) == city)
                & (plot_data["location"].astype(str) == location)
            ]

            if subset.empty:
                continue

            ax.scatter(
                subset["PCoA1"],
                subset["PCoA2"],
                s=65,
                color=PALETTE[location],
                marker=CITY_MARKERS[city],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.85,
            )

        v = variance[distance_name]
        ax.axhline(0, color="0.80", linewidth=0.8, zorder=0)
        ax.axvline(0, color="0.80", linewidth=0.8, zorder=0)
        ax.set_xlabel(
            f"{distance_name} PCoA1 ({v['PCoA1'] * 100:.1f}%)"
        )
        ax.set_ylabel(
            f"{distance_name} PCoA2 ({v['PCoA2'] * 100:.1f}%)"
        )
        ax.grid(False)
        sns.despine(ax=ax)

    location_legend = fig.legend(
        handles=location_handles,
        loc="upper center",
        bbox_to_anchor=(0.38, 1.03),
        ncol=len(LOCATIONS),
        frameon=False,
    )
    fig.add_artist(location_legend)

    fig.legend(
        handles=city_handles,
        loc="upper center",
        bbox_to_anchor=(0.79, 1.03),
        ncol=len(CITIES),
        frameon=False,
    )

    fig.subplots_adjust(
        top=0.80,
        bottom=0.16,
        left=0.09,
        right=0.98,
        wspace=0.32,
    )
    fig.savefig(
        OUTPUT_DIR / "Figure2_PCoA.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_dispersion(dispersion: pd.DataFrame) -> None:
    """Save one two-panel PERMDISP-distance figure per city."""
    handles = [
        Patch(
            facecolor=PALETTE[location],
            edgecolor="black",
            label=location,
        )
        for location in LOCATIONS
    ]

    for city in CITIES:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))

        for ax, distance_name in zip(axes, DISTANCES):
            plot_data = dispersion.loc[
                (dispersion["scope"] == city)
                & (dispersion["distance"] == distance_name)
            ]

            sns.boxplot(
                data=plot_data,
                x="location",
                y="distance_to_spatial_median",
                order=LOCATIONS,
                hue="location",
                hue_order=LOCATIONS,
                palette=PALETTE,
                showfliers=False,
                width=0.65,
                linewidth=1.2,
                legend=False,
                ax=ax,
            )
            sns.stripplot(
                data=plot_data,
                x="location",
                y="distance_to_spatial_median",
                order=LOCATIONS,
                color="black",
                size=4,
                jitter=0.18,
                alpha=0.65,
                ax=ax,
            )
            ax.set_xlabel("")
            ax.set_ylabel(f"{distance_name} distance to spatial median")
            ax.grid(False)
            sns.despine(ax=ax)

        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(LOCATIONS),
            frameon=False,
        )
        fig.subplots_adjust(
            top=0.82,
            bottom=0.16,
            left=0.09,
            right=0.98,
            wspace=0.32,
        )
        fig.savefig(
            OUTPUT_DIR / f"Figure2_{city}_PERMDISP.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


# =============================================================================
# Main analysis
# =============================================================================

def main() -> None:
    configure_plotting()

    df = read_data(INPUT_FILE)
    abund = aggregate_species(df)

    # Alpha diversity
    alpha = calculate_alpha_metrics(abund)
    alpha_omnibus, alpha_posthoc = run_alpha_statistics(alpha)
    alpha_summary = alpha_descriptives(alpha)

    alpha.to_csv(
        OUTPUT_DIR / "Figure2_alpha_diversity.tsv",
        sep="\t",
        index=False,
    )
    alpha_omnibus.to_csv(
        OUTPUT_DIR / "Figure2_alpha_Kruskal_Wallis_Bonferroni.tsv",
        sep="\t",
        index=False,
    )
    alpha_posthoc.to_csv(
        OUTPUT_DIR / "Figure2_alpha_Dunn_Bonferroni.tsv",
        sep="\t",
        index=False,
    )
    alpha_summary.to_csv(
        OUTPUT_DIR / "Figure2_alpha_descriptive_statistics.tsv",
        sep="\t",
        index=False,
    )

    # Beta diversity matrices
    species_matrix, metadata = build_species_matrix(abund)
    ids = species_matrix.index.astype(str).tolist()
    abundance_values = species_matrix.to_numpy(dtype=float)
    presence_absence = abundance_values > PRESENCE_THRESHOLD

    distance_matrices = {
        "Bray–Curtis": make_distance_matrix(
            abundance_values,
            ids,
            DISTANCES["Bray–Curtis"],
        ),
        "Jaccard": make_distance_matrix(
            presence_absence,
            ids,
            DISTANCES["Jaccard"],
        ),
    }

    # Save distance matrices and calculate PCoA
    coordinate_tables = []
    variance = {}

    for distance_name, dm in distance_matrices.items():
        file_label = distance_name.replace("–", "_").replace(" ", "_")
        pd.DataFrame(
            dm.data,
            index=dm.ids,
            columns=dm.ids,
        ).to_csv(
            OUTPUT_DIR / f"Figure2_{file_label}_distance_matrix.tsv",
            sep="\t",
        )

        coords, variance[distance_name] = calculate_pcoa(dm)
        coords["sample_ID"] = coords.index
        coords["distance"] = distance_name
        coords = coords.merge(
            metadata.reset_index(),
            on="sample_ID",
            how="left",
        )
        coordinate_tables.append(coords)

    coordinates = pd.concat(coordinate_tables, ignore_index=True)
    coordinates.to_csv(
        OUTPUT_DIR / "Figure2_PCoA_coordinates.tsv",
        sep="\t",
        index=False,
    )

    # Beta-diversity statistics
    beta_omnibus, pairwise_permanova, pairwise_permdisp = (
        run_beta_statistics(distance_matrices, metadata)
    )

    beta_omnibus.to_csv(
        OUTPUT_DIR / "Figure2_PERMANOVA_PERMDISP_Bonferroni.tsv",
        sep="\t",
        index=False,
    )
    pairwise_permanova.to_csv(
        OUTPUT_DIR / "Figure2_pairwise_PERMANOVA_Bonferroni.tsv",
        sep="\t",
        index=False,
    )
    pairwise_permdisp.to_csv(
        OUTPUT_DIR / "Figure2_pairwise_PERMDISP_Bonferroni.tsv",
        sep="\t",
        index=False,
    )

    # Distances to location spatial medians
    dispersion_tables = []

    scopes = {"all_samples": metadata}
    scopes.update(
        {
            city: metadata.loc[
                metadata["city"].astype(str) == city
            ].copy()
            for city in CITIES
        }
    )

    for scope, scope_metadata in scopes.items():
        scope_ids = scope_metadata.index.astype(str).tolist()

        for distance_name, full_dm in distance_matrices.items():
            dm = (
                full_dm
                if scope == "all_samples"
                else subset_distance_matrix(full_dm, scope_ids)
            )
            dispersion_tables.append(
                calculate_dispersion(
                    dm,
                    scope_metadata,
                    scope,
                    distance_name,
                )
            )

    dispersion = pd.concat(dispersion_tables, ignore_index=True)
    dispersion.to_csv(
        OUTPUT_DIR / "Figure2_distances_to_spatial_median.tsv",
        sep="\t",
        index=False,
    )

    # Figures
    plot_alpha(alpha)
    plot_pcoa(coordinates, variance)
    plot_dispersion(dispersion)

    print(f"Figure 2 analysis complete. Outputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
