#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:50:23 2026

@author: jaanajurvansuuCreate all Figure 4 analyses and save each panel as a separate PNG.

Panels
------
A. Group-level viral accession rank–abundance curves.
B. Occupancy-frequency distributions and separate statistical TSV outputs.
C-D. Four-set Venn diagrams for Tampere and Kuopio, respectively.
E. Viral prevalence heatmap with clustering of city-location columns only.
F. Combined city-agreement UH–WWTP and UH–SA occupancy effects.

Normalised RPKMF > 0 defines detection. Statistical methods for panel B are
preserved from the original analysis. SA1 and SA2 remain separate for the Venn
diagrams and heatmap, but are pooled into SA for UH–SA effect calculations.
Only PNG figures and TSV tables are written; no combined figure or PDF is made.
"""

from __future__ import annotations

import re
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2_contingency

# Configuration

CITIES = ["TRE", "KUO"]
CITY_LABELS = {
    "TRE": "Tampere",
    "KUO": "Kuopio",
}

LOCATIONS = ["UH", "WWTP", "SA1", "SA2"]
ENVIRONMENTS = ["UH", "WWTP", "SA"]

GROUPS = [
    f"{city}_{location}"
    for city in CITIES
    for location in LOCATIONS
]

STRATA = [
    "TRE_UH",
    "KUO_UH",
    "TRE_WWTP",
    "KUO_WWTP",
    "TRE_SA1",
    "KUO_SA1",
    "TRE_SA2",
    "KUO_SA2",
]

PALETTE = {
    "UH": "#E67E22",
    "WWTP": "#2E86C1",
    "SA1": "#52BE80",
    "SA2": "#239B56",
}

ENVIRONMENT_PALETTE = {
    "UH": PALETTE["UH"],
    "WWTP": PALETTE["WWTP"],
    "SA": PALETTE["SA1"],
}

CITY_LINESTYLES = {
    "TRE": "-",
    "KUO": "--",
}

BIN_LABELS = [
    ">0–20%",
    ">20–40%",
    ">40–60%",
    ">60–80%",
    "80–100%",
]

BIN_MIDPOINTS = np.array([0.10, 0.30, 0.50, 0.70, 0.90])

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "EsVirutu_abundance_metadata.tsv"
OUTPUT_DIR = ROOT

DETECTION_THRESHOLD = 0.0
N_PERMUTATIONS = 9999
RANDOM_SEED = 20260717
DPI = 300
TOP_N_POSITIVE = 12
TOP_N_NEGATIVE = 12
REMOVE_GLOBAL_SINGLETONS = True

# Plotting style

def configure_plotting() -> None:
    """Apply the manuscript-wide plotting style."""

    sns.set_theme(style="ticks", context="notebook")

    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["axes.titlesize"] = 14
    mpl.rcParams["legend.fontsize"] = 14
    mpl.rcParams["axes.linewidth"] = 1.2
    mpl.rcParams["xtick.major.width"] = 1.2
    mpl.rcParams["ytick.major.width"] = 1.2
    mpl.rcParams["xtick.major.size"] = 5
    mpl.rcParams["ytick.major.size"] = 5
    mpl.rcParams["savefig.facecolor"] = "white"

# General helpers

def shorten_species_name(name: str, maximum_length: int = 52) -> str:
    """Shorten a long species name while retaining both ends."""

    text = str(name)

    if len(text) <= maximum_length:
        return text

    left = maximum_length // 2 - 2
    right = maximum_length - left - 3
    return text[:left] + "..." + text[-right:]

def clean_species_name(name: str) -> str:
    """Remove a leading species taxonomy prefix for plot labels only."""

    return re.sub(r"^s__", "", str(name).strip())

def assign_occupancy_bin(occupancy: pd.Series) -> pd.Categorical:
    """
    Assign occupancy classes.

    The 80% boundary is included in the core class to match the targeted
    core-occupancy analysis.
    """

    values = occupancy.to_numpy(dtype=float)
    labels = np.full(len(values), None, dtype=object)

    labels[(values > 0.00) & (values <= 0.20)] = BIN_LABELS[0]
    labels[(values > 0.20) & (values <= 0.40)] = BIN_LABELS[1]
    labels[(values > 0.40) & (values <= 0.60)] = BIN_LABELS[2]
    labels[(values > 0.60) & (values < 0.80)] = BIN_LABELS[3]
    labels[(values >= 0.80) & (values <= 1.00)] = BIN_LABELS[4]

    if np.any(pd.isna(labels)):
        invalid = values[pd.isna(labels)]
        raise ValueError(
            "Some occupancy values could not be assigned to a bin: "
            f"{invalid[:10]}"
        )

    return pd.Categorical(
        labels,
        categories=BIN_LABELS,
        ordered=True,
    )

def save_tsv(dataframe: pd.DataFrame, path: Path) -> None:
    """Save a table as a tab-separated file."""

    dataframe.to_csv(path, sep="\t", index=False)

# Load and prepare data

def load_data(
    input_file: str | Path,
    detection_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the input table and prepare abundance and detection tables.

    Returns
    -------
    filtered_data
        Filtered Homo-associated rows with city and location metadata.
    sample_metadata
        Unique sample, city and location combinations.
    sample_species
        Sample-level species abundance and presence.
    """

    data = pd.read_csv(
        input_file,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    required_columns = {
        "sample_ID",
        "host_genus",
        "species",
        "Accession",
        "RPKMF_norm",
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data["RPKMF_norm"] = pd.to_numeric(
        data["RPKMF_norm"],
        errors="coerce",
    )

    invalid_count = int(data["RPKMF_norm"].isna().sum())

    if invalid_count:
        warnings.warn(
            f"{invalid_count} non-numeric RPKMF_norm values were "
            "converted to zero."
        )

    data["RPKMF_norm"] = data["RPKMF_norm"].fillna(0.0)

    if (data["RPKMF_norm"] < 0).any():
        raise ValueError("Negative RPKMF_norm values were detected.")

    data = data.loc[
        data["host_genus"].str.strip().str.casefold().eq("homo")
    ].copy()

    data["species"] = data["species"].str.strip()
    data["Accession"] = data["Accession"].str.strip()

    blank_species = int(data["species"].eq("").sum())
    blank_accessions = int(data["Accession"].eq("").sum())

    if blank_species:
        warnings.warn(
            f"Removing {blank_species} rows with blank species names."
        )

    if blank_accessions:
        warnings.warn(
            f"Removing {blank_accessions} rows with blank accession IDs."
        )

    data = data.loc[
        data["species"].ne("") & data["Accession"].ne("")
    ].copy()

    if data.empty:
        raise ValueError(
            "No Homo-associated records remained after filtering."
        )

    sample_parts = data["sample_ID"].str.split("_", expand=True)

    if sample_parts.shape[1] < 4:
        raise ValueError(
            "sample_ID must contain at least four underscore-separated "
            "components, with city in component 3 and location in component 4."
        )

    data["city"] = sample_parts[2].str.strip()
    data["location"] = sample_parts[3].str.strip()

    data = data.loc[
        data["city"].isin(CITIES)
        & data["location"].isin(LOCATIONS)
    ].copy()

    if data.empty:
        raise ValueError(
            "No records matched the expected cities and locations."
        )

    data["group"] = data["city"] + "_" + data["location"]
    data["environment"] = data["location"].replace(
        {"SA1": "SA", "SA2": "SA"}
    )
    data["stratum"] = data["city"] + "_" + data["location"]

    sample_metadata = (
        data[
            [
                "sample_ID",
                "city",
                "location",
                "group",
                "environment",
                "stratum",
            ]
        ]
        .drop_duplicates()
        .sort_values(["city", "location", "sample_ID"])
        .reset_index(drop=True)
    )

    sample_species = (
        data.groupby(
            [
                "sample_ID",
                "city",
                "location",
                "group",
                "environment",
                "species",
            ],
            as_index=False,
            observed=True,
        )
        .agg(RPKMF_norm=("RPKMF_norm", "sum"))
    )

    sample_species["present"] = (
        sample_species["RPKMF_norm"] > detection_threshold
    )

    return data, sample_metadata, sample_species

# Complete sample-by-feature matrices

def complete_sample_feature_table(
    observed: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    feature_column: str,
    value_column: str,
    fill_value: float | bool,
) -> pd.DataFrame:
    """Create a complete sample-by-feature table with explicit absences."""

    features = pd.Index(
        sorted(observed[feature_column].unique()),
        name=feature_column,
    )

    samples = pd.Index(
        sample_metadata["sample_ID"].drop_duplicates(),
        name="sample_ID",
    )

    complete_index = pd.MultiIndex.from_product(
        [samples, features],
        names=["sample_ID", feature_column],
    )

    complete = (
        observed[["sample_ID", feature_column, value_column]]
        .drop_duplicates(["sample_ID", feature_column])
        .set_index(["sample_ID", feature_column])
        .reindex(complete_index, fill_value=fill_value)
        .reset_index()
        .merge(
            sample_metadata,
            on="sample_ID",
            how="left",
            validate="many_to_one",
        )
    )

    return complete

# Panel A: rank–abundance analysis

def calculate_rank_abundance(
    data: pd.DataFrame,
    sample_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate accession rank–abundance profiles for all eight groups."""

    sample_accession = (
        data.groupby(
            ["sample_ID", "Accession"],
            as_index=False,
            observed=True,
        )
        .agg(RPKMF_norm=("RPKMF_norm", "sum"))
    )

    complete_abundance = complete_sample_feature_table(
        observed=sample_accession,
        sample_metadata=sample_metadata,
        feature_column="Accession",
        value_column="RPKMF_norm",
        fill_value=0.0,
    )

    group_abundance = (
        complete_abundance.groupby(
            ["city", "location", "group", "Accession"],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_RPKMF_norm=("RPKMF_norm", "mean"),
            prevalence_n=("RPKMF_norm", lambda x: int((x > 0).sum())),
            sample_n=("sample_ID", "nunique"),
        )
    )

    group_abundance["prevalence"] = (
        group_abundance["prevalence_n"]
        / group_abundance["sample_n"]
    )

    rank_abundance = group_abundance.loc[
        group_abundance["mean_RPKMF_norm"] > 0
    ].copy()

    if rank_abundance.empty:
        raise ValueError(
            "No accessions had positive mean abundance."
        )

    group_totals = rank_abundance.groupby(
        "group",
        observed=True,
    )["mean_RPKMF_norm"].transform("sum")

    rank_abundance["relative_abundance"] = (
        rank_abundance["mean_RPKMF_norm"] / group_totals
    )

    rank_abundance["relative_abundance_percent"] = (
        100.0 * rank_abundance["relative_abundance"]
    )

    rank_abundance = rank_abundance.sort_values(
        ["group", "relative_abundance", "Accession"],
        ascending=[True, False, True],
    )

    rank_abundance["rank"] = (
        rank_abundance.groupby("group", observed=True)
        .cumcount()
        .add(1)
    )

    rank_abundance["group"] = pd.Categorical(
        rank_abundance["group"],
        categories=GROUPS,
        ordered=True,
    )

    return (
        rank_abundance.sort_values(["group", "rank"])
        .reset_index(drop=True)
    )

def plot_rank_abundance(
    ax: plt.Axes,
    rank_abundance: pd.DataFrame,
    include_legend: bool = True,
) -> None:
    """Plot panel A."""

    for city in CITIES:
        for location in LOCATIONS:
            group = f"{city}_{location}"

            subset = rank_abundance.loc[
                rank_abundance["group"] == group
            ]

            if subset.empty:
                warnings.warn(
                    f"No positive accession abundances found for {group}."
                )
                continue

            ax.plot(
                subset["rank"],
                subset["relative_abundance_percent"],
                color=PALETTE[location],
                linestyle=CITY_LINESTYLES[city],
                linewidth=2.2,
                alpha=0.95,
            )

    ax.set_yscale("log")
    ax.set_xlabel("Viral accession rank")
    ax.set_ylabel("Mean relative abundance (%)")
    ax.grid(False)
    sns.despine(ax=ax)

    if include_legend:
        location_handles = [
            Line2D(
                [0],
                [0],
                color=PALETTE[location],
                linewidth=2.4,
                label=location,
            )
            for location in LOCATIONS
        ]

        city_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=CITY_LINESTYLES[city],
                linewidth=2.2,
                label=CITY_LABELS[city],
            )
            for city in CITIES
        ]

        legend_locations = ax.legend(
            handles=location_handles,
            ncol=4,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.12),
            handlelength=2.0,
            columnspacing=1.1,
        )
        ax.add_artist(legend_locations)

        ax.legend(
            handles=city_handles,
            ncol=2,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            handlelength=2.4,
            columnspacing=1.6,
        )

# Panel B: occupancy-frequency analysis

def calculate_city_location_occupancy(
    sample_metadata: pd.DataFrame,
    sample_species: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate species occupancy separately within each city-location group."""

    detected = sample_species.loc[
        sample_species["present"],
        ["sample_ID", "city", "location", "species"],
    ].copy()

    sample_counts = (
        sample_metadata.groupby(
            ["city", "location"],
            observed=True,
        )["sample_ID"]
        .nunique()
        .rename("total_samples")
        .reset_index()
    )

    occupancy = (
        detected.groupby(
            ["city", "location", "species"],
            as_index=False,
            observed=True,
        )
        .agg(positive_samples=("sample_ID", "nunique"))
        .merge(
            sample_counts,
            on=["city", "location"],
            how="left",
            validate="many_to_one",
        )
    )

    occupancy["occupancy_proportion"] = (
        occupancy["positive_samples"]
        / occupancy["total_samples"]
    )
    occupancy["occupancy_percent"] = (
        100.0 * occupancy["occupancy_proportion"]
    )
    occupancy["low_occupancy"] = (
        (occupancy["occupancy_proportion"] > 0.0)
        & (occupancy["occupancy_proportion"] <= 0.20)
    )
    occupancy["core_occupancy"] = (
        occupancy["occupancy_proportion"] >= 0.80
    )
    occupancy["plot_group"] = occupancy["location"].replace(
        {"SA1": "SA", "SA2": "SA"}
    )
    occupancy["stat_group"] = occupancy["plot_group"]
    occupancy["occupancy_bin"] = assign_occupancy_bin(
        occupancy["occupancy_proportion"]
    )

    return occupancy.sort_values(
        [
            "plot_group",
            "city",
            "location",
            "occupancy_proportion",
            "species",
        ]
    ).reset_index(drop=True)

def calculate_occupancy_frequency(
    occupancy: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the percentage of species in each occupancy bin."""

    counts = (
        occupancy.groupby(
            ["plot_group", "occupancy_bin"],
            observed=False,
        )
        .size()
        .rename("n_viruses")
        .reset_index()
    )

    totals = (
        occupancy.groupby(
            "plot_group",
            observed=True,
        )
        .size()
        .rename("total_viruses")
        .reset_index()
    )

    frequency = counts.merge(
        totals,
        on="plot_group",
        how="left",
        validate="many_to_one",
    )

    frequency["virus_percent"] = (
        100.0
        * frequency["n_viruses"]
        / frequency["total_viruses"]
    )

    midpoint_map = dict(zip(BIN_LABELS, BIN_MIDPOINTS))

    frequency["bin_midpoint"] = (
        frequency["occupancy_bin"]
        .astype(str)
        .map(midpoint_map)
        .astype(float)
    )

    return frequency.sort_values(
        ["plot_group", "bin_midpoint"]
    ).reset_index(drop=True)

def event_summary(
    occupancy: pd.DataFrame,
    event_column: str,
) -> pd.DataFrame:
    """Summarise low- or core-occupancy species proportions."""

    rows = []

    for group in ENVIRONMENTS:
        subset = occupancy.loc[
            occupancy["stat_group"].eq(group)
        ]

        total = len(subset)
        event_count = int(subset[event_column].sum())

        rows.append(
            {
                "group": group,
                "event": event_column,
                "total_detected_viruses": total,
                "event_viruses": event_count,
                "event_percent": (
                    100.0 * event_count / total
                    if total > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)

def plot_occupancy_frequency(
    ax: plt.Axes,
    frequency: pd.DataFrame,
    include_legend: bool = True,
) -> None:
    """Plot panel B."""

    for group in ENVIRONMENTS:
        subset = frequency.loc[
            frequency["plot_group"].eq(group)
        ].sort_values("bin_midpoint")

        ax.plot(
            subset["bin_midpoint"],
            subset["virus_percent"],
            color=ENVIRONMENT_PALETTE[group],
            linewidth=2.6,
            marker="o",
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=group,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(BIN_MIDPOINTS, BIN_LABELS)
    ax.tick_params(axis="x", labelrotation=25)
    ax.set_xlabel("Occupancy proportion")
    ax.set_ylabel("Viral species in occupancy bin (%)")
    ax.grid(False)
    sns.despine(ax=ax)

    if include_legend:
        ax.legend(
            ncol=3,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            handlelength=2.0,
            columnspacing=1.4,
        )

# Panel B statistics

def build_city_binary_matrix(
    sample_metadata: pd.DataFrame,
    sample_species: pd.DataFrame,
    city: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a binary sample-by-species matrix for one city."""

    city_samples = (
        sample_metadata.loc[
            sample_metadata["city"].eq(city)
        ]
        .sort_values(["location", "sample_ID"])
        .reset_index(drop=True)
    )

    detected = sample_species.loc[
        sample_species["present"]
        & sample_species["city"].eq(city),
        ["sample_ID", "species"],
    ]

    species = sorted(detected["species"].unique())

    sample_index = {
        sample_id: index
        for index, sample_id in enumerate(city_samples["sample_ID"])
    }
    species_index = {
        species_name: index
        for index, species_name in enumerate(species)
    }

    matrix = np.zeros(
        (len(city_samples), len(species)),
        dtype=np.int8,
    )

    for row in detected.itertuples(index=False):
        matrix[
            sample_index[row.sample_ID],
            species_index[row.species],
        ] = 1

    return city_samples, matrix

def site_event_counts(
    binary_matrix: np.ndarray,
    assigned_labels: np.ndarray,
    event: str,
) -> dict[str, tuple[int, int]]:
    """Return event and non-event counts for each original location."""

    result: dict[str, tuple[int, int]] = {}

    for location in LOCATIONS:
        sample_indices = np.where(
            assigned_labels == location
        )[0]

        n_samples = len(sample_indices)

        if n_samples == 0:
            result[location] = (0, 0)
            continue

        positive_counts = binary_matrix[
            sample_indices
        ].sum(axis=0)

        positive_counts = positive_counts[
            positive_counts > 0
        ]

        total_detected = len(positive_counts)
        proportions = positive_counts / n_samples

        if event == "low_occupancy":
            event_count = int(
                np.sum(
                    (proportions > 0.0)
                    & (proportions <= 0.20)
                )
            )
        elif event == "core_occupancy":
            event_count = int(
                np.sum(proportions >= 0.80)
            )
        else:
            raise ValueError(
                "event must be 'low_occupancy' or 'core_occupancy'"
            )

        result[location] = (
            event_count,
            total_detected - event_count,
        )

    return result

def collapse_site_counts_to_environments(
    site_counts: dict[str, tuple[int, int]],
) -> np.ndarray:
    """Pool SA1 and SA2 after site-specific event classification."""

    sa_event = (
        site_counts["SA1"][0]
        + site_counts["SA2"][0]
    )
    sa_non_event = (
        site_counts["SA1"][1]
        + site_counts["SA2"][1]
    )

    return np.asarray(
        [
            [site_counts["UH"][0], site_counts["UH"][1]],
            [site_counts["WWTP"][0], site_counts["WWTP"][1]],
            [sa_event, sa_non_event],
        ],
        dtype=int,
    )

def remove_empty_margins(table: np.ndarray) -> np.ndarray:
    """Remove zero-sum rows and columns from a contingency table."""

    reduced = np.asarray(table, dtype=int)
    reduced = reduced[reduced.sum(axis=1) > 0]
    reduced = reduced[:, reduced.sum(axis=0) > 0]
    return reduced

def chi_square_statistic(table: np.ndarray) -> float:
    """Return an uncorrected Pearson chi-square statistic."""

    reduced = remove_empty_margins(table)

    if reduced.shape[0] < 2 or reduced.shape[1] < 2:
        return np.nan

    statistic, _, _, _ = chi2_contingency(
        reduced,
        correction=False,
    )

    return float(statistic)

def cramers_v_bias_corrected(table: np.ndarray) -> float:
    """Calculate bias-corrected Cramér's V."""

    reduced = remove_empty_margins(table)

    if reduced.shape[0] < 2 or reduced.shape[1] < 2:
        return np.nan

    chi2, _, _, _ = chi2_contingency(
        reduced,
        correction=False,
    )

    n = reduced.sum()
    rows, columns = reduced.shape

    if n <= 1:
        return np.nan

    phi2 = chi2 / n
    phi2_corrected = max(
        0.0,
        phi2
        - ((columns - 1) * (rows - 1))
        / (n - 1),
    )
    rows_corrected = (
        rows
        - ((rows - 1) ** 2) / (n - 1)
    )
    columns_corrected = (
        columns
        - ((columns - 1) ** 2) / (n - 1)
    )
    denominator = min(
        rows_corrected - 1,
        columns_corrected - 1,
    )

    if denominator <= 0:
        return np.nan

    return float(
        np.sqrt(phi2_corrected / denominator)
    )

def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Adjust P values using the Holm step-down procedure."""

    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)

    running_max = 0.0
    m = len(p_values)

    for rank, index in enumerate(order):
        candidate = (m - rank) * p_values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(running_max, 1.0)

    return adjusted

def odds_ratio_with_ci(
    table_2x2: np.ndarray,
) -> tuple[float, float, float]:
    """Calculate an odds ratio and approximate 95% confidence interval."""

    table = np.asarray(table_2x2, dtype=float)

    if np.any(table == 0):
        table = table + 0.5

    a, b = table[0]
    c, d = table[1]

    odds_ratio = (a * d) / (b * c)

    standard_error = np.sqrt(
        1.0 / a
        + 1.0 / b
        + 1.0 / c
        + 1.0 / d
    )

    log_or = np.log(odds_ratio)

    lower = np.exp(
        log_or - 1.96 * standard_error
    )
    upper = np.exp(
        log_or + 1.96 * standard_error
    )

    return (
        float(odds_ratio),
        float(lower),
        float(upper),
    )

def observed_event_table(
    occupancy: pd.DataFrame,
    event_column: str,
) -> np.ndarray:
    """Return the observed UH, WWTP and pooled-SA event table."""

    rows = []

    for group in ENVIRONMENTS:
        subset = occupancy.loc[
            occupancy["stat_group"].eq(group)
        ]

        event_count = int(
            subset[event_column].sum()
        )
        total = len(subset)

        rows.append(
            [event_count, total - event_count]
        )

    return np.asarray(rows, dtype=int)

def overall_permutation_test(
    city_data: dict[str, tuple[pd.DataFrame, np.ndarray]],
    occupancy: pd.DataFrame,
    event: str,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int, float]:
    """Run an overall city-stratified sample-level permutation test."""

    observed_table = observed_event_table(
        occupancy=occupancy,
        event_column=event,
    )

    observed_chi2 = chi_square_statistic(observed_table)
    observed_v = cramers_v_bias_corrected(observed_table)

    reduced = remove_empty_margins(observed_table)

    if reduced.shape[0] < 2 or reduced.shape[1] < 2:
        return np.nan, np.nan, np.nan, 0, np.nan

    _, _, degrees_freedom, expected = chi2_contingency(
        reduced,
        correction=False,
    )

    permuted_statistics = np.empty(
        n_permutations,
        dtype=float,
    )

    for permutation_index in range(n_permutations):
        pooled_table = np.zeros(
            (len(ENVIRONMENTS), 2),
            dtype=int,
        )

        for city in CITIES:
            city_samples, binary_matrix = city_data[city]
            labels = city_samples["location"].to_numpy()
            permuted_labels = rng.permutation(labels)

            site_counts = site_event_counts(
                binary_matrix=binary_matrix,
                assigned_labels=permuted_labels,
                event=event,
            )

            pooled_table += collapse_site_counts_to_environments(
                site_counts
            )

        permuted_statistics[
            permutation_index
        ] = chi_square_statistic(pooled_table)

    permuted_statistics = permuted_statistics[
        np.isfinite(permuted_statistics)
    ]

    p_value = (
        (
            np.sum(
                permuted_statistics >= observed_chi2
            )
            + 1
        )
        / (len(permuted_statistics) + 1)
        if len(permuted_statistics) > 0
        else np.nan
    )

    return (
        observed_chi2,
        float(p_value),
        observed_v,
        int(degrees_freedom),
        float(expected.min()),
    )

def pairwise_permutation_test(
    city_data: dict[str, tuple[pd.DataFrame, np.ndarray]],
    occupancy: pd.DataFrame,
    event: str,
    group_1: str,
    group_2: str,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float]:
    """Run a pairwise city-stratified sample-level permutation test."""

    group_to_sites = {
        "UH": ["UH"],
        "WWTP": ["WWTP"],
        "SA": ["SA1", "SA2"],
    }

    selected_sites = (
        group_to_sites[group_1]
        + group_to_sites[group_2]
    )

    observed_full = observed_event_table(
        occupancy=occupancy,
        event_column=event,
    )

    index_map = {
        group: index
        for index, group in enumerate(ENVIRONMENTS)
    }

    observed_table = observed_full[
        [
            index_map[group_1],
            index_map[group_2],
        ]
    ]

    observed_chi2 = chi_square_statistic(observed_table)

    odds_ratio, lower, upper = odds_ratio_with_ci(
        observed_table
    )

    permuted_statistics = np.empty(
        n_permutations,
        dtype=float,
    )

    for permutation_index in range(n_permutations):
        pooled_pair = np.zeros((2, 2), dtype=int)

        for city in CITIES:
            city_samples, binary_matrix = city_data[city]

            selected_mask = city_samples[
                "location"
            ].isin(selected_sites).to_numpy()

            matrix_subset = binary_matrix[selected_mask]
            labels = city_samples.loc[
                selected_mask,
                "location",
            ].to_numpy()

            permuted_labels = rng.permutation(labels)

            site_counts = site_event_counts(
                binary_matrix=matrix_subset,
                assigned_labels=permuted_labels,
                event=event,
            )

            def collapsed_count(
                group: str,
            ) -> tuple[int, int]:
                sites = group_to_sites[group]

                return (
                    sum(site_counts[site][0] for site in sites),
                    sum(site_counts[site][1] for site in sites),
                )

            count_1 = collapsed_count(group_1)
            count_2 = collapsed_count(group_2)

            pooled_pair += np.asarray(
                [
                    [count_1[0], count_1[1]],
                    [count_2[0], count_2[1]],
                ],
                dtype=int,
            )

        permuted_statistics[
            permutation_index
        ] = chi_square_statistic(pooled_pair)

    permuted_statistics = permuted_statistics[
        np.isfinite(permuted_statistics)
    ]

    p_value = (
        (
            np.sum(
                permuted_statistics >= observed_chi2
            )
            + 1
        )
        / (len(permuted_statistics) + 1)
        if len(permuted_statistics) > 0
        else np.nan
    )

    return (
        observed_chi2,
        float(p_value),
        odds_ratio,
        lower,
        upper,
    )

def run_event_tests(
    sample_metadata: pd.DataFrame,
    sample_species: pd.DataFrame,
    occupancy: pd.DataFrame,
    event: str,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run overall and conditional pairwise tests for one event."""

    rng = np.random.default_rng(seed)

    city_data = {
        city: build_city_binary_matrix(
            sample_metadata=sample_metadata,
            sample_species=sample_species,
            city=city,
        )
        for city in CITIES
    }

    (
        overall_chi2,
        overall_p,
        overall_v,
        overall_df,
        minimum_expected,
    ) = overall_permutation_test(
        city_data=city_data,
        occupancy=occupancy,
        event=event,
        n_permutations=n_permutations,
        rng=rng,
    )

    overall = pd.DataFrame(
        {
            "event": [event],
            "test": [
                "City-stratified sample-level permutation chi-square"
            ],
            "groups": [",".join(ENVIRONMENTS)],
            "chi_square": [overall_chi2],
            "df": [overall_df],
            "permutation_p": [overall_p],
            "cramers_v_bias_corrected": [overall_v],
            "minimum_expected_count": [minimum_expected],
            "permutations": [n_permutations],
        }
    )

    pairwise_rows = []

    if np.isfinite(overall_p) and overall_p < 0.05:
        for group_1, group_2 in combinations(
            ENVIRONMENTS,
            2,
        ):
            (
                chi2_value,
                p_value,
                odds_ratio,
                lower,
                upper,
            ) = pairwise_permutation_test(
                city_data=city_data,
                occupancy=occupancy,
                event=event,
                group_1=group_1,
                group_2=group_2,
                n_permutations=n_permutations,
                rng=rng,
            )

            pairwise_rows.append(
                {
                    "event": event,
                    "group_1": group_1,
                    "group_2": group_2,
                    "chi_square": chi2_value,
                    "permutation_p_raw": p_value,
                    "odds_ratio_group1_vs_group2": odds_ratio,
                    "odds_ratio_ci95_lower": lower,
                    "odds_ratio_ci95_upper": upper,
                    "permutations": n_permutations,
                }
            )

    pairwise_columns = [
        "event",
        "group_1",
        "group_2",
        "chi_square",
        "permutation_p_raw",
        "permutation_p_holm",
        "odds_ratio_group1_vs_group2",
        "odds_ratio_ci95_lower",
        "odds_ratio_ci95_upper",
        "permutations",
    ]

    pairwise = pd.DataFrame(
        pairwise_rows,
        columns=[
            column
            for column in pairwise_columns
            if column != "permutation_p_holm"
        ],
    )

    if not pairwise.empty:
        pairwise["permutation_p_holm"] = holm_adjust(
            pairwise["permutation_p_raw"].to_numpy()
        )
        pairwise = pairwise.reindex(columns=pairwise_columns)
    else:
        pairwise = pd.DataFrame(columns=pairwise_columns)

    return overall, pairwise

# Panel C: separate city-specific four-set Venn diagrams

def calculate_city_location_species_sets(
    sample_species: pd.DataFrame,
) -> dict[str, dict[str, set[str]]]:
    """Return detected species sets for every city and location."""

    detected = sample_species.loc[
        sample_species["present"]
    ]

    result: dict[str, dict[str, set[str]]] = {}

    for city in CITIES:
        result[city] = {}

        for location in LOCATIONS:
            species = set(
                detected.loc[
                    detected["city"].eq(city)
                    & detected["location"].eq(location),
                    "species",
                ]
            )
            result[city][location] = species

    return result

def exclusive_venn_counts(
    sets_by_location: dict[str, set[str]],
) -> dict[str, int]:
    """
    Count species in each exclusive four-set membership combination.

    Keys are four-character binary strings in UH, WWTP, SA1, SA2 order.
    """

    universe = set().union(
        *(sets_by_location[location] for location in LOCATIONS)
    )

    counts = {
        f"{value:04b}": 0
        for value in range(1, 16)
    }

    for species in universe:
        signature = "".join(
            "1" if species in sets_by_location[location] else "0"
            for location in LOCATIONS
        )
        counts[signature] += 1

    return counts

def venn_count_table(
    city_sets: dict[str, dict[str, set[str]]],
) -> pd.DataFrame:
    """Create a table of exact exclusive four-set Venn counts."""

    rows = []

    for city in CITIES:
        counts = exclusive_venn_counts(city_sets[city])

        for signature, count in counts.items():
            included_locations = [
                location
                for location, flag in zip(LOCATIONS, signature)
                if flag == "1"
            ]

            rows.append(
                {
                    "city": city,
                    "membership_signature_UH_WWTP_SA1_SA2": signature,
                    "exclusive_locations": ",".join(included_locations),
                    "n_species": count,
                }
            )

    return pd.DataFrame(rows)

def save_city_venn(
    sets_by_location: dict[str, set[str]],
    output_file: Path,
    dpi: int,
) -> None:
    """Save one correctly constructed four-set Venn diagram."""

    try:
        from publiplots import venn as publiplots_venn
    except ImportError as error:
        raise ImportError(
            "Panel C requires the 'publiplots' package. Install it with: "
            "python -m pip install publiplots"
        ) from error

    ordered_sets = {
        location: sets_by_location[location]
        for location in LOCATIONS
    }

    fig, ax = plt.subplots(figsize=(6.2, 6.0))

    publiplots_venn(
        ordered_sets,
        colors=[PALETTE[location] for location in LOCATIONS],
        alpha=0.42,
        ax=ax,
        fmt="{size}",
        color_labels=True,
    )

    for text in ax.texts:
        text.set_fontsize(16)

    # The library calculates region positions from the actual four-set geometry,
    # so every count remains inside the corresponding Venn region.
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(
        output_file,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

# Panels D-F: prevalence heatmap and combined city-specific occupancy effects

def calculate_complete_presence(
    sample_species: pd.DataFrame,
    sample_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Create a complete sample-by-species presence table."""

    complete = complete_sample_feature_table(
        observed=sample_species[["sample_ID", "species", "present"]].copy(),
        sample_metadata=sample_metadata,
        feature_column="species",
        value_column="present",
        fill_value=False,
    )
    complete["present"] = complete["present"].astype(bool)

    required = {"city", "location", "environment", "stratum"}
    missing = required.difference(complete.columns)
    if missing:
        raise ValueError(
            "Complete presence table is missing metadata columns: "
            + ", ".join(sorted(missing))
        )
    return complete

def prepare_prevalence_analysis(
    complete_presence: pd.DataFrame,
    remove_global_singletons: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Calculate stratum, pooled and city-specific prevalence."""

    positive_counts = (
        complete_presence.groupby("species", observed=True)["present"]
        .sum()
        .astype(int)
    )
    retained_species = (
        positive_counts.loc[positive_counts > 1].index
        if remove_global_singletons
        else positive_counts.index
    )
    retained = complete_presence.loc[
        complete_presence["species"].isin(retained_species)
    ].copy()
    if retained.empty:
        raise ValueError("No species remained after singleton filtering.")

    stratum_prevalence = (
        retained.groupby(["species", "stratum"], observed=True)["present"]
        .mean()
        .unstack("stratum")
        .reindex(columns=STRATA)
    )
    pooled_prevalence = (
        retained.groupby(["species", "environment"], observed=True)["present"]
        .mean()
        .unstack("environment")
        .reindex(columns=ENVIRONMENTS)
    )
    city_prevalence = (
        retained.groupby(
            ["species", "city", "environment"], observed=True
        )["present"]
        .mean()
        .unstack("environment")
        .reindex(columns=ENVIRONMENTS)
        .reset_index()
    )

    pooled_prevalence["delta_UH_minus_WWTP"] = (
        pooled_prevalence["UH"] - pooled_prevalence["WWTP"]
    )
    pooled_prevalence["delta_UH_minus_SA"] = (
        pooled_prevalence["UH"] - pooled_prevalence["SA"]
    )
    city_prevalence["delta_UH_minus_WWTP"] = (
        city_prevalence["UH"] - city_prevalence["WWTP"]
    )
    city_prevalence["delta_UH_minus_SA"] = (
        city_prevalence["UH"] - city_prevalence["SA"]
    )

    return stratum_prevalence, pooled_prevalence, city_prevalence, positive_counts

def build_city_agreement_table(
    pooled_prevalence: pd.DataFrame,
    city_prevalence: pd.DataFrame,
) -> pd.DataFrame:
    """Combine pooled effects with city effects and agreement indicators."""

    wide = city_prevalence.pivot(
        index="species",
        columns="city",
        values=["delta_UH_minus_WWTP", "delta_UH_minus_SA"],
    )
    wide.columns = [f"{effect}_{city}" for effect, city in wide.columns]
    table = pooled_prevalence.join(wide, how="left")

    for comparator in ["WWTP", "SA"]:
        base = f"delta_UH_minus_{comparator}"
        tre = f"{base}_TRE"
        kuo = f"{base}_KUO"
        table[f"city_agreement_UH_minus_{comparator}"] = (
            table[tre].notna()
            & table[kuo].notna()
            & ((table[tre] * table[kuo]) > 0)
        )

    table["maximum_absolute_pooled_effect"] = table[
        ["delta_UH_minus_WWTP", "delta_UH_minus_SA"]
    ].abs().max(axis=1)
    return table

def select_agreeing_species(
    agreement_table: pd.DataFrame,
    effect_column: str,
    agreement_column: str,
    n_positive: int,
    n_negative: int,
) -> list[str]:
    """Select balanced effects only where both cities agree in direction."""

    eligible = agreement_table.loc[
        agreement_table[agreement_column].fillna(False)
        & agreement_table[effect_column].notna()
    ]
    positive = eligible.loc[eligible[effect_column] > 0].nlargest(
        n_positive, effect_column
    )
    negative = eligible.loc[eligible[effect_column] < 0].nsmallest(
        n_negative, effect_column
    )
    return positive.index.tolist() + negative.index.tolist()

def build_combined_effect_selection(
    agreement_table: pd.DataFrame,
    n_positive: int,
    n_negative: int,
) -> tuple[list[str], pd.DataFrame]:
    """Select the union of agreeing UH-WWTP and UH-SA effects."""

    selected_wwtp = select_agreeing_species(
        agreement_table,
        "delta_UH_minus_WWTP",
        "city_agreement_UH_minus_WWTP",
        n_positive,
        n_negative,
    )
    selected_sa = select_agreeing_species(
        agreement_table,
        "delta_UH_minus_SA",
        "city_agreement_UH_minus_SA",
        n_positive,
        n_negative,
    )
    selected_union = list(dict.fromkeys(selected_wwtp + selected_sa))
    if not selected_union:
        raise ValueError(
            "No species met the city-agreement and effect-selection criteria."
        )

    selection = agreement_table.reindex(selected_union).copy()
    selection["selected_for_UH_minus_WWTP"] = selection.index.isin(selected_wwtp)
    selection["selected_for_UH_minus_SA"] = selection.index.isin(selected_sa)

    effect_wwtp = selection["delta_UH_minus_WWTP"]
    effect_sa = selection["delta_UH_minus_SA"]
    both_positive = (effect_wwtp > 0) & (effect_sa > 0)
    both_negative = (effect_wwtp < 0) & (effect_sa < 0)
    mean_effect = selection[
        ["delta_UH_minus_WWTP", "delta_UH_minus_SA"]
    ].mean(axis=1, skipna=True)

    selection["biological_order_group"] = np.select(
        [both_positive, both_negative, mean_effect >= 0],
        [0, 3, 1],
        default=2,
    )
    selection = selection.sort_values(
        ["biological_order_group", "maximum_absolute_pooled_effect"],
        ascending=[True, False],
    )
    return selection.index.tolist(), selection

def cluster_location_columns(
    prevalence_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Cluster only city-location columns by Bray-Curtis/average linkage."""

    available = [
        column for column in STRATA
        if column in prevalence_matrix.columns
        and prevalence_matrix[column].notna().any()
    ]
    matrix = prevalence_matrix.reindex(columns=available)
    if len(available) < 2:
        raise ValueError("At least two city-location columns are required.")

    distances = pdist(
        matrix.T.fillna(0.0).to_numpy(dtype=float), metric="braycurtis"
    )
    if not np.all(np.isfinite(distances)):
        raise ValueError("Non-finite Bray-Curtis distances were produced.")

    linkage_matrix = linkage(distances, method="average")
    info = dendrogram(linkage_matrix, labels=available, no_plot=True)
    order = list(info["ivl"])
    return matrix.reindex(columns=order), linkage_matrix, available

def save_prevalence_heatmap(
    prevalence_matrix: pd.DataFrame,
    positive_counts: pd.Series,
    selected_species: list[str],
    output_figure: Path,
    output_table: Path,
    dpi: int,
) -> None:
    """
    Save an overview prevalence heatmap containing all Homo-associated viruses.

    Viral species are ordered hierarchically by:
        1. mean UH occupancy;
        2. mean WWTP occupancy;
        3. mean SA occupancy.

    Only city-location columns are clustered. Species names are omitted.
    Zero occupancy is displayed in dark grey.
    """

    # Include all Homo-associated viral species

    all_species_matrix = prevalence_matrix.copy()

    # Remove rows without any finite prevalence values.
    all_species_matrix = all_species_matrix.loc[
        all_species_matrix.notna().any(axis=1)
    ].copy()

    if all_species_matrix.empty:
        raise ValueError(
            "No Homo-associated viral species were available for the heatmap."
        )

    # Order species by UH, then WWTP, then SA occupancy

    uh_columns = [
        column
        for column in ["TRE_UH", "KUO_UH"]
        if column in all_species_matrix.columns
    ]

    wwtp_columns = [
        column
        for column in ["TRE_WWTP", "KUO_WWTP"]
        if column in all_species_matrix.columns
    ]

    sa_columns = [
        column
        for column in [
            "TRE_SA1",
            "KUO_SA1",
            "TRE_SA2",
            "KUO_SA2",
        ]
        if column in all_species_matrix.columns
    ]

    if not uh_columns:
        raise ValueError(
            "No UH columns were found in the prevalence matrix."
        )

    if not wwtp_columns:
        raise ValueError(
            "No WWTP columns were found in the prevalence matrix."
        )

    if not sa_columns:
        raise ValueError(
            "No SA1/SA2 columns were found in the prevalence matrix."
        )

    ordering_table = all_species_matrix.copy()

    ordering_table["UH_mean"] = (
        ordering_table[uh_columns]
        .mean(axis=1, skipna=True)
    )

    ordering_table["WWTP_mean"] = (
        ordering_table[wwtp_columns]
        .mean(axis=1, skipna=True)
    )

    ordering_table["SA_mean"] = (
        ordering_table[sa_columns]
        .mean(axis=1, skipna=True)
    )

    ordering_table = ordering_table.sort_values(
        by=[
            "UH_mean",
            "WWTP_mean",
            "SA_mean",
        ],
        ascending=[
            False,
            False,
            False,
        ],
        na_position="last",
    )

    all_species_matrix = ordering_table.drop(
        columns=[
            "UH_mean",
            "WWTP_mean",
            "SA_mean",
        ]
    )

    # Cluster city-location columns only

    (
        clustered_matrix,
        linkage_matrix,
        linkage_labels,
    ) = cluster_location_columns(
        all_species_matrix
    )

    # Save complete heatmap table

    output = clustered_matrix.copy()

    output.insert(
        0,
        "global_positive_sample_count",
        positive_counts.reindex(output.index),
    )

    output.insert(
        1,
        "display_species",
        [
            clean_species_name(species)
            for species in output.index
        ],
    )

    save_tsv(
        output.reset_index(),
        output_table,
    )

    n_species, n_locations = clustered_matrix.shape

    figure_height = max(
        7,
        min(
            12.0,
            0.05 * n_species,
        ),
    )

    figure_width = max(
        2,
        0.1 * n_locations,
    )

    fig = plt.figure(
        figsize=(
            figure_width,
            figure_height,
        )
    )

    grid = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[
            0.8,
            8.0,
        ],
        width_ratios=[
            1.0,
            0.05,
        ],
        hspace=0.25,
        wspace=0.14,
    )

    ax_tree = fig.add_subplot(
        grid[0, 0]
    )

    ax_heat = fig.add_subplot(
        grid[1, 0]
    )

    ax_cbar = fig.add_subplot(
        grid[1, 1]
    )

    dendrogram(
        linkage_matrix,
        labels=[
            label.replace("_", " ")
            for label in linkage_labels
        ],
        ax=ax_tree,
        orientation="top",
        color_threshold=0,
        above_threshold_color="black",
        link_color_func=lambda _: "black",
        leaf_rotation=90,
        leaf_font_size=10,
    )

    ax_tree.tick_params(
        axis="x",
        bottom=False,
        top=False,
        labelbottom=True,
        labeltop=False,
        length=0,
        pad=0,
    )

    ax_tree.set_ylabel(
        "Bray–Curtis\ndissimilarity",
    )

    ax_tree.set_xlabel("")

    sns.despine(
        ax=ax_tree,
        bottom=True,
    )

    # Colour map
    #
    # Exact zero occupancy is dark grey.
    # Positive occupancy values use viridis.
    # Missing values are light grey.

    positive_cmap = plt.get_cmap("coolwarm")

    colours = np.vstack(
        [
            np.array([0.25, 0.25, 0.25, 1.0]),
            positive_cmap(
                np.linspace(
                    0.08,
                    1.0,
                    255,
                )
            ),
        ]
    )

    cmap = mpl.colors.ListedColormap(
        colours
    )

    cmap.set_bad(
        color="lightgrey"
    )

    boundaries = np.concatenate(
        [
            [0.0, np.nextafter(0.0, 1.0)],
            np.linspace(
                np.nextafter(0.0, 1.0),
                1.0,
                256,
            )[1:],
        ]
    )

    norm = mpl.colors.BoundaryNorm(
        boundaries,
        cmap.N,
        clip=True,
    )

    heatmap_values = clustered_matrix.to_numpy(
        dtype=float
    )

    image = ax_heat.imshow(
        np.ma.masked_invalid(
            heatmap_values
        ),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )

    # Location labels appear on the dendrogram only.
    ax_heat.set_xticks(
        np.arange(n_locations)
    )

    ax_heat.set_xticklabels([])

    ax_heat.tick_params(
        axis="x",
        bottom=False,
        length=0,
    )

    # Species rows are displayed without labels.
    ax_heat.set_yticks([])

    ax_heat.set_ylabel(
        f"Human-associated viral species (n={n_species})"
    )

    colourbar = fig.colorbar(
        image,
        cax=ax_cbar,
    )

    colourbar.set_label(
        "Occupancy"
    )

    colourbar.set_ticks(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    colourbar.minorticks_off()

    fig.subplots_adjust(
        left=0.01,
        right=0.93,
        top=0.96,
        bottom=0.06,
    )

    fig.savefig(
        output_figure,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.close(fig)

def save_combined_city_effect_plot(
    city_prevalence: pd.DataFrame,
    selection_table: pd.DataFrame,
    selected_species: list[str],
    output_figure: Path,
    output_table: Path,
    dpi: int,
) -> None:
    """Plot both comparisons and both cities in one figure."""

    output = selection_table.copy().reset_index()
    if "species" not in output.columns:
        output = output.rename(columns={output.columns[0]: "species"})
    output.insert(1, "display_species", output["species"].map(clean_species_name))
    save_tsv(output, output_table)

    comparison_columns = {
        "UH − WWTP": "delta_UH_minus_WWTP",
        "UH − SA": "delta_UH_minus_SA",
    }
    comparison_colours = {
        "UH − WWTP": ENVIRONMENT_PALETTE["WWTP"],
        "UH − SA": ENVIRONMENT_PALETTE["SA"],
    }
    selected_columns = {
        "UH − WWTP": "selected_for_UH_minus_WWTP",
        "UH − SA": "selected_for_UH_minus_SA",
    }
    city_markers = {"TRE": "o", "KUO": "s"}
    comparison_offsets = {"UH − WWTP": -0.17, "UH − SA": 0.17}
    city_offsets = {"TRE": -0.045, "KUO": 0.045}

    n_species = len(selected_species)
    y_base = np.arange(n_species)
    figure_height = max(7.0, 0.40 * n_species)
    fig, ax = plt.subplots(figsize=(10.8, figure_height))

    city_lookup = city_prevalence.set_index(["species", "city"])

    for species_index, species in enumerate(selected_species):
        for comparison, effect_column in comparison_columns.items():
            if not bool(selection_table.loc[species, selected_columns[comparison]]):
                continue

            values = []
            y_values = []
            for city in CITIES:
                try:
                    value = float(city_lookup.loc[(species, city), effect_column]) * 100.0
                except KeyError:
                    value = np.nan
                values.append(value)
                y_values.append(
                    species_index
                    + comparison_offsets[comparison]
                    + city_offsets[city]
                )

            if np.all(np.isfinite(values)):
                ax.plot(
                    values,
                    y_values,
                    color=comparison_colours[comparison],
                    linewidth=1.0,
                    alpha=0.55,
                    zorder=1,
                )

            for city, value, y_value in zip(CITIES, values, y_values):
                if not np.isfinite(value):
                    continue
                ax.scatter(
                    value,
                    y_value,
                    marker=city_markers[city],
                    s=54,
                    facecolor=(
                        comparison_colours[comparison]
                        if city == "TRE"
                        else "white"
                    ),
                    edgecolor=comparison_colours[comparison],
                    linewidth=1.3,
                    zorder=3,
                )

    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlim(-105, 105)
    ax.set_yticks(y_base)
    ax.set_yticklabels(
        [
            shorten_species_name(
                clean_species_name(species), maximum_length=58
            )
            for species in selected_species
        ],
        fontsize=14,
    )
    ax.set_ylim(n_species - 0.5, -0.5)
    ax.set_xlabel("UH occupancy minus comparator (percentage points)")
    ax.set_ylabel("Viral species")
    ax.grid(False)
    sns.despine(ax=ax)

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="-",
            color=comparison_colours["UH − WWTP"],
            markerfacecolor=comparison_colours["UH − WWTP"],
            label="UH − WWTP",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="-",
            color=comparison_colours["UH − SA"],
            markerfacecolor=comparison_colours["UH − SA"],
            label="UH − SA",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="None", color="black",
            markerfacecolor="black", label="Tampere",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="None", color="black",
            markerfacecolor="white", label="Kuopio",
        ),
    ]
    ax.legend(
        handles=handles,
        ncol=4,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        handletextpad=0.5,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.42, right=0.97, top=0.94, bottom=0.08)
    fig.savefig(output_figure, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# Separate panel figures

def save_rank_abundance_figure(
    rank_abundance: pd.DataFrame,
    output_file: Path,
    dpi: int,
) -> None:
    """Save panel A."""

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    plot_rank_abundance(ax, rank_abundance, include_legend=True)
    fig.subplots_adjust(top=0.75, left=0.15, right=0.97, bottom=0.15)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def save_occupancy_frequency_figure(
    occupancy_frequency: pd.DataFrame,
    output_file: Path,
    dpi: int,
) -> None:
    """Save panel B."""

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    plot_occupancy_frequency(ax, occupancy_frequency, include_legend=True)
    fig.subplots_adjust(top=0.86, left=0.15, right=0.97, bottom=0.20)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# Main workflow

def main() -> None:
    """Run all Figure 4 analyses and save separate figures and tables."""

    configure_plotting()

    data, sample_metadata, sample_species = load_data(
        input_file=INPUT_FILE,
        detection_threshold=DETECTION_THRESHOLD,
    )

    # Panel A -----------------------------------------------------------------
    rank_abundance = calculate_rank_abundance(
        data=data,
        sample_metadata=sample_metadata,
    )
    save_tsv(
        rank_abundance,
        OUTPUT_DIR / "Figure4_A_rank_abundance.tsv",
    )
    save_rank_abundance_figure(
        rank_abundance,
        OUTPUT_DIR / "Figure4_A_rank_abundance.png",
        DPI,
    )

    # Panel B -----------------------------------------------------------------
    occupancy = calculate_city_location_occupancy(
        sample_metadata=sample_metadata,
        sample_species=sample_species,
    )
    occupancy_frequency = calculate_occupancy_frequency(occupancy)

    low_summary = event_summary(occupancy, "low_occupancy")
    core_summary = event_summary(occupancy, "core_occupancy")

    low_overall, low_pairwise = run_event_tests(
        sample_metadata=sample_metadata,
        sample_species=sample_species,
        occupancy=occupancy,
        event="low_occupancy",
        n_permutations=N_PERMUTATIONS,
        seed=RANDOM_SEED + 1,
    )
    core_overall, core_pairwise = run_event_tests(
        sample_metadata=sample_metadata,
        sample_species=sample_species,
        occupancy=occupancy,
        event="core_occupancy",
        n_permutations=N_PERMUTATIONS,
        seed=RANDOM_SEED + 2,
    )

    panel_b_tables = {
        "Figure4_B_occupancy_by_city_location.tsv": occupancy,
        "Figure4_B_occupancy_frequency.tsv": occupancy_frequency,
        "Figure4_B_low_occupancy_summary.tsv": low_summary,
        "Figure4_B_core_occupancy_summary.tsv": core_summary,
        "Figure4_B_low_occupancy_overall_test.tsv": low_overall,
        "Figure4_B_low_occupancy_pairwise_tests.tsv": low_pairwise,
        "Figure4_B_core_occupancy_overall_test.tsv": core_overall,
        "Figure4_B_core_occupancy_pairwise_tests.tsv": core_pairwise,
    }
    for filename, table in panel_b_tables.items():
        save_tsv(table, OUTPUT_DIR / filename)

    save_occupancy_frequency_figure(
        occupancy_frequency,
        OUTPUT_DIR / "Figure4_B_occupancy_frequency.png",
        DPI,
    )

    # Panel C -----------------------------------------------------------------
    city_sets = calculate_city_location_species_sets(sample_species)

    save_tsv(
        venn_count_table(city_sets),
        OUTPUT_DIR / "Figure4_C-D_city_specific_venn_counts.tsv",
    )

    membership_rows = []
    for city in CITIES:
        universe = set().union(
            *(city_sets[city][location] for location in LOCATIONS)
        )
        for species in sorted(universe):
            membership_rows.append(
                {
                    "city": city,
                    "species": species,
                    **{
                        location: int(
                            species in city_sets[city][location]
                        )
                        for location in LOCATIONS
                    },
                }
            )

    save_tsv(
        pd.DataFrame(membership_rows),
        OUTPUT_DIR / "Figure4_C-D_city_specific_venn_membership.tsv",
    )

    save_city_venn(
        city_sets["TRE"],
        OUTPUT_DIR / "Figure4_C_Tampere_venn.png",
        DPI,
    )
    save_city_venn(
        city_sets["KUO"],
        OUTPUT_DIR / "Figure4_D_Kuopio_venn.png",
        DPI,
    )

    # Panels D-F --------------------------------------------------------------
    complete_presence = calculate_complete_presence(
        sample_species=sample_species,
        sample_metadata=sample_metadata,
    )

    (
        stratum_prevalence,
        pooled_prevalence,
        city_prevalence,
        positive_counts,
    ) = prepare_prevalence_analysis(
        complete_presence=complete_presence,
        remove_global_singletons=REMOVE_GLOBAL_SINGLETONS,
    )

    agreement_table = build_city_agreement_table(
        pooled_prevalence=pooled_prevalence,
        city_prevalence=city_prevalence,
    )

    selected_species, selection_table = build_combined_effect_selection(
        agreement_table=agreement_table,
        n_positive=TOP_N_POSITIVE,
        n_negative=TOP_N_NEGATIVE,
    )

    all_prevalence_output = (
        stratum_prevalence.add_prefix("occupancy_")
        .join(
            agreement_table.rename(
                columns={
                    "UH": "pooled_occupancy_UH",
                    "WWTP": "pooled_occupancy_WWTP",
                    "SA": "pooled_occupancy_SA",
                }
            ),
            how="outer",
        )
    )
    all_prevalence_output.insert(
        0,
        "global_positive_sample_count",
        positive_counts.reindex(all_prevalence_output.index),
    )
    save_tsv(
        all_prevalence_output.reset_index(),
        OUTPUT_DIR / "Figure4_E-F_all_prevalence_values.tsv",
    )

    save_prevalence_heatmap(
        prevalence_matrix=stratum_prevalence,
        positive_counts=positive_counts,
        selected_species=selected_species,
        output_figure=(
            output_dir
            / "Figure4_E_prevalence_heatmap.png"
        ),
        output_table=(
            OUTPUT_DIR / "Figure4_E_prevalence_matrix.tsv"
        ),
        dpi=DPI,
    )

    save_combined_city_effect_plot(
        city_prevalence=city_prevalence,
        selection_table=selection_table,
        selected_species=selected_species,
        output_figure=(
            OUTPUT_DIR / "Figure4_F_city_agreement_occupancy_effects.png"
        ),
        output_table=(
            OUTPUT_DIR / "Figure4_F_city_agreement_occupancy_effects.tsv"
        ),
        dpi=DPI,
    )

    sample_counts = (
        sample_metadata.groupby(
            ["city", "location"],
            observed=True,
        )["sample_ID"]
        .nunique()
        .rename("n_samples")
        .reset_index()
    )
    save_tsv(
        sample_counts,
        OUTPUT_DIR / "Figure4_sample_counts.tsv",
    )

    print("\nFigure 4 analysis completed.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("\nSeparate PNG figures written:")
    for filename in [
        "Figure4_A_rank_abundance.png",
        "Figure4_B_occupancy_frequency.png",
        "Figure4_C_Tampere_venn.png",
        "Figure4_D_Kuopio_venn.png",
        "Figure4_E_prevalence_heatmap.png",
        "Figure4_F_city_agreement_occupancy_effects.png",
    ]:
        print(OUTPUT_DIR / filename)

    print("\nLow-occupancy overall test:")
    print(low_overall.to_string(index=False))
    print("\nCore-occupancy overall test:")
    print(core_overall.to_string(index=False))

if __name__ == "__main__":
    main()
