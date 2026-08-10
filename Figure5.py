#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:09:31 2026

@author: jaanajurvansuu

Figure 5. Human-associated viruses in wastewater.

This script generates all panels of Figure 5 for the manuscript from the
processed long-format virus abundance table and MaAsLin3 output tables.

Panels produced
---------------
A. Quadrant plot summarising viral species significantly associated with
   hospital wastewater (UH), municipal wastewater (WWTP) and sewer access
   (SA) based on MaAsLin3 differential abundance and prevalence analyses.

B. Monthly abundance profiles of selected human-associated viruses in
   wastewater together with corresponding health-register incidence data
   for Tampere (TRE) and Kuopio (KUO). Wastewater and health-register
   datasets are normalised independently for visual comparison only.

C. Monthly composition of the most abundant viral families based on
   sample-level relative abundance (RPKMF_norm). Duplicate sample–family
   observations are summed before calculating sample proportions.
   Family names are displayed without the 'f__' prefix.

D. Monthly composition of viral tropisms based on sample-level relative
   abundance (RPKMF_norm).

E. Monthly composition of viral tropisms based on unique viral accessions,
   where each accession is counted once per sample before monthly
   aggregation.

General notes
-------------
- RPKMF_norm values are already normalised and are not normalised again.
- Monthly compositions are calculated from sample-level proportions to
  avoid bias caused by differing numbers of viral taxa per sample.
- Sample locations (UH, WWTP, SA1 and SA2) are extracted from sample_ID,
  with SA1 and SA2 combined as SA.
- Figures are saved as 300 dpi PNG files only.
- Statistical results are exported separately as TSV files.
- All figures use a common publication style (Arial font, Seaborn 'ticks'
  theme and consistent colour palettes) to ensure a coherent appearance
  across Figure 5.
"""

from __future__ import annotations

import calendar
import re
import textwrap
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import PchipInterpolator
from scipy.stats import pearsonr, spearmanr

DPI = 300

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "EsVirutu_abundance_metadata.tsv"
MAASLIN_RESULTS_FILE = ROOT / "Figure5_MaAsLin3_results.tsv"
OUTPUT_DIR = ROOT

TOP_FAMILIES = 12
MINIMUM_FAMILY_PERCENTAGE = 0.0

PALETTE = {
    "UH": "#E67E22",
    "WWTP": "#2E86C1",
    "SA1": "#52BE80",
    "SA2": "#239B56",
}
PALETTE["SA"] = PALETTE["SA1"]

def configure_plotting() -> None:
    """Apply the shared Figure 5 plotting style."""

    sns.set_theme(style="ticks", context="notebook")
    mpl.rcParams.update({

        "font.family": "Arial",

        # Base
        "font.size": 20,

        # Axes
        "axes.titlesize": 20,
        "axes.labelsize": 20,

        # Tick labels
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,

        # Legend
        "legend.fontsize": 20,
        "legend.title_fontsize": 20,

        # Figure
        "figure.titlesize": 20,

        # Lines
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
    })

# Panel A: MaAsLin3 four-box quadrant
# Settings

# Panel A settings
FDR = 0.10
RANDOM_SEED = 42

UH_BOXES = {
    "UH prevalence",
    "UH abundance and prevalence",
    "UH abundance",
}

LOCATION_PALETTE = PALETTE

MARKERS = {
    "UH vs WWTP": "o",
    "UH vs SA": "^",
    "Shared": "D",
}

# Highlighted virus rim and label groups
"""Known to be seasonal or epidemic"""
BLACK_RIM_VIRUSES = {
    "Betacoronavirus 1",
    "Human coronavirus NL63",
    "Severe acute respiratory syndrome-related coronavirus",

    "Rhinovirus A",
    "Rhinovirus B",
    "Rhinovirus C",

    "Enterovirus A",
    "Enterovirus B",
    "Enterovirus C",

    "Human mastadenovirus A",
    "Human mastadenovirus B",
    "Human mastadenovirus D",
    "Human mastadenovirus F",

    "Rotavirus A",
    "HMO Astrovirus A",

    "Sapporo virus",
    "Sapovirus (Sapozj-9)",

    "Bocaparvovirus primate1",
    "Bocaparvovirus primate2",

    "Hepatitis E virus",
    "Salivirus FHB",
}

"""Known to be activated or constantly on stressed and immunocompromised"""
DARK_RED_RIM_VIRUSES = {
    "Alphapapillomavirus 9",
    "Betapapillomavirus 3",

    "Betapolyomavirus secuhominis",   # JC polyomavirus

    "Cytomegalovirus humanbeta5",

    "Simplexvirus humanalpha1",
    "Simplexvirus humanalpha2",

    "Hepatitis E virus",

    "Torque teno virus",
    "TTV-like mini virus",
}

DARK_RED = "#A50026"

# Asymmetric four-box layout

X_MIN, X_MAX = -3.0, 1.0
Y_MIN, Y_MAX = -3.0, 1.0

X_DIVIDER = -0.5
Y_DIVIDER = -0.1

# Reserved distances for markers.
OUTER_X_MARGIN = 0.20
OUTER_Y_MARGIN = 0.20
VERTICAL_DIVIDER_MARGIN = 0.35
HORIZONTAL_DIVIDER_MARGIN = 0.30

# Larger upper margin leaves room for marker labels and box titles.
TOP_LABEL_MARGIN = 0.35

BOX_BOUNDS = {
    "WWTP/SA higher": (
        X_MIN + OUTER_X_MARGIN,
        X_DIVIDER - VERTICAL_DIVIDER_MARGIN,
        Y_MIN + OUTER_Y_MARGIN,
        Y_DIVIDER - HORIZONTAL_DIVIDER_MARGIN,
    ),

    "UH prevalence": (
        X_MIN + OUTER_X_MARGIN,
        X_DIVIDER - VERTICAL_DIVIDER_MARGIN,
        Y_DIVIDER + HORIZONTAL_DIVIDER_MARGIN,
        Y_MAX - TOP_LABEL_MARGIN,
    ),

    "UH abundance and prevalence": (
        X_DIVIDER + VERTICAL_DIVIDER_MARGIN,
        X_MAX - OUTER_X_MARGIN,
        Y_DIVIDER + HORIZONTAL_DIVIDER_MARGIN,
        Y_MAX - TOP_LABEL_MARGIN,
    ),

    "UH abundance": (
        X_DIVIDER + VERTICAL_DIVIDER_MARGIN,
        X_MAX - OUTER_X_MARGIN,
        Y_MIN + OUTER_Y_MARGIN,
        Y_DIVIDER - HORIZONTAL_DIVIDER_MARGIN,
    ),
}

# Read and reshape MaAsLin3 results

def read_comparison(comparison: str, filename: Path) -> pd.DataFrame:
    """Read one comparison from the combined MaAsLin3 result table."""

    data = pd.read_csv(
        filename,
        sep="\t",
        low_memory=False,
    )

    required_columns = {
        "comparison",
        "metadata",
        "feature",
        "model",
        "coef",
        "stderr",
        "qval_individual",
        "qval_joint",
    }
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(
            "Combined MaAsLin3 results are missing columns: "
            + ", ".join(sorted(missing))
        )

    comparison_lookup = {
        "UH vs WWTP": "UH_vs_WWTP",
        "UH vs SA": "UH_vs_SA",
    }

    data = data[
        data["comparison"].astype(str).eq(
            comparison_lookup[comparison]
        )
        & data["metadata"].astype(str).eq("comparison_group")
    ].copy()

    for column in [
        "coef",
        "stderr",
        "qval_individual",
        "qval_joint",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data["model"] = data["model"].astype(str).str.lower()
    data = data[
        data["model"].isin(["abundance", "prevalence"])
    ].copy()

    data = (
        data.sort_values("qval_individual", na_position="last")
        .drop_duplicates(["feature", "model"], keep="first")
    )

    wide = data.pivot(
        index="feature",
        columns="model",
        values=[
            "coef",
            "stderr",
            "qval_individual",
            "qval_joint",
        ],
    )

    wide.columns = [
        f"{column}_{model}"
        for column, model in wide.columns
    ]
    wide = wide.reset_index()

    required = [
        "coef_abundance",
        "coef_prevalence",
        "stderr_abundance",
        "stderr_prevalence",
        "qval_individual_abundance",
        "qval_individual_prevalence",
        "qval_joint_abundance",
        "qval_joint_prevalence",
    ]

    for column in required:
        if column not in wide.columns:
            wide[column] = np.nan
        wide[column] = pd.to_numeric(
            wide[column],
            errors="coerce",
        )

    wide["qval_joint"] = wide[
        [
            "qval_joint_abundance",
            "qval_joint_prevalence",
        ]
    ].min(axis=1, skipna=True)

    wide["abundance_significant"] = (
        wide["qval_individual_abundance"] < FDR
    )
    wide["prevalence_significant"] = (
        wide["qval_individual_prevalence"] < FDR
    )
    wide["joint_significant"] = wide["qval_joint"] < FDR

    wide = wide[
        wide["abundance_significant"]
        | wide["prevalence_significant"]
        | wide["joint_significant"]
    ].copy()

    wide["species"] = (
        wide["feature"]
        .astype(str)
        .str.replace(r"^s__", "", regex=True)
        .str.strip()
    )

    wide["comparison"] = comparison

    wide["best_q"] = wide[
        [
            "qval_individual_abundance",
            "qval_individual_prevalence",
            "qval_joint",
        ]
    ].min(axis=1, skipna=True)

    wide["best_q"] = pd.to_numeric(
        wide["best_q"],
        errors="coerce",
    )

    wide = wide.dropna(subset=["best_q"]).copy()
    wide["best_q"] = wide["best_q"].clip(lower=1e-300)

    wide["significance_strength"] = -np.log10(
        wide["best_q"].to_numpy(dtype=float)
    )

    return wide

# Direction and box assignment

def direction(coef):

    if pd.isna(coef):
        return None

    # UH is the reference:
    # positive = higher in WWTP/SA
    # negative = higher in UH
    if coef > 0:
        return "other"

    if coef < 0:
        return "UH"

    return None

def joint_direction(row):

    abundance_direction = direction(
        row["coef_abundance"]
    )

    prevalence_direction = direction(
        row["coef_prevalence"]
    )

    if abundance_direction == prevalence_direction:
        return abundance_direction

    if abundance_direction is None:
        return prevalence_direction

    if prevalence_direction is None:
        return abundance_direction

    abundance_score = 0.0
    prevalence_score = 0.0

    if (
        pd.notna(row["stderr_abundance"])
        and row["stderr_abundance"] > 0
    ):
        abundance_score = abs(
            row["coef_abundance"]
            / row["stderr_abundance"]
        )

    if (
        pd.notna(row["stderr_prevalence"])
        and row["stderr_prevalence"] > 0
    ):
        prevalence_score = abs(
            row["coef_prevalence"]
            / row["stderr_prevalence"]
        )

    if abundance_score >= prevalence_score:
        return abundance_direction

    return prevalence_direction

def assign_box(row):

    abundance_sig = bool(
        row["abundance_significant"]
    )

    prevalence_sig = bool(
        row["prevalence_significant"]
    )

    abundance_direction = direction(
        row["coef_abundance"]
    )

    prevalence_direction = direction(
        row["coef_prevalence"]
    )

    # Joint-only associations are treated as combined results.
    if (
        row["joint_significant"]
        and not abundance_sig
        and not prevalence_sig
    ):

        combined_direction = joint_direction(row)

        if combined_direction == "UH":
            return "UH abundance and prevalence"

        if combined_direction == "other":
            return "WWTP/SA higher"

        return None

    abundance_higher_uh = (
        abundance_sig
        and abundance_direction == "UH"
    )

    prevalence_higher_uh = (
        prevalence_sig
        and prevalence_direction == "UH"
    )

    abundance_higher_other = (
        abundance_sig
        and abundance_direction == "other"
    )

    prevalence_higher_other = (
        prevalence_sig
        and prevalence_direction == "other"
    )

    if abundance_higher_other or prevalence_higher_other:
        return "WWTP/SA higher"

    if abundance_higher_uh and prevalence_higher_uh:
        return "UH abundance and prevalence"

    if prevalence_higher_uh:
        return "UH prevalence"

    if abundance_higher_uh:
        return "UH abundance"

    return None

# Merge species shared between comparisons

def merge_shared_species(data):

    rows = []

    for (species, box), group in data.groupby(
        ["species", "box"],
        sort=False,
    ):

        comparisons = set(
            group["comparison"]
        )

        if comparisons == {
            "UH vs WWTP",
            "UH vs SA",
        }:
            comparison_type = "Shared"

        elif comparisons == {"UH vs WWTP"}:
            comparison_type = "UH vs WWTP"

        else:
            comparison_type = "UH vs SA"

        best_row = group.loc[
            group["best_q"].idxmin()
        ]

        rows.append({
            "species": species,
            "box": box,
            "comparison_type": comparison_type,
            "WWTP_present": (
                "UH vs WWTP" in comparisons
            ),
            "SA_present": (
                "UH vs SA" in comparisons
            ),
            "best_q": group["best_q"].min(),
            "significance_strength": group[
                "significance_strength"
            ].max(),
            "coef_abundance": best_row[
                "coef_abundance"
            ],
            "coef_prevalence": best_row[
                "coef_prevalence"
            ],
            "qval_individual_abundance": group[
                "qval_individual_abundance"
            ].min(skipna=True),
            "qval_individual_prevalence": group[
                "qval_individual_prevalence"
            ].min(skipna=True),
            "qval_joint": group[
                "qval_joint"
            ].min(skipna=True),
        })

    return pd.DataFrame(rows)

# Collision-aware marker placement

def add_jitter(
    data,
    min_distance=0.28,
    size_distance_scale=0.018,
    max_attempts=20000,
):
    """
    Place markers randomly within each box while reducing overlap.

    Parameters
    ----------
    data : pandas.DataFrame
        Must contain: box, best_q, and marker_size.

    min_distance : float
        Base separation between marker centres in plot coordinates.

    size_distance_scale : float
        Controls how strongly marker size increases the required separation.

    max_attempts : int
        Maximum candidate positions tested for each marker.
    """

    required_columns = {
        "box",
        "best_q",
        "marker_size",
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            "add_jitter() is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    rng = np.random.default_rng(RANDOM_SEED)

    output = data.copy()

    output["plot_x"] = np.nan
    output["plot_y"] = np.nan

    for box, group in output.groupby("box", sort=False):

        x_low, x_high, y_low, y_high = BOX_BOUNDS[box]

        if x_low >= x_high or y_low >= y_high:
            raise ValueError(
                f"Invalid BOX_BOUNDS for {box!r}: "
                f"{BOX_BOUNDS[box]}"
            )

        # Store x, y, and the collision radius of each placed marker.
        placed = []

        # Place the most significant markers first.
        indices = group.sort_values(
            "best_q",
            ascending=True,
            na_position="last",
        ).index

        for index in indices:

            marker_size = float(
                output.loc[index, "marker_size"]
            )

            # Matplotlib scatter size is expressed as marker area.
            # sqrt(marker_size) therefore approximates marker diameter.
            collision_radius = (
                min_distance
                + size_distance_scale * np.sqrt(marker_size)
            )

            position_found = False

            best_candidate = None
            best_clearance = -np.inf

            for _ in range(max_attempts):

                x = rng.uniform(x_low, x_high)
                y = rng.uniform(y_low, y_high)

                if not placed:
                    best_candidate = (x, y)
                    position_found = True
                    break

                clearances = []

                for old_x, old_y, old_radius in placed:

                    centre_distance = np.hypot(
                        x - old_x,
                        y - old_y,
                    )

                    required_distance = (
                        collision_radius + old_radius
                    )

                    clearances.append(
                        centre_distance - required_distance
                    )

                minimum_clearance = min(clearances)

                # Keep the best candidate in case no completely
                # overlap-free position can be found.
                if minimum_clearance > best_clearance:
                    best_clearance = minimum_clearance
                    best_candidate = (x, y)

                if minimum_clearance >= 0:
                    position_found = True
                    break

            if best_candidate is None:
                raise RuntimeError(
                    f"Could not generate a candidate position "
                    f"for marker {index!r} in box {box!r}."
                )

            x, y = best_candidate

            output.loc[index, "plot_x"] = x
            output.loc[index, "plot_y"] = y

            placed.append(
                (
                    x,
                    y,
                    collision_radius,
                )
            )

            if not position_found:
                print(
                    f"Warning: marker {index!r} in {box!r} "
                    "could not be placed without overlap; "
                    "the best available position was used."
                )

    return output

# Marker styling

def scale_marker_sizes(data):

    # Marker areas are in points squared.
    minimum_size = 100
    maximum_size = 600

    strengths = data[
        "significance_strength"
    ].astype(float)

    if strengths.max() == strengths.min():

        data["marker_size"] = (
            minimum_size + maximum_size
        ) / 2

        return data

    data["marker_size"] = (
        minimum_size
        + (
            strengths - strengths.min()
        )
        / (
            strengths.max() - strengths.min()
        )
        * (
            maximum_size - minimum_size
        )
    )

    return data

def marker_color(row):

    if row["box"] in {
        "UH prevalence",
        "UH abundance and prevalence",
        "UH abundance",
    }:
        return LOCATION_PALETTE["UH"]

    if row["comparison_type"] == "UH vs WWTP":
        return LOCATION_PALETTE["WWTP"]

    if row["comparison_type"] == "UH vs SA":
        return LOCATION_PALETTE["SA"]

    return "#7F8C8D"

def marker_edge(species):
    """Return a conspicuous rim for predefined highlighted viruses."""
    if species in DARK_RED_RIM_VIRUSES:
        return DARK_RED, 4

    if species in BLACK_RIM_VIRUSES:
        return "black", 4

    return "black", 0.9

# Main

def run_panel_a(maaslin_results_file: Path, output_dir: Path) -> None:
    """Create panel A and its supporting tables."""

    output_png = output_dir / "Figure5_A_MaAsLin3_associations.png"
    output_tsv = output_dir / "Figure5_A_MaAsLin3_associations.tsv"
    output_virus_list = output_dir / "Figure5_A_significant_viruses.txt"

    results = pd.concat(
        [
            read_comparison(
                comparison,
                maaslin_results_file,
            )
            for comparison in ["UH vs WWTP", "UH vs SA"]
        ],
        ignore_index=True,
    )

    if results.empty:
        raise ValueError(
            f"No significant results at q < {FDR}."
        )

    results["box"] = results.apply(
        assign_box,
        axis=1,
    )

    results = results.dropna(
        subset=["box"]
    ).copy()

    merged = merge_shared_species(results)
    merged = scale_marker_sizes(merged)
    merged = add_jitter(merged)

    merged["highlight_group"] = merged["species"].map(
        lambda species: (
            "dark red rim"
            if species in DARK_RED_RIM_VIRUSES
            else "black rim"
            if species in BLACK_RIM_VIRUSES
            else "standard"
        )
    )

    merged.to_csv(
        output_tsv,
        sep="\t",
        index=False,
    )

    # Print unique significant viruses

    unique_species = sorted(
        merged["species"]
        .dropna()
        .unique()
    )

    print(
        f"\nUnique significant viruses: "
        f"{len(unique_species)}"
    )

    for species in unique_species:
        print(species)

    with open(
        output_virus_list,
        "w",
        encoding="utf-8",
    ) as output_file:

        for species in unique_species:
            output_file.write(
                f"{species}\n"
            )

    # Draw plot

    fig, ax = plt.subplots(
        figsize=(14.5, 10.5),
        constrained_layout=True
    )

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)

    ax.axvline(
        X_DIVIDER,
        color="black",
        linewidth=1.5,
    )

    ax.axhline(
        Y_DIVIDER,
        color="black",
        linewidth=1.5,
    )

    # Draw markers first so labels can be positioned against the final geometry.
    for _, row in merged.iterrows():

        face_color = marker_color(row)
        edge_color, edge_width = marker_edge(row["species"])

        ax.scatter(
                row["plot_x"], row["plot_y"],
                marker=MARKERS[row["comparison_type"]],
                s=row["marker_size"] * 1.28,
                facecolor=face_color,
                edgecolor="white",
                linewidth=7.2,
                alpha=0.98,
                zorder=2.8,
            )

        ax.scatter(
            row["plot_x"], row["plot_y"],
            marker=MARKERS[row["comparison_type"]],
            s=row["marker_size"],
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=edge_width,
            alpha=0.94,
            zorder=3,
        )

    # Label only the three UH-enriched boxes. Labels are centered directly
    # above their markers, with no connector lines. The vertical offset scales
    # with marker radius so text starts just above the marker edge.
    labeled = merged.loc[merged["box"].isin(UH_BOXES)].copy()

    for _, row in labeled.sort_values("best_q").iterrows():
        label = textwrap.fill(str(row["species"]), width=20)
        marker_radius_points = np.sqrt(float(row["marker_size"])) / 2.0
        vertical_offset = marker_radius_points + 5.0

        ax.annotate(
            label,
            xy=(row["plot_x"], row["plot_y"]),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            fontweight=(
                "bold"
                if row["species"] in (
                    DARK_RED_RIM_VIRUSES | BLACK_RIM_VIRUSES
                )
                else "normal"
            ),
            color=(
                DARK_RED
                if row["species"] in DARK_RED_RIM_VIRUSES
                else "black"
            ),
            ha="center",
            va="bottom",
            zorder=5,
            annotation_clip=False,
        )

    # Short box titles

    left_center = (
        X_MIN + X_DIVIDER
    ) / 2

    right_center = (
        X_DIVIDER + X_MAX
    ) / 2

    ax.text(
        left_center,
        Y_MAX - 0,
        "UH prevalence",
        ha="center",
        va="top",
        fontweight="bold",
        bbox=dict(
            facecolor="#F2F2F2",
            edgecolor="none",
            boxstyle="round,pad=0.5",
            ),
    )

    ax.text(
        right_center,
        Y_MAX - 0,
        "UH abundance + prevalence",
        ha="center",
        va="top",
        fontweight="bold",
        bbox=dict(
            facecolor="#F2F2F2",
            edgecolor="none",
            boxstyle="round,pad=0.5",
            ),
    )

    ax.text(
        right_center,
        Y_MIN + 0,
        "UH abundance",
        ha="center",
        va="bottom",
        fontweight="bold",
        bbox=dict(
            facecolor="#F2F2F2",
            edgecolor="none",
            boxstyle="round,pad=0.5",
            ),
    )

    ax.text(
        left_center,
        Y_MIN + 0,
        "WWTP/SA enriched",
        ha="center",
        va="bottom",
        fontweight="bold",
        bbox=dict(
            facecolor="#F2F2F2",
            edgecolor="none",
            boxstyle="round,pad=0.5",
            ),
    )

    ax.set_xticks([])
    ax.set_yticks([])

    # Legends

    comparison_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=LOCATION_PALETTE["WWTP"],
            markeredgecolor="black",
            markersize=10.0,
            label="UH vs WWTP",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor=LOCATION_PALETTE["SA"],
            markeredgecolor="black",
            markersize=10.0,
            label="UH vs SA",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="#6E6E6E",
            markeredgecolor="black",
            markersize=10.0,
            label="Shared",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=LOCATION_PALETTE["UH"],
            markeredgecolor="black",
            markersize=10.0,
            label="UH enriched",
        ),
    ]

    first_legend = ax.legend(
        handles=comparison_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=4,
        handletextpad=0.45,
        columnspacing=0.9,
    )

    ax.add_artist(first_legend)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(
        output_png,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nSpecies by box:")
    print(
        merged["box"]
        .value_counts()
        .to_string()
    )

    print("\nHighlight groups:")
    highlight_counts = pd.Series(
        np.select(
            [
                merged["species"].isin(DARK_RED_RIM_VIRUSES),
                merged["species"].isin(BLACK_RIM_VIRUSES),
            ],
            ["dark red", "black"],
            default="standard",
        )
    ).value_counts()
    print(highlight_counts.to_string())

    print("\nLabels shown:")
    print(f"UH-box markers labeled: {len(labeled)} / {len(merged)}")

    print("\nOutputs:")
    print(output_png)
    print(output_tsv)
    print(output_virus_list)

# Panel B: seasonal viruses and health-register profiles
# Species included in the analysis

selected_species = [
    "Alphainfluenzavirus influenzae",
    "Betainfluenzavirus influenzae",
    "Orthopneumovirus hominis",
    "Metapneumovirus hominis",
    "Respirovirus pneumoniae",
    "Norwalk virus",
    "Rotavirus A",

    "Enterovirus A",
    "Enterovirus B",
    "Enterovirus C",
    "Enterovirus D",

    "Rhinovirus A",
    "Rhinovirus B",
    "Rhinovirus C",

    "Human parechovirus 1B",
    "Parechovirus A",
    "Sapporo virus",

    "Human coronavirus NL63",
    "Betacoronavirus 1",
    "Severe acute respiratory syndrome-related coronavirus",
]

# Plot order:
# Enteroviruses and Rhinoviruses are grouped together.
plot_species = selected_species.copy()

# Short labels shown in the figure.
species_display = {
    "Alphainfluenzavirus influenzae": "Influenza A",
    "Betainfluenzavirus influenzae": "Influenza B",
    "Orthopneumovirus hominis": "RSV",
    "Metapneumovirus hominis": "Human metapneumovirus",
    "Respirovirus pneumoniae": "Parainfluenza virus",
    "Norwalk virus": "Norovirus",
    "Rotavirus A": "Rotavirus A",
    "Severe acute respiratory syndrome-related coronavirus": "SARS-CoV-2",
}

# Settings

cities = ["TRE", "KUO"]

wwtp_location = "WWTP"
hospital_location = "UH"

month_order = [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3]

month_labels = [
    "Apr", "May", "Jun", "Jul", "Aug", "Sep",
    "Oct", "Nov", "Dec", "Jan", "Feb", "Mar",
]

colours = {"WWTP": PALETTE["WWTP"], "UH": PALETTE["UH"], "health": "#000000"}

# Significance criterion.
significance_alpha = 0.05

# Figure geometry.
row_offset = 1.35
profile_height = 1.0

# Position of significance circles after the profile.
significance_x_wwtp = 11.3
significance_x_uh = 11.7
significance_y_offset = 0.45
significance_marker_size = 200

# Health-register data
#
# Month order:
# Apr 2024 ... Mar 2025

health_data = {
    "TRE": {
        "Influenssa": [
            27, 41, 3, 3, 1, 1, 2, 7, 113, 196, 403, 576
        ],
        "Influenssa A": [
            12, 8, 2, 3, 1, 1, 1, 6, 104, 176, 337, 484
        ],
        "Influenssa B": [
            15, 33, 1, 0, 0, 0, 1, 1, 8, 19, 64, 92
        ],
        "RSV": [
            10, 6, 0, 1, 0, 0, 0, 0, 14, 19, 42, 85
        ],
        "Adenovirus": [
            6, 11, 5, 5, 6, 25, 41, 72, 23, 12, 13, 8
        ],
        "Rinovirus": [
            10, 24, 21, 27, 26, 52, 48, 46, 20, 13, 13, 20
        ],
        "Metapneumovirus": [
            8, 12, 1, 0, 1, 0, 1, 1, 2, 1, 13, 26
        ],
        "Parainfluenssavirus": [
            12, 25, 16, 2, 3, 2, 3, 2, 3, 5, 3, 1
        ],
        "Norovirus": [
            52, 59, 15, 10, 5, 4, 6, 4, 8, 23, 20, 32
        ],
        "Rotavirus": [
            4, 5, 5, 0, 1, 0, 1, 0, 0, 1, 0, 4
        ],
        "Enterovirus": [
            2, 1, 0, 0, 7, 12, 6, 0, 3, 1, 1, 1
        ],
        "Koronavirus SARS-CoV-2": [
            23, 35, 49, 174, 381, 230,
            172, 201, 425, 225, 82, 49
        ],
    },

    "KUO": {
        "Influenssa": [
            44, 46, 4, 3, 1, 1, 1, 16, 57, 128, 144, 232
        ],
        "Influenssa A": [
            19, 4, 0, 1, 0, 1, 1, 15, 54, 107, 105, 167
        ],
        "Influenssa B": [
            25, 42, 4, 2, 1, 0, 0, 1, 3, 20, 38, 64
        ],
        "RSV": [
            13, 8, 1, 1, 0, 0, 0, 2, 7, 5, 20, 65
        ],
        "Adenovirus": [
            25, 24, 6, 4, 55, 49, 27, 8, 5, 5, 6, 7
        ],
        "Rinovirus": [
            2, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 2
        ],
        "Metapneumovirus": [
            3, 1, 1, 0, 0, 0, 0, 0, 2, 6, 10, 17
        ],
        "Parainfluenssavirus": [
            29, 17, 9, 0, 3, 1, 1, 3, 2, 3, 2, 5
        ],
        "Norovirus": [
            29, 8, 3, 4, 0, 0, 1, 0, 1, 6, 7, 23
        ],
        "Rotavirus": [
            2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4
        ],
        "Enterovirus": [
            0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0
        ],
        "Koronavirus SARS-CoV-2": [
            22, 40, 28, 76, 100, 72,
            105, 122, 149, 86, 17, 4
        ],
    },
}

# Species-to-health-register mapping

species_to_health_category = {
    "Alphainfluenzavirus influenzae": "Influenssa A",
    "Betainfluenzavirus influenzae": "Influenssa B",
    "Orthopneumovirus hominis": "RSV",
    "Metapneumovirus hominis": "Metapneumovirus",
    "Respirovirus pneumoniae": "Parainfluenssavirus",
    "Norwalk virus": "Norovirus",
    "Rotavirus A": "Rotavirus",

    "Enterovirus A": "Enterovirus",
    "Enterovirus B": "Enterovirus",
    "Enterovirus C": "Enterovirus",
    "Enterovirus D": "Enterovirus",

    "Rhinovirus A": "Rinovirus",
    "Rhinovirus B": "Rinovirus",
    "Rhinovirus C": "Rinovirus",

    "Severe acute respiratory syndrome-related coronavirus":
        "Koronavirus SARS-CoV-2",
}

# Validate health-register data

def run_panel_b(input_file: Path, output_dir: Path) -> None:
    """Create panel B and save monthly profiles and correlations."""

    output_monthly_values = (
        output_dir / "Figure5_B_monthly_profiles.tsv"
    )
    output_correlations = (
        output_dir / "Figure5_B_health_correlations.tsv"
    )
    output_png = output_dir / "Figure5_B_seasonal_viruses_health_register.png"

    for city, city_health_data in health_data.items():
        for category, values in city_health_data.items():
            if len(values) != len(month_order):
                raise ValueError(
                    f"{city}, {category}: expected {len(month_order)} values, "
                    f"found {len(values)}."
                )

    # Read wastewater data

    df = pd.read_csv(
        input_file,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "sample_ID",
        "host_genus",
        "species",
        "RPKMF_norm",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    df["RPKMF_norm"] = pd.to_numeric(
        df["RPKMF_norm"],
        errors="coerce",
    )

    invalid_abundance = df["RPKMF_norm"].isna()

    if invalid_abundance.any():
        print(
            f"Warning: removing {invalid_abundance.sum()} rows "
            "with non-numeric RPKMF_norm values."
        )
        df = df.loc[~invalid_abundance].copy()

    # Keep Homo-associated viruses.
    df = df[
        df["host_genus"].str.strip().str.lower() == "homo"
    ].copy()

    # Clean species names.
    df["species"] = (
        df["species"]
        .replace("", "Unknown")
        .fillna("Unknown")
        .astype(str)
        .str.replace(r"^s__", "", regex=True)
        .str.strip()
    )

    # Parse sample metadata
    #
    # Expected:
    # year_month_city_location_...

    parts = df["sample_ID"].str.split("_", expand=True)

    if parts.shape[1] < 4:
        raise ValueError(
            "sample_ID must contain at least four underscore-separated "
            "fields: year_month_city_location"
        )

    df["year"] = parts[0]
    df["month"] = pd.to_numeric(parts[1], errors="coerce")
    df["city"] = parts[2].str.strip()
    df["location"] = parts[3].str.strip()

    invalid_sample_id = (
        df["month"].isna()
        | ~df["month"].between(1, 12)
        | df["city"].eq("")
        | df["location"].eq("")
    )

    if invalid_sample_id.any():
        print(
            f"Warning: removing {invalid_sample_id.sum()} rows "
            "with invalid sample metadata."
        )
        df = df.loc[~invalid_sample_id].copy()

    df["month"] = df["month"].astype(int)

    # Keep selected cities, species, WWTP and UH only.
    df = df[
        df["species"].isin(selected_species)
        & df["city"].isin(cities)
        & df["location"].isin([wwtp_location, hospital_location])
    ].copy()

    missing_species = sorted(
        set(selected_species) - set(df["species"].unique())
    )

    if missing_species:
        print("Warning: selected species absent from wastewater data:")
        for species in missing_species:
            print(f"  - {species}")

    # Helper functions

    def create_monthly_profile(location_df):
        """
        Create species-by-month mean abundance and sample-count tables.

        Missing species-month combinations are represented as zero.
        """

        monthly = (
            location_df
            .groupby(["species", "month"], as_index=False)
            .agg(
                mean_RPKMF_norm=("RPKMF_norm", "mean"),
                sample_count=("RPKMF_norm", "size"),
            )
        )

        profile = monthly.pivot_table(
            index="species",
            columns="month",
            values="mean_RPKMF_norm",
            fill_value=0,
        )

        profile = profile.reindex(
            index=selected_species,
            columns=month_order,
            fill_value=0,
        ).astype(float)

        counts = monthly.pivot_table(
            index="species",
            columns="month",
            values="sample_count",
            fill_value=0,
        )

        counts = counts.reindex(
            index=selected_species,
            columns=month_order,
            fill_value=0,
        ).astype(int)

        return profile, counts

    def normalise_to_max(values):
        """Normalise a series to its own maximum."""

        values = np.asarray(values, dtype=float)

        if not np.isfinite(values).any():
            return np.full_like(values, np.nan, dtype=float)

        maximum = np.nanmax(values)

        if maximum <= 0:
            return np.zeros_like(values, dtype=float)

        return values / maximum

    def smooth_profile(x, values, x_smooth):
        """
        Produce a continuous PCHIP curve.

        Small negative interpolation artefacts are set to zero.
        """

        values = np.asarray(values, dtype=float)

        interpolator = PchipInterpolator(
            x,
            values,
            extrapolate=False,
        )

        smoothed = interpolator(x_smooth)

        return np.clip(smoothed, 0, None)

    def safe_correlations(wastewater_values, health_values):
        """
        Calculate Spearman and Pearson correlations.

        Correlations are exported but not printed inside the figure.
        """

        wastewater_values = np.asarray(
            wastewater_values,
            dtype=float,
        )

        health_values = np.asarray(
            health_values,
            dtype=float,
        )

        valid = (
            np.isfinite(wastewater_values)
            & np.isfinite(health_values)
        )

        x = wastewater_values[valid]
        y = health_values[valid]

        result = {
            "n_months": int(valid.sum()),
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
        }

        if len(x) < 3:
            return result

        if np.all(x == x[0]) or np.all(y == y[0]):
            return result

        spearman_result = spearmanr(x, y)
        pearson_result = pearsonr(x, y)

        result["spearman_rho"] = float(spearman_result.statistic)
        result["spearman_p"] = float(spearman_result.pvalue)
        result["pearson_r"] = float(pearson_result.statistic)
        result["pearson_p"] = float(pearson_result.pvalue)

        return result

    def is_significant_positive(correlation_result):
        """
        Return True for a positive Spearman correlation with p < alpha.
        """

        rho = correlation_result["spearman_rho"]
        p_value = correlation_result["spearman_p"]

        return (
            np.isfinite(rho)
            and np.isfinite(p_value)
            and rho > 0
            and p_value < significance_alpha
        )

    def circular_peak_distance(index_1, index_2, n_months=12):
        """Calculate circular distance between two peak months."""

        difference = abs(index_1 - index_2)

        return min(
            difference,
            n_months - difference,
        )

    def prepare_city_data(city):
        """Prepare profiles and shared WWTP/UH normalisation."""

        city_df = df[df["city"] == city].copy()

        wwtp_df = city_df[
            city_df["location"] == wwtp_location
        ].copy()

        uh_df = city_df[
            city_df["location"] == hospital_location
        ].copy()

        if wwtp_df.empty:
            print(f"Warning: no WWTP data found for {city}.")

        if uh_df.empty:
            print(f"Warning: no UH data found for {city}.")

        wwtp_profile, wwtp_counts = create_monthly_profile(wwtp_df)
        uh_profile, uh_counts = create_monthly_profile(uh_df)

        # Shared species-specific scale for WWTP and UH.
        shared_maximum = pd.concat(
            [
                wwtp_profile.max(axis=1).rename("WWTP_max"),
                uh_profile.max(axis=1).rename("UH_max"),
            ],
            axis=1,
        ).max(axis=1)

        shared_maximum = shared_maximum.replace(0, np.nan)

        wwtp_normalised = wwtp_profile.div(
            shared_maximum,
            axis=0,
        )

        uh_normalised = uh_profile.div(
            shared_maximum,
            axis=0,
        )

        return {
            "wwtp_profile": wwtp_profile,
            "uh_profile": uh_profile,
            "wwtp_counts": wwtp_counts,
            "uh_counts": uh_counts,
            "shared_maximum": shared_maximum,
            "wwtp_normalised": wwtp_normalised,
            "uh_normalised": uh_normalised,
        }

    # Prepare data for both cities

    city_results = {
        city: prepare_city_data(city)
        for city in cities
    }

    # Calculate and store correlations

    all_monthly_rows = []
    all_correlation_rows = []

    # Convenient lookup for plotting significance circles.
    correlation_lookup = {}

    for city in cities:

        results = city_results[city]
        correlation_lookup[city] = {}

        for species in selected_species:

            health_category = species_to_health_category.get(species)

            if health_category is None:
                health_values = np.full(
                    len(month_order),
                    np.nan,
                )

                health_normalised = np.full(
                    len(month_order),
                    np.nan,
                )
            else:
                health_values = np.asarray(
                    health_data[city][health_category],
                    dtype=float,
                )

                health_normalised = normalise_to_max(
                    health_values
                )

            wwtp_values = (
                results["wwtp_profile"]
                .loc[species, month_order]
                .to_numpy(dtype=float)
            )

            uh_values = (
                results["uh_profile"]
                .loc[species, month_order]
                .to_numpy(dtype=float)
            )

            wwtp_normalised_values = (
                results["wwtp_normalised"]
                .loc[species, month_order]
                .to_numpy(dtype=float)
            )

            uh_normalised_values = (
                results["uh_normalised"]
                .loc[species, month_order]
                .to_numpy(dtype=float)
            )

            if health_category is not None:

                wwtp_correlation = safe_correlations(
                    wwtp_values,
                    health_values,
                )

                uh_correlation = safe_correlations(
                    uh_values,
                    health_values,
                )

                health_peak_index = int(np.nanargmax(health_values))
                wwtp_peak_index = int(np.nanargmax(wwtp_values))
                uh_peak_index = int(np.nanargmax(uh_values))

                wwtp_peak_distance = circular_peak_distance(
                    wwtp_peak_index,
                    health_peak_index,
                )

                uh_peak_distance = circular_peak_distance(
                    uh_peak_index,
                    health_peak_index,
                )

            else:
                wwtp_correlation = {
                    "n_months": 0,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                }

                uh_correlation = {
                    "n_months": 0,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                }

                wwtp_peak_distance = np.nan
                uh_peak_distance = np.nan

            wwtp_significant_positive = is_significant_positive(
                wwtp_correlation
            )

            uh_significant_positive = is_significant_positive(
                uh_correlation
            )

            correlation_lookup[city][species] = {
                "WWTP": wwtp_correlation,
                "UH": uh_correlation,
                "WWTP_significant_positive": wwtp_significant_positive,
                "UH_significant_positive": uh_significant_positive,
            }

            all_correlation_rows.append({
                "city": city,
                "species": species,
                "display_name": species_display.get(species, species),
                "health_category": health_category,

                "WWTP_health_n_months":
                    wwtp_correlation["n_months"],

                "WWTP_health_spearman_rho":
                    wwtp_correlation["spearman_rho"],

                "WWTP_health_spearman_p":
                    wwtp_correlation["spearman_p"],

                "WWTP_significant_positive_spearman":
                    wwtp_significant_positive,

                "WWTP_health_pearson_r":
                    wwtp_correlation["pearson_r"],

                "WWTP_health_pearson_p":
                    wwtp_correlation["pearson_p"],

                "UH_health_n_months":
                    uh_correlation["n_months"],

                "UH_health_spearman_rho":
                    uh_correlation["spearman_rho"],

                "UH_health_spearman_p":
                    uh_correlation["spearman_p"],

                "UH_significant_positive_spearman":
                    uh_significant_positive,

                "UH_health_pearson_r":
                    uh_correlation["pearson_r"],

                "UH_health_pearson_p":
                    uh_correlation["pearson_p"],

                "WWTP_health_peak_distance_months":
                    wwtp_peak_distance,

                "UH_health_peak_distance_months":
                    uh_peak_distance,
            })

            for month_index, month in enumerate(month_order):

                all_monthly_rows.append({
                    "city": city,
                    "species": species,
                    "display_name": species_display.get(species, species),
                    "health_category": health_category,
                    "month": month,
                    "month_label": month_labels[month_index],

                    "WWTP_sample_count":
                        int(results["wwtp_counts"].loc[species, month]),

                    "UH_sample_count":
                        int(results["uh_counts"].loc[species, month]),

                    "WWTP_mean_RPKMF_norm":
                        wwtp_values[month_index],

                    "UH_mean_RPKMF_norm":
                        uh_values[month_index],

                    "shared_WWTP_UH_max_RPKMF_norm":
                        results["shared_maximum"].loc[species],

                    "WWTP_shared_normalised":
                        wwtp_normalised_values[month_index],

                    "UH_shared_normalised":
                        uh_normalised_values[month_index],

                    "health_register_cases":
                        health_values[month_index],

                    "health_register_normalised":
                        health_normalised[month_index],
                })

    # Save numerical outputs

    pd.DataFrame(all_monthly_rows).to_csv(
        output_monthly_values,
        sep="\t",
        index=False,
        na_rep="NA",
    )

    pd.DataFrame(all_correlation_rows).to_csv(
        output_correlations,
        sep="\t",
        index=False,
        na_rep="NA",
    )

    # Combined two-panel figure

    x = np.arange(len(month_order))
    x_smooth = np.linspace(0, len(month_order) - 1, 400)

    n_species = len(plot_species)

    fig_height = max(
        12,
        0.80 * n_species,
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(18, fig_height),
        sharey=True,
    )

    for panel_index, (ax, city) in enumerate(zip(axes, cities)):

        results = city_results[city]

        for species_index, species in enumerate(plot_species):

            # First species appears at the top.
            base = (n_species - 1 - species_index) * row_offset

            wwtp_values = (
                results["wwtp_normalised"]
                .loc[species, month_order]
                .fillna(0)
                .to_numpy(dtype=float)
            )

            uh_values = (
                results["uh_normalised"]
                .loc[species, month_order]
                .fillna(0)
                .to_numpy(dtype=float)
            )

            wwtp_smooth = smooth_profile(
                x,
                wwtp_values,
                x_smooth,
            )

            uh_smooth = smooth_profile(
                x,
                uh_values,
                x_smooth,
            )

            # Row baseline.
            ax.hlines(
                y=base,
                xmin=x[0],
                xmax=x[-1],
                color="#BFBFBF",
                linewidth=0.7,
                zorder=0,
            )

            # WWTP as a filled profile.
            ax.fill_between(
                x_smooth,
                base,
                base + profile_height * wwtp_smooth,
                color=colours["WWTP"],
                alpha=0.85,
                linewidth=0,
                zorder=1,
            )

            # UH as a continuous orange line.
            ax.plot(
                x_smooth,
                base + profile_height * uh_smooth,
                color=colours["UH"],
                linewidth=2.2,
                linestyle="-",
                zorder=3,
            )

            # Health register as a continuous black line.
            health_category = species_to_health_category.get(species)

            if health_category is not None:

                health_values = np.asarray(
                    health_data[city][health_category],
                    dtype=float,
                )

                health_normalised = normalise_to_max(
                    health_values
                )

                health_smooth = smooth_profile(
                    x,
                    health_normalised,
                    x_smooth,
                )

                ax.plot(
                    x_smooth,
                    base + profile_height * health_smooth,
                    color=colours["health"],
                    linewidth=2.2,
                    linestyle="-",
                    zorder=4,
                )

            # Species labels only on the left panel.
            if panel_index == 0:

                display_label = species_display.get(
                    species,
                    species,
                )

                ax.text(
                    -0.45,
                    base + 0.35,
                    display_label,
                    fontweight="bold",
                    ha="right",
                    va="center",
                )

            # Significant positive Spearman-correlation circles

            significance = correlation_lookup[city][species]

            marker_y = base + significance_y_offset

            # Blue circle: WWTP significant positive correlation.
            if significance["WWTP_significant_positive"]:
                ax.scatter(
                    significance_x_wwtp,
                    marker_y,
                    s=significance_marker_size,
                    color=colours["WWTP"],
                    edgecolor="black",
                    clip_on=False,
                    zorder=8,
                )

            # Orange circle: UH significant positive correlation.
            if significance["UH_significant_positive"]:
                ax.scatter(
                    significance_x_uh,
                    marker_y,
                    s=significance_marker_size,
                    color=colours["UH"],
                    edgecolor="black",
                    clip_on=False,
                    zorder=8,
                )

        # City panel label.
        ax.text(
            0.5,
            1.01,
            city,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

        # Extra horizontal room for the significance circles.
        ax.set_xlim(
            -0.5,
            12.25,
        )

        ax.set_ylim(
            -0.2,
            (n_species - 1) * row_offset + 1.2,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(month_labels)

        ax.set_yticks([])

        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_color("#808080")

    # One shared legend

    legend_elements = [
        Patch(
            facecolor=colours["WWTP"],
            edgecolor="black",
            alpha=0.85,
            label="WWTP",
        ),
        Line2D(
            [0],
            [0],
            color=colours["UH"],
            linewidth=3,
            linestyle="-",
            label="UH",
        ),
        Line2D(
            [0],
            [0],
            color=colours["health"],
            linewidth=3,
            linestyle="-",
            label="Health register cases",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colours["WWTP"],
            markeredgecolor="black",
            markersize=15,
            label="Significant positive WWTP–health correlation",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colours["UH"],
            markeredgecolor="black",
            markersize=15,
            label="Significant positive UH–health correlation",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.55, 0.06),
    )

    fig.subplots_adjust(
        left=0.24,
        right=0.98,
        top=0.97,
        bottom=0.11,
        wspace=0.01,
    )

    fig.savefig(
        output_png,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:")
    print(f"  {output_png}")
    print(f"  {output_monthly_values}")
    print(f"  {output_correlations}")

# Panels C-E: family and tropism composition
CITIES = ("TRE", "KUO")
LOCATIONS = ("UH", "WWTP", "SA1", "SA2")
ENVIRONMENTS = ("UH", "WWTP", "SA")
MONTH_ORDER = (4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3)
MONTH_LABELS = tuple(calendar.month_abbr[month] for month in MONTH_ORDER)
UNKNOWN = "Unknown"
OTHER = "Other"

location_palette = {
    "UH": "#E67E22",
    "WWTP": "#2E86C1",
    "SA1": "#52BE80",
    "SA2": "#239B56",
}

def first_existing(columns: pd.Index, candidates: tuple[str, ...], label: str) -> str:
    """Return the first available column from a list of accepted names."""

    for column in candidates:
        if column in columns:
            return column
    raise KeyError(f"Could not find {label}. Expected one of: {', '.join(candidates)}")

def extract_code(sample_id: object, allowed: tuple[str, ...], label: str) -> str:
    """Extract one city or sampling-location code from sample_ID."""

    text = str(sample_id).upper()
    strict = [
        code
        for code in allowed
        if re.search(rf"(^|[_\-.]){re.escape(code)}($|[_\-.])", text)
    ]
    matches = strict or [code for code in allowed if code in text]
    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely identify {label} from sample_ID "
            f"'{sample_id}'. Matches: {matches}"
        )
    return matches[0]

def parse_month(data: pd.DataFrame) -> pd.Series:
    """Identify sampling month from an existing column or sample_ID."""

    for column in (
        "date",
        "Date",
        "sample_date",
        "sampling_date",
        "collection_date",
        "collectionDate",
    ):
        if column in data.columns:
            parsed = pd.to_datetime(data[column], errors="coerce")
            if parsed.notna().any():
                return parsed.dt.month

    month_lookup = {
        name.lower(): number
        for number, name in enumerate(calendar.month_name)
        if name
    }
    month_lookup.update(
        {
            name.lower(): number
            for number, name in enumerate(calendar.month_abbr)
            if name
        }
    )

    for column in ("month", "Month", "sampling_month", "collection_month"):
        if column in data.columns:
            numeric = pd.to_numeric(data[column], errors="coerce")
            if numeric.between(1, 12).any():
                return numeric
            text_month = data[column].astype(str).str.strip().str.lower().map(month_lookup)
            if text_month.notna().any():
                return text_month

    extracted = data["sample_ID"].astype(str).str.extract(
        r"(?:19|20)\d{2}[-_.]?([01]?\d)(?:[-_.]?\d{1,2})?",
        expand=False,
    )
    extracted = pd.to_numeric(extracted, errors="coerce")
    if extracted.between(1, 12).any():
        return extracted

    raise ValueError("Could not identify sampling month.")

def load_data(input_file: str | Path) -> tuple[pd.DataFrame, str, str]:
    """Load and validate the processed Homo-associated virus table."""

    data = pd.read_csv(input_file, sep="\t", low_memory=False)
    abundance = first_existing(
        data.columns,
        ("RPKMF_norm", "norm_RPKMF", "normalized_RPKMF", "normalised_RPKMF"),
        "normalised RPKMF column",
    )
    family = first_existing(
        data.columns,
        ("family", "Family", "virus_family", "viral_family", "family_name"),
        "viral family column",
    )
    accession = first_existing(
        data.columns,
        (
            "Accession",
            "accession",
            "accession_id",
            "accession_ID",
            "virus_accession",
            "seq_accession",
            "sequence_accession",
            "NCBI_accession",
        ),
        "accession column",
    )

    required = {
        "sample_ID",
        "host_genus",
        "tropism_broad",
        abundance,
        family,
        accession,
    }
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(sorted(missing))}")

    data = data.loc[
        data["host_genus"].astype(str).str.strip().str.casefold().eq("homo")
    ].copy()

    data[abundance] = pd.to_numeric(data[abundance], errors="coerce")
    invalid_n = int(data[abundance].isna().sum())
    if invalid_n:
        warnings.warn(f"Removing {invalid_n} rows with non-numeric RPKMF_norm.")

    data = data.dropna(subset=["sample_ID", abundance]).copy()
    if (data[abundance] < 0).any():
        raise ValueError("Negative RPKMF_norm values were detected.")

    data["viral_family"] = (
        data[family]
        .fillna(UNKNOWN)
        .astype(str)
        .str.strip()
        .replace("", UNKNOWN)
        .str.replace(r"^f__", "", regex=True)
    )
    data["tropism_broad"] = (
        data["tropism_broad"]
        .fillna(UNKNOWN)
        .astype(str)
        .str.strip()
        .replace("", UNKNOWN)
    )
    data[accession] = data[accession].astype("string").str.strip()

    data["city"] = data["sample_ID"].map(
        lambda value: extract_code(value, CITIES, "city")
    )
    data["location"] = data["sample_ID"].map(
        lambda value: extract_code(value, LOCATIONS, "location")
    )
    data["environment"] = data["location"].replace({"SA1": "SA", "SA2": "SA"})
    data["month"] = parse_month(data)
    data = data.loc[data["month"].between(1, 12)].copy()
    data["month"] = data["month"].astype(int)

    return data, abundance, accession

def sample_relative_composition(
    data: pd.DataFrame,
    abundance: str,
    category: str,
) -> pd.DataFrame:
    """Calculate category fractions within each sample after summing duplicates."""

    values = (
        data.groupby(
            ["sample_ID", "environment", "month", category],
            observed=True,
            as_index=False,
        )[abundance]
        .sum()
        .rename(columns={abundance: "category_RPKMF_norm"})
    )
    totals = values.groupby("sample_ID", observed=True)[
        "category_RPKMF_norm"
    ].transform("sum")
    values["fraction"] = np.divide(
        values["category_RPKMF_norm"],
        totals,
        out=np.zeros(len(values), dtype=float),
        where=totals.to_numpy() > 0,
    )
    return values

def collapse_categories(
    sample_values: pd.DataFrame,
    category: str,
    display_category: str,
    top_n: int,
    minimum_percentage: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Retain the most abundant categories and collapse the remainder as Other."""

    importance = (
        sample_values.groupby(category, observed=True)["fraction"]
        .mean()
        .sort_values(ascending=False)
    )
    retained = importance.index.tolist()
    if top_n > 0:
        retained = retained[:top_n]
    if minimum_percentage > 0:
        retained = [
            item
            for item in retained
            if importance.loc[item] >= minimum_percentage / 100.0
        ]

    values = sample_values.assign(
        **{
            display_category: sample_values[category].where(
                sample_values[category].isin(retained), OTHER
            )
        }
    )
    values = (
        values.groupby(
            ["sample_ID", "environment", "month", display_category],
            observed=True,
            as_index=False,
        )
        .agg(
            category_RPKMF_norm=("category_RPKMF_norm", "sum"),
            fraction=("fraction", "sum"),
        )
    )

    order = [item for item in importance.index if item in retained]
    if values[display_category].eq(OTHER).any():
        order.append(OTHER)
    return values, order

def monthly_composition(
    sample_values: pd.DataFrame,
    category: str,
    category_order: list[str],
) -> pd.DataFrame:
    """Average sample fractions by month and renormalise each composition."""

    monthly = (
        sample_values.groupby(
            ["environment", "month", category],
            observed=True,
            as_index=False,
        )["fraction"]
        .agg(mean_fraction="mean", standard_deviation="std", sample_n="count")
    )
    monthly["standard_error"] = monthly["standard_deviation"] / np.sqrt(
        monthly["sample_n"]
    )

    full_index = pd.MultiIndex.from_product(
        [ENVIRONMENTS, MONTH_ORDER, category_order],
        names=["environment", "month", category],
    )
    monthly = (
        monthly.set_index(["environment", "month", category])
        .reindex(full_index)
        .reset_index()
    )
    numeric = ["mean_fraction", "standard_deviation", "standard_error", "sample_n"]
    monthly[numeric] = monthly[numeric].fillna(0.0)
    monthly["sample_n"] = monthly["sample_n"].astype(int)

    totals = monthly.groupby(["environment", "month"], observed=True)[
        "mean_fraction"
    ].transform("sum")
    monthly["composition_fraction"] = np.divide(
        monthly["mean_fraction"],
        totals,
        out=np.zeros(len(monthly), dtype=float),
        where=totals.to_numpy() > 0,
    )
    monthly["percentage"] = 100.0 * monthly["composition_fraction"]
    return monthly

def accession_composition(
    data: pd.DataFrame,
    abundance: str,
    accession: str,
    tropism_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate unique-accession composition by tropism."""

    sample_counts = (
        data.loc[(data[abundance] > 0) & data[accession].notna()]
        .groupby(
            ["sample_ID", "environment", "month", "tropism_broad"],
            observed=True,
            as_index=False,
        )[accession]
        .nunique()
        .rename(columns={accession: "accession_count"})
    )
    totals = sample_counts.groupby("sample_ID", observed=True)[
        "accession_count"
    ].transform("sum")
    sample_counts["fraction"] = np.divide(
        sample_counts["accession_count"],
        totals,
        out=np.zeros(len(sample_counts), dtype=float),
        where=totals.to_numpy() > 0,
    )
    return sample_counts, monthly_composition(
        sample_counts,
        "tropism_broad",
        tropism_order,
    )

# Plotting

def category_colours(category_order: list[str]) -> dict[str, object]:
    """Create a fixed colour map for stacked categories."""

    cmap = plt.get_cmap("tab20")
    colours = {
        category: cmap(index % cmap.N)
        for index, category in enumerate(category_order)
    }
    if OTHER in colours:
        colours[OTHER] = (0.65, 0.65, 0.65, 1.0)
    if UNKNOWN in colours:
        colours[UNKNOWN] = (0.80, 0.80, 0.80, 1.0)
    return colours

def legend_handles(order: list[str], colours: dict[str, object]) -> list[Line2D]:
    """Create compact handles for stacked-area legends."""

    return [
        Line2D([0], [0], color=colours[item], linewidth=8, label=item)
        for item in order
    ]

def draw_environment_column(
    axes: np.ndarray,
    monthly: pd.DataFrame,
    category: str,
    category_order: list[str],
    colours: dict[str, object],
    y_label: str,
) -> None:
    """Draw UH, WWTP and SA vertically from top to bottom."""

    x = np.arange(len(MONTH_ORDER))
    for axis, environment in zip(np.ravel(axes), ENVIRONMENTS):
        subset = monthly.loc[monthly["environment"].eq(environment)]
        matrix = (
            subset.pivot_table(
                index="month",
                columns=category,
                values="composition_fraction",
                aggfunc="first",
                fill_value=0.0,
            )
            .reindex(index=MONTH_ORDER, columns=category_order, fill_value=0.0)
        )

        axis.stackplot(
            x,
            *[100.0 * matrix[item].to_numpy(float) for item in category_order],
            colors=[colours[item] for item in category_order],
            alpha=0.92,
        )
        axis.set_title(environment)
        axis.set_xlim(0, len(MONTH_ORDER) - 1)
        axis.set_ylim(0, 100)
        axis.set_ylabel(y_label)
        axis.grid(False)
        sns.despine(ax=axis)

    axes = np.ravel(axes)
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)
    axes[-1].set_xticks(x, MONTH_LABELS, rotation=45, ha="right")
    axes[-1].set_xlabel("Month")

def save_panel_c(
    monthly: pd.DataFrame,
    family_order: list[str],
    colours: dict[str, object],
    output_file: Path,
) -> None:
    """Save panel C with vertically stacked environments and a right-side legend."""

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11.5), sharex=True, sharey=True)
    draw_environment_column(
        axes,
        monthly,
        "display_family",
        family_order,
        colours,
        "Relative abundance (%)",
    )
    fig.legend(
        handles=legend_handles(family_order, colours),
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.75, 0.5),
    )
    fig.subplots_adjust(left=0.08, right=0.72, top=0.98, bottom=0.08, hspace=0.18)
    fig.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def save_composition_panel(
    monthly: pd.DataFrame,
    category: str,
    category_order: list[str],
    colours: dict[str, object],
    y_label: str,
    output_file: Path,
) -> None:
    """Save one monthly composition panel for UH, WWTP and SA."""

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.5, 11.5),
        sharex=True,
        sharey=True,
    )

    draw_environment_column(
        axes,
        monthly,
        category,
        category_order,
        colours,
        y_label,
    )

    fig.legend(
        handles=legend_handles(category_order, colours),
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.75, 0.5),
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.72,
        top=0.98,
        bottom=0.08,
        hspace=0.18,
    )

    fig.savefig(
        output_file,
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

def run_panels_cde(
    input_file: Path,
    output_dir: Path,
    top_families: int,
    minimum_family_percentage: float,
) -> None:
    """Create panel C and combined panels D-E with their TSV outputs."""

    data, abundance, accession = load_data(input_file)

    sample_families = sample_relative_composition(data, abundance, "viral_family")
    sample_families, family_order = collapse_categories(
        sample_families,
        category="viral_family",
        display_category="display_family",
        top_n=top_families,
        minimum_percentage=minimum_family_percentage,
    )
    family_monthly = monthly_composition(
        sample_families,
        "display_family",
        family_order,
    )

    sample_tropism = sample_relative_composition(data, abundance, "tropism_broad")
    tropism_order = (
        sample_tropism.groupby("tropism_broad", observed=True)["fraction"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    tropism_monthly = monthly_composition(
        sample_tropism,
        "tropism_broad",
        tropism_order,
    )
    sample_accessions, accession_monthly = accession_composition(
        data,
        abundance,
        accession,
        tropism_order,
    )

    family_colours = category_colours(family_order)
    tropism_colours = category_colours(tropism_order)

    outputs = {
        "Figure5_C_family_sample_values.tsv": sample_families,
        "Figure5_C_family_monthly_statistics.tsv": family_monthly,
        "Figure5_D_tropism_abundance_sample_values.tsv": sample_tropism,
        "Figure5_D_tropism_abundance_monthly_statistics.tsv": tropism_monthly,
        "Figure5_E_tropism_accession_sample_values.tsv": sample_accessions,
        "Figure5_E_tropism_accession_monthly_statistics.tsv": accession_monthly,
    }
    for filename, table in outputs.items():
        table.to_csv(output_dir / filename, sep="\t", index=False)

    save_panel_c(
        family_monthly,
        family_order,
        family_colours,
        output_dir / "Figure5_C_viral_family_composition.png",
    )
    save_composition_panel(
        tropism_monthly,
        "tropism_broad",
        tropism_order,
        tropism_colours,
        "Relative abundance (%)",
        output_dir / "Figure5_D_tropism_abundance_composition.png",
    )

    save_composition_panel(
        accession_monthly,
        "tropism_broad",
        tropism_order,
        tropism_colours,
        "Accession composition (%)",
        output_dir / "Figure5_E_tropism_accession_composition.png",
    )

def main() -> None:
    """Run all Figure 5 analyses."""

    configure_plotting()

    run_panel_a(
        MAASLIN_RESULTS_FILE,
        OUTPUT_DIR,
    )

    run_panel_b(
        INPUT_FILE,
        OUTPUT_DIR,
    )

    run_panels_cde(
        INPUT_FILE,
        OUTPUT_DIR,
        TOP_FAMILIES,
        MINIMUM_FAMILY_PERCENTAGE,
    )

    print(f"All Figure 5 outputs written to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
