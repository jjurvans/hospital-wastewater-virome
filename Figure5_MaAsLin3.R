#!/usr/bin/env Rscript

# Figure 5: MaAsLin3 differential abundance and prevalence analysis
#
# Required input files:
#   maaslin3_species_abundance.tsv
#   maaslin3_metadata.tsv
#
# Two comparisons are run:
#   UH vs WWTP
#   UH vs SA
#
# UH is the reference group.

suppressPackageStartupMessages({
    library(maaslin3)
})

feature_file <- "maaslin3_species_abundance.tsv"
metadata_file <- "maaslin3_metadata.tsv"

features <- read.delim(
    feature_file,
    row.names = 1,
    check.names = FALSE,
    stringsAsFactors = FALSE
)

metadata <- read.delim(
    metadata_file,
    row.names = 1,
    check.names = FALSE,
    stringsAsFactors = FALSE
)

samples <- intersect(rownames(features), rownames(metadata))
features <- features[samples, , drop = FALSE]
metadata <- metadata[samples, , drop = FALSE]
metadata <- metadata[rownames(features), , drop = FALSE]

fdr_threshold <- 0.10
minimum_prevalence_per_group <- 0.10
minimum_positive_samples <- 3

run_comparison <- function(other_group) {

    comparison_name <- paste0("UH_vs_", other_group)

    keep_samples <- metadata$location_group %in% c("UH", other_group)

    comparison_metadata <- metadata[
        keep_samples,
        ,
        drop = FALSE
    ]

    comparison_features <- features[
        rownames(comparison_metadata),
        ,
        drop = FALSE
    ]

    # UH is the reference group.
    comparison_metadata$comparison_group <- factor(
        comparison_metadata$location_group,
        levels = c("UH", other_group)
    )

    # Species must be detected in at least 10% of samples in both groups,
    # with a minimum of three positive samples per group.
    is_uh <- comparison_metadata$comparison_group == "UH"
    is_other <- comparison_metadata$comparison_group == other_group

    required_uh <- max(
        minimum_positive_samples,
        ceiling(minimum_prevalence_per_group * sum(is_uh))
    )

    required_other <- max(
        minimum_positive_samples,
        ceiling(minimum_prevalence_per_group * sum(is_other))
    )

    positive_uh <- colSums(
        comparison_features[is_uh, , drop = FALSE] > 0
    )

    positive_other <- colSums(
        comparison_features[is_other, , drop = FALSE] > 0
    )

    keep_species <- (
        positive_uh >= required_uh &
        positive_other >= required_other
    )

    comparison_features <- comparison_features[
        ,
        keep_species,
        drop = FALSE
    ]

    if (ncol(comparison_features) == 0) {
        warning(
            comparison_name,
            ": no species passed the detection filter."
        )
        return(invisible(NULL))
    }

    maaslin3(
        input_data = comparison_features,
        input_metadata = comparison_metadata,
        output = comparison_name,

        formula = "~ comparison_group",

        # RPKMF values were normalised before MaAsLin3.
        normalization = "NONE",
        transform = "LOG",
        zero_threshold = 0,

        # Filtering was performed above.
        min_prevalence = 0,
        min_abundance = 0,

        correction = "BH",
        max_significance = fdr_threshold,

        # Correct abundance associations for the overall
        # community-level abundance shift.
        median_comparison_abundance = TRUE,

        # Test direct differences in detection probability.
        median_comparison_prevalence = FALSE,

        standardize = FALSE,
        augment = TRUE,
        warn_prevalence = FALSE,
        cores = 1
    )
}

run_comparison("WWTP")
run_comparison("SA")
