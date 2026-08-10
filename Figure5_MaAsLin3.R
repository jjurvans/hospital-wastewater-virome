#!/usr/bin/env Rscript

# Figure 5: MaAsLin3 differential abundance and prevalence analysis
#
# Inputs:
#   maaslin3_species_abundance.tsv
#   maaslin3_metadata.tsv
#
# Outputs:
#   UH_vs_WWTP/                  standard MaAsLin3 output
#   UH_vs_SA/                    standard MaAsLin3 output
#   Figure5_MaAsLin3_results.tsv combined result table used by Figure5.py

suppressPackageStartupMessages({
    library(maaslin3)
})

feature_file <- "maaslin3_species_abundance.tsv"
metadata_file <- "maaslin3_metadata.tsv"
combined_result_file <- "Figure5_MaAsLin3_results.tsv"

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

# Keep only samples present in both tables.
samples <- intersect(rownames(features), rownames(metadata))

if (length(samples) == 0) {
    stop("No matching sample identifiers between abundance and metadata tables.")
}

features <- features[samples, , drop = FALSE]
metadata <- metadata[samples, , drop = FALSE]
metadata <- metadata[rownames(features), , drop = FALSE]

# Analysis settings.
fdr_threshold <- 0.10
minimum_prevalence_per_group <- 0.10
minimum_positive_samples <- 3


run_comparison <- function(other_group) {

    comparison_name <- paste0("UH_vs_", other_group)
    comparison_output <- comparison_name

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

    if (length(unique(comparison_metadata$comparison_group)) != 2) {
        stop(comparison_name, ": both comparison groups are required.")
    }

    # Species must be detected in at least 10% of samples in BOTH groups,
    # with an absolute minimum of three positive samples per group.
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
        stop(comparison_name, ": no species passed the detection filter.")
    }

    # Run MaAsLin3.
    maaslin3(
        input_data = comparison_features,
        input_metadata = comparison_metadata,
        output = comparison_output,

        formula = "~ comparison_group",

        normalization = "NONE",
        transform = "LOG",
        zero_threshold = 0,

        min_prevalence = 0,
        min_abundance = 0,

        correction = "BH",
        max_significance = fdr_threshold,

        # Correct abundance associations relative to the overall
        # community-level abundance shift.
        median_comparison_abundance = TRUE,

        # Test direct differences in detection probability.
        median_comparison_prevalence = FALSE,

        standardize = FALSE,
        augment = TRUE,
        warn_prevalence = FALSE,
        cores = 1
    )

    # MaAsLin3 writes the complete association table here.
    result_file <- file.path(
        comparison_output,
        "all_results.tsv"
    )

    if (!file.exists(result_file)) {
        stop(
            comparison_name,
            ": expected MaAsLin3 output was not created: ",
            result_file
        )
    }

    result <- read.delim(
        result_file,
        check.names = FALSE,
        stringsAsFactors = FALSE
    )

    result$comparison <- comparison_name

    # Put comparison first.
    result <- result[
        c(
            "comparison",
            setdiff(colnames(result), "comparison")
        )
    ]

    return(result)
}


# Run both comparisons and capture their result tables.
results_wwtp <- run_comparison("WWTP")
results_sa <- run_comparison("SA")

combined_results <- rbind(
    results_wwtp,
    results_sa
)

# Write the single table used by the Python Figure 5 script.
write.table(
    combined_results,
    file = combined_result_file,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

if (!file.exists(combined_result_file)) {
    stop("Combined Figure 5 result file was not created.")
}

cat(
    "\nFinished.\n",
    "Combined results: ",
    normalizePath(combined_result_file),
    "\n",
    sep = ""
)
