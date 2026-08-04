library(argparser)
library(dplyr)
library(ggplot2)
library(sf)
library(redist)

CONFIG_VERSION <- 1L
SMC_MAP_FIELDS <- c("pop_col", "n_dists", "pop_tol", "pop_bounds")
SMC_RUN_FIELDS <- c(
    "n_sims", "rng_seed", "compactness", "resample", "adapt_k_thresh",
    "seq_alpha", "pop_temper", "final_infl", "verbose",
    "silent", "tally_columns"
)

is_json_object <- function(value) {
    is.list(value) && !is.null(names(value))
}

require_object <- function(value, label) {
    if (!is_json_object(value)) {
        stop(paste0(label, " must be a JSON object"), call. = FALSE)
    }
    value
}

require_fields <- function(value, required, label) {
    missing <- setdiff(required, names(value))
    if (length(missing) > 0) {
        stop(
            paste0(label, " is missing required fields: ", paste(missing, collapse = ", ")),
            call. = FALSE
        )
    }
}

load_config <- function(config_arg) {
    raw_config <- if (startsWith(trimws(config_arg, which = "left"), "{")) {
        config_arg
    } else {
        if (!file.exists(config_arg)) {
            stop(paste0("Config file ", config_arg, " does not exist"), call. = FALSE)
        }
        readChar(config_arg, file.info(config_arg)$size)
    }

    config <- tryCatch(
        jsonlite::fromJSON(raw_config, simplifyVector = FALSE),
        error = function(error) {
            stop(
                paste0("Config is not valid JSON: ", conditionMessage(error)),
                call. = FALSE
            )
        }
    )
    config <- require_object(config, "config")
    require_fields(config, c("version", "engine", "io", "map", "run", "constraints"), "config")

    if (!is.integer(config$version) || length(config$version) != 1) {
        stop("config.version must be an integer", call. = FALSE)
    }
    if (config$version != CONFIG_VERSION) {
        stop(paste0("Unsupported config version ", config$version), call. = FALSE)
    }
    if (!identical(config$engine, "smc")) {
        stop(paste0("Expected engine 'smc', got '", config$engine, "'"), call. = FALSE)
    }

    io_config <- require_object(config$io, "config.io")
    map_config <- require_object(config$map, "config.map")
    run_config <- require_object(config$run, "config.run")
    require_fields(io_config, c("graph", "output", "writer"), "config.io")
    require_fields(map_config, SMC_MAP_FIELDS, "config.map")
    require_fields(run_config, SMC_RUN_FIELDS, "config.run")

    if (!is.list(config$constraints) || !is.null(names(config$constraints))) {
        stop("config.constraints must be a JSON array", call. = FALSE)
    }
    if (!(io_config$writer %in% c("csv", "jsonl", "ben"))) {
        stop(paste0("Unsupported SMC writer '", io_config$writer, "'"), call. = FALSE)
    }
    if (!is.null(io_config$output) && !is.character(io_config$output)) {
        stop("config.io.output must be a string or null", call. = FALSE)
    }

    list(config = config, raw = raw_config)
}

metadata_path <- function(output_path) {
    paste0(tools::file_path_sans_ext(output_path), "_metadata.jsonl")
}

write_metadata <- function(output_path, raw_config) {
    writeLines(raw_config, metadata_path(output_path), useBytes = TRUE)
}

p <- arg_parser("This is a basic CLI app for generating SMC plans.")

p <- add_argument(
    p,
    "--config",
    help = "Versioned gerrytools config: inline JSON (starts with '{') or a path to a JSON file",
    default = NULL,
    type = "character"
)

# TAGS FOR THE REDIST_MAP CALL
p <- add_argument(
    p,
    "--shapefile",
    help = "Enter the name of the shapefile",
    type = "character"
)
p <- add_argument(
    p,
    "--pop-col",
    help = "Enter the name of the population column within the shapefile",
    default = "TOTPOP",
    type = "character"
)
p <- add_argument(
    p,
    "--n-dists",
    help = "Enter the number of districts for the redistricting.",
    type = "integer"
)
p <- add_argument(
    p,
    "--pop-tol",
    help = "Enter the allowable population deviance [between 0 and 1]",
    default = 0.01,
    type = "double"
)
p <- add_argument(
    p,
    "--pop-bounds",
    help = "Enter the population bounds with formatting (lower, target, upper)",
    default = NULL,
    type = "integer",
    nargs = 3
)

# TAGS FOR THE REDIST_SMC CALL
p <- add_argument(
    p,
    "--n-sims",
    help = "Enter the number of simulations to draw from",
    default = 1000,
    type = "integer"
)
p <- add_argument(
    p,
    "--compactness",
    help = "Enter the compactness measure for the generated districts",
    default = 1.0,
    type = "double"
)
p <- add_argument(
    p,
    "--resample",
    help = "Including this flag will set the resampling to true",
    flag = TRUE
)
p <- add_argument(
    p,
    "--constraints",
    help = "JSON array of constraint specs (see gerrytools.mgrp.Constraints)",
    default = NULL,
    type = "character"
)
p <- add_argument(
    p,
    "--adapt-k-thresh",
    help = "Enter the threshold used to select ki for each splitting iteration",
    default = 0.985,
    type = "double"
)
p <- add_argument(
    p,
    "--seq-alpha",
    help = "Enter the amount to adjust the weights by at each resampling step.",
    default = 0.5,
    type = "double"
)
p <- add_argument(
    p,
    "--pop-temper",
    help = "Enter the strength of the automatic population tempering",
    default = 0.0,
    type = "double"
)
p <- add_argument(
    p,
    "--final-infl",
    help = "Enter the multiplier for the population constraint",
    default = 1,
    type = "double"
)
p <- add_argument(
    p,
    "--verbose",
    help = "Include intermediate redist output",
    flag = TRUE
)
p <- add_argument(
    p,
    "--silent",
    help = "Suppress all diagnostic output while sampling",
    flag = TRUE
)

# OTHER FLAGS FOR DATA PROCESSING AND REPRODUCIBILITY
p <- add_argument(
    p,
    "--rng-seed",
    help = "Enter the rng seed for the run",
    default = 42,
    type = "integer"
)
p <- add_argument(
    p,
    "--tally-cols",
    help = "Enter the names of the columns that you would like to tally",
    default = NULL,
    type = "character",
    nargs = "+"
)
p <- add_argument(
    p,
    "--output-file",
    help = "Enter the name of the output file.",
    default = "./test_output.csv",
    type = "character"
)
p <- add_argument(
    p,
    "--print",
    help = "Print the output to the console",
    flag = TRUE
)

raw_args <- commandArgs(trailingOnly = TRUE)
argv <- parse_args(p)
config_mode <- !is.null(argv$config)
config_only <- (length(raw_args) == 2 && raw_args[[1]] == "--config") ||
    (length(raw_args) == 1 && startsWith(raw_args[[1]], "--config="))
if (config_mode && !config_only) {
    stop("--config cannot be combined with legacy flags", call. = FALSE)
}

raw_config <- NULL
if (config_mode) {
    loaded <- load_config(argv$config)
    config <- loaded$config
    raw_config <- loaded$raw

    argv$shapefile <- config$io$graph
    argv$output_file <- config$io$output
    argv$print <- !identical(config$io$writer, "csv") || is.null(config$io$output)
    argv$pop_col <- config$map$pop_col
    argv$n_dists <- config$map$n_dists
    argv$pop_tol <- config$map$pop_tol
    argv$pop_bounds <- unlist(config$map$pop_bounds, use.names = FALSE)
    argv$n_sims <- config$run$n_sims
    argv$rng_seed <- config$run$rng_seed
    argv$compactness <- config$run$compactness
    argv$resample <- config$run$resample
    argv$adapt_k_thresh <- config$run$adapt_k_thresh
    argv$seq_alpha <- config$run$seq_alpha
    argv$pop_temper <- config$run$pop_temper
    argv$final_infl <- config$run$final_infl
    argv$verbose <- config$run$verbose
    argv$silent <- config$run$silent
    argv$tally_cols <- unlist(config$run$tally_columns, use.names = FALSE)
    constraint_specs <- config$constraints
} else {
    constraint_specs <- list()
    if (!is.null(argv$constraints) && !is.na(argv$constraints) && nchar(argv$constraints) > 0) {
        constraint_specs <- jsonlite::fromJSON(argv$constraints, simplifyVector = FALSE)
    }
}

vtds <- st_read(dsn = argv$shapefile)

if (is.null(argv$pop_bounds) || length(argv$pop_bounds) != 3) {
    argv$pop_bounds <- NULL
}

seed <- redist_map(
    vtds,
    pop_tol = argv$pop_tol,
    total_pop = argv$pop_col,
    ndists = argv$n_dists,
    pop_bounds = argv$pop_bounds
)

smc_constraints <- list()
if (length(constraint_specs) > 0) {
    constr <- redist_constr(seed)
    for (spec in constraint_specs) {
        kind <- spec$constraint
        if (kind == "group_hinge") {
            total_pop <- if (is.null(spec$total_pop_col)) NULL else seed[[spec$total_pop_col]]
            constr <- add_constr_grp_hinge(
                constr,
                spec$strength,
                seed[[spec$group_pop_col]],
                total_pop,
                tgts_group = unlist(spec$targets)
            )
        } else if (kind == "group_power") {
            constr <- add_constr_grp_pow(
                constr,
                spec$strength,
                seed[[spec$group_pop_col]],
                seed[[spec$total_pop_col]],
                tgt_group = spec$target_group,
                tgt_other = spec$target_other,
                pow = spec$pow
            )
        } else if (kind == "status_quo") {
            constr <- add_constr_status_quo(constr, spec$strength, seed[[spec$plan_col]])
        } else if (kind == "splits") {
            constr <- add_constr_splits(constr, spec$strength, seed[[spec$admin_col]])
        } else if (kind == "incumbency") {
            constr <- add_constr_incumbency(
                constr,
                spec$strength,
                seed[[spec$incumbents_col]]
            )
        } else {
            stop(paste0("Unknown constraint '", kind, "' for the smc runner"))
        }
    }
    smc_constraints <- constr
}

set.seed(argv$rng_seed)
plans <- redist_smc(
    seed,
    nsims = argv$n_sims,
    compactness = argv$compactness,
    constraints = smc_constraints,
    resample = argv$resample,
    adapt_k_thresh = argv$adapt_k_thresh,
    seq_alpha = argv$seq_alpha,
    pop_temper = argv$pop_temper,
    final_infl = argv$final_infl,
    verbose = argv$verbose,
    silent = argv$silent
)

if (length(argv$tally_cols) > 0) {
    for (column in argv$tally_cols) {
        plans <- plans %>% mutate(!!column := tally_var(seed, !!rlang::sym(column)))
    }
}

# In jsonl/ben mode stdout carries only assignment records for the parser, so
# tallied per-district values ride in a CSV sidecar next to the output file
# instead of being silently dropped. The csv writer keeps them in the plans CSV.
if (config_mode && length(argv$tally_cols) > 0 && argv$print && !is.null(argv$output_file)) {
    tally_path <- paste0(tools::file_path_sans_ext(argv$output_file), "_tallies.csv")
    dir.create(dirname(tally_path), recursive = TRUE, showWarnings = FALSE)
    write.csv(plans, tally_path, row.names = FALSE)
}

if (argv$print) {
    cat("\nNow printing the plans:\n")

    plans <- t(as.matrix(plans))
    apply(plans, 1, function(row) {
        cat(paste0("[", paste(row, collapse = ","), "]", "\n"))
    })
    invisible(NULL)
} else {
    file_name <- argv$output_file
    dir.create(dirname(file_name), recursive = TRUE, showWarnings = FALSE)
    write.csv(plans, file_name)
    write.csv(
        t(as.matrix(plans)),
        paste0(tools::file_path_sans_ext(file_name), "_assignments.csv")
    )
    if (config_mode) {
        write_metadata(file_name, raw_config)
    }
}
