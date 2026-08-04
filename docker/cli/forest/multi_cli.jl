using ArgParse
using JSON

include("run_sampler.jl")

const CONFIG_VERSION = 1
const FOREST_RUN_FIELDS = [
    "levels",
    "pop_col",
    "num_dists",
    "pop_dev",
    "gamma",
    "n_steps",
    "rng_seed",
    "edge_weights",
    "output_freq",
]

function require_object(value, label)
    value isa Dict || error("$(label) must be a JSON object")
    return value
end

function require_fields(value, required, label)
    missing = [field for field in required if !haskey(value, field)]
    isempty(missing) || error("$(label) is missing required fields: $(join(missing, ", "))")
end

function load_config(config_arg)
    raw_config = if startswith(lstrip(config_arg), "{")
        config_arg
    else
        isfile(config_arg) || error("Config file $(config_arg) does not exist")
        read(config_arg, String)
    end

    config = try
        JSON.parse(raw_config)
    catch exception
        error("Config is not valid JSON: $(exception)")
    end
    require_object(config, "config")
    require_fields(config, ["version", "engine", "io", "run", "constraints"], "config")

    version = config["version"]
    (version isa Integer && !(version isa Bool)) || error("config.version must be an integer")
    version == CONFIG_VERSION || error("Unsupported config version $(version)")
    config["engine"] == "forest" || error("Expected engine 'forest', got '$(config["engine"])'")

    io_config = require_object(config["io"], "config.io")
    run_config = require_object(config["run"], "config.run")
    require_fields(io_config, ["graph", "output", "writer"], "config.io")
    require_fields(run_config, FOREST_RUN_FIELDS, "config.run")
    levels = run_config["levels"]
    (levels isa Vector && !isempty(levels) && all(level -> level isa String, levels)) ||
        error("config.run.levels must be a non-empty array of strings")
    config["constraints"] isa Vector || error("config.constraints must be a JSON array")

    writer = io_config["writer"]
    writer in ["raw", "jsonl", "ben"] || error("Unsupported forest writer '$(writer)'")
    output = io_config["output"]
    (output === nothing || output isa String) || error("config.io.output must be a string or null")

    return config, raw_config
end

function metadata_path(output_path)
    stem, _ = splitext(output_path)
    return stem * "_metadata.jsonl"
end

function write_metadata(output_path, raw_config)
    open(metadata_path(output_path), "w") do metadata_file
        write(metadata_file, raw_config)
        write(metadata_file, "\n")
    end
end

parser = ArgParseSettings()

@add_arg_table! parser begin
    "--config"
        help = "Versioned gerrytools config: inline JSON (starts with '{') or a path to a JSON file"
    "--input-file-name"
        help = "Name of input file (rest of path assumed)"
    "--output-file-name"
        help = "Name of output file (rest of path assumed)"
    "--subregion-name"
        help = "Label for the subregion column"
    "--region-name"
        help = "Label for the region column"
    "--pop-name"
        help = "Label for the population column"
    "--num-dists"
        help = "Number of districts"
        arg_type = Int
    "--rng-seed"
        help = "Seed for the rng"
        arg_type = Int
    "--pop-dev"
        help = "Allowable population deviance (between 0 and 1)"
        arg_type = Float64
    "--gamma"
        help = "Value for gamma in the multiscale"
        arg_type = Float64
    "--steps"
        help = "Number of steps allowed"
        arg_type = Int
    "--constraints"
        help = "JSON array of constraint specs (see gerrytools.mgrp.Constraints)"
        arg_type = String
        default = ""
end

args = parse_args(parser)
config_mode = args["config"] !== nothing
config_only = (length(ARGS) == 2 && ARGS[1] == "--config") ||
    (length(ARGS) == 1 && startswith(ARGS[1], "--config="))
if config_mode && !config_only
    error("--config cannot be combined with legacy flags")
end

raw_config = nothing
writer = "raw"
if config_mode
    config, raw_config = load_config(args["config"])
    io_config = config["io"]
    run_config = config["run"]

    writer = io_config["writer"]
    output_path = writer == "raw" && io_config["output"] !== nothing ?
        Some(io_config["output"]) : nothing
    pct_graph_path = io_config["graph"]
    levels = Vector{String}(run_config["levels"])
    population_col = run_config["pop_col"]
    num_dists = run_config["num_dists"]
    rng_seed = run_config["rng_seed"]
    pop_dev = run_config["pop_dev"]
    gamma = run_config["gamma"]
    steps = run_config["n_steps"]
    edge_weights = run_config["edge_weights"]
    output_freq = run_config["output_freq"]
    constraints_json = JSON.json(config["constraints"])
else
    output_path = args["output-file-name"] === nothing ?
        nothing : Some(args["output-file-name"])
    pct_graph_path = args["input-file-name"]
    # The legacy flags describe exactly two hierarchy levels.
    levels = String[args["region-name"], args["subregion-name"]]
    population_col = args["pop-name"]
    num_dists = args["num-dists"]
    rng_seed = args["rng-seed"]
    pop_dev = args["pop-dev"]
    gamma = args["gamma"]
    steps = args["steps"]
    edge_weights = "connections"
    output_freq = 1
    constraints_json = args["constraints"]
end

run_multiscale2(
    pctGraphPath=pct_graph_path,
    levels=levels,
    population_col=population_col,
    output_path=output_path,
    num_dists=num_dists,
    rng_seed=rng_seed,
    pop_dev=pop_dev,
    gamma=gamma,
    steps=steps,
    edge_weights=edge_weights,
    output_freq=output_freq,
    constraints_json=constraints_json,
)

if config_mode && writer == "raw" && output_path !== nothing
    write_metadata(output_path.value, raw_config)
end
