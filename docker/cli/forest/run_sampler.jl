import Pkg
push!(LOAD_PATH, "..")

using JSON
using RandomNumbers
using MultiScaleMapSampler

function run_multiscale2(
    ; pctGraphPath::String,
    levels::Vector{String},
    population_col::String,
    output_path::Union{Some{String}, Nothing}=nothing,
    num_dists=1,
    rng_seed=42,
    pop_dev=0.2,
    gamma=0,
    steps=1000,
    edge_weights="connections",
    output_freq=1,
    constraints_json="",
)

    nodeData = Set(vcat(levels, [population_col]))
    base_graph = BaseGraph(
        pctGraphPath,
        population_col,
        inc_node_data=nodeData,
        edge_weights=edge_weights
    )
    graph = MultiLevelGraph(base_graph, levels)

    # Defaults first; user specs below overwrite same-type entries because
    # add_constraint! keys the constraint dict by constraint type.
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(graph, num_dists, pop_dev))
    add_constraint!(constraints, ConstrainDiscontinuousTraversals(graph))
    add_constraint!(constraints, MaxCoarseNodeSplits(num_dists+1))

    if constraints_json !== nothing && !isempty(constraints_json)
        for spec in JSON.parse(constraints_json)
            kind = spec["constraint"]
            if kind == "pack_nodes"
                add_constraint!(
                    constraints,
                    PackNodeConstraint(graph, Int(spec["unpack"]), num_dists=num_dists),
                )
            elseif kind == "max_coarse_node_splits"
                add_constraint!(constraints, MaxCoarseNodeSplits(Int(spec["max_splits"])))
            elseif kind == "allowed_excess_dists_in_coarse_nodes"
                add_constraint!(
                    constraints,
                    AllowedExcessDistsInCoarseNodes(
                        graph, num_dists, Int(spec["allowable_excess"])
                    ),
                )
            elseif kind == "max_discontinuous_traversal_segments"
                add_constraint!(
                    constraints,
                    ConstrainDiscontinuousTraversals(Int(spec["max_line_segments"])),
                )
            else
                error("Unknown constraint '$(kind)' for the forest runner")
            end
        end
    end

    rng = PCG.PCGStateOneseq(UInt64, rng_seed)
    partition = MultiLevelPartition(graph, constraints, num_dists; rng=rng)

    proposal = build_forest_recom2(constraints)
    measure = Measure(Float64(gamma))

    # Writer opens its output itself now (no IO method), so pipe mode rides
    # through /dev/stdout instead of the stdout handle. smartOpen's
    # getFileExtension crashes on extension-less paths, so the route goes
    # through a .jsonl-suffixed symlink to /dev/stdout.
    writer_path = if output_path === nothing
        stdout_link = joinpath(mktempdir(), "stdout.jsonl")
        symlink("/dev/stdout", stdout_link)
        stdout_link
    else
        output_path.value
    end
    writer = Writer(measure, constraints, partition, writer_path)

    run_metropolis_hastings!(
        partition,
        proposal,
        measure,
        steps,
        rng,
        writer=writer,
        output_freq=output_freq
    )
    return nothing
end