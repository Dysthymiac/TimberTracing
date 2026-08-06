
module TimberDatasets

export All, find_paths, create_dataset, ClusterHeightmapBoardPaths, ClusterHeightmapBoardDataset, TypeOfData, get_flat_image_paths, not

# include("DigisawIO.jl")
using DigisawIO

using Images, DataLoaders, Augmentor

using Transducers
using Glob

struct All end

struct Not{T}
    x::T
end

not(x) = Not(x)
isnot(_::Not{T}) where T = true
isnot(_) = false

const all = All()
Base.broadcastable(::All) = Ref(all)
Base.show(io::IO, ::All) = print(io, "all")

isall(::All) = true
isall(_) = false
Base.iterate(::All) = (all, all)
Base.iterate(::All, state) = nothing

const AllOr{T} = Union{All, T, Not{T}}
const ProposalDict{K, V} = AllOr{Dict{<:AllOr{K}, V}}

struct LogPathsBase{C, H, B}
    cluster::C
    heightmaps::H
    boards::B
end

const LogPaths = LogPathsBase{String, 
                        Dict{Integer, String}, 
                        Dict{Integer, Dict{BoardSide, String}}}

Base.broadcastable(proposal::LogPathsBase{C, H, B}) where {C, H, B} = Ref(proposal)

const DatasetPaths = Dict{LogDataset, Dict{Integer, LogPaths}}


get_iters(::All; dir=pwd(), mapf=identity) = map(s->(mapf(s), all), readdir(dir))
get_iters(pair::Pair; dir=pwd(), mapf=identity) = [pair]
get_iters(dict; dir=pwd(), mapf=identity) = dict

get_filter(::All) = _ -> true
get_filter(dict::AbstractDict{K, V}) where {K, V} = ∈(keys(dict))
get_filter(vals::AbstractVector) = ∈(vals)

get_filter2(::All) = (_, _) -> true
get_filter2(dict::AbstractDict{K, V}) where {K, V} = (x, y) -> get_filter(dict)(x) && get_filter(dict[x])(y)

function check_modality_file(parse, file, filter, regex)
    !isfile(file) && return nothing
    matches = match(regex, file)
    isnothing(matches) && return nothing
    parsed = parse(matches.captures...)
    (any(isnothing.(parsed)) || !filter(parsed...)) && return nothing
    return parsed
end

function parse_board(board_str)
    res = (parse_board_side ∘ uppercase)(board_str)
    return res
end

heightmap_suffix() = "" # "_flat_cubic"

function rglob(pattern, dir=".")
    result = []
    for (root, dirs, _) in walkdir(dir)
        for dir in dirs
            subdir = joinpath(root, dir)
            push!(result, glob(pattern, subdir)...)
        end
    end
    return result
end

propagate_log_proposal(_::All, _) = all
propagate_log_proposal(proposal::Not{T}, field) where T = not(getfield(proposal, field))
propagate_log_proposal(proposal, field) = getfield(proposal, field)


function find_paths(root_dir, proposal = all; heightmap_suffix=heightmap_suffix())::DatasetPaths
    start_path = pwd()
    cd(root_dir)
    result::DatasetPaths = DatasetPaths()

    heightmapstr = "log\\1_run\\2$heightmap_suffix."
    boardstr = raw"log\1_board\2_\3."
    clusterstr = raw"log\1_clusters."

    get_replace_pairs(replacements) = ["\\$i" => r for (i, r) ∈ enumerate(replacements)]
    insert_values(str, values...) =  replace(str, get_replace_pairs(values)...)

    glob_to_regex(str, n=5) = Regex(insert_values(replace(str, "."=>"\\."), repeat(["([^_^\\.]+)"], n)...))
    try_get_captures(x) = isnothing(x) ? nothing : x.captures
    get_matches(str, pattern) = match(glob_to_regex(pattern), str) |> try_get_captures

    all_datasets = map(parse_dataset, readdir())

    process_all(::All) = [(all, all)]
    process_all(x::Pair) = [x]
    process_all(x) = x

    
    in_all(x, y::Not{T}) where T = x ∉ y.x
    in_all(_, ::All) = true
    in_all(x, y) = x == y
    # in_all(x, y) = x ∈ y

    for (datasets, vals) ∈ process_all(proposal)
        for dataset ∈ all_datasets |> Filter(x->in_all(x, datasets))
            !isdir(string(dataset)) && continue
            cd(string(dataset))
            dataset_dict = Dict{Integer, LogPaths}()

            all_clusters = rglob(insert_values(clusterstr, "*") * "*") |> Map() do p
                matches = get_matches(basename(p), clusterstr)
                isnothing(matches) && return nothing
                log = tryparse(Int, matches[1])
                isnothing(log) && return nothing
                return (log, p |> abspath)
            end |> Filter(x->!isnothing(x)) |> collect
            all_boards = rglob(insert_values(boardstr, "*", "*", "*") * "*")|> Map() do p
                matches = get_matches(basename(p), boardstr)
                isnothing(matches) && return nothing
                log = tryparse(Int, matches[1])
                board = tryparse(Int, matches[2])
                side = parse_board(matches[3])
                isnothing.((log, board, side)) |> any && return nothing
                return (log, board, side, p |> abspath)
            end |> Filter(x->!isnothing(x)) |> collect
            all_heightmaps = rglob(insert_values(heightmapstr, "*", "*") * "*")|> Map() do p
                matches = get_matches(basename(p), heightmapstr)
                isnothing(matches) && return nothing
                log = tryparse(Int, matches[1])
                run = tryparse(Int, matches[2])
                isnothing.((log, run)) |> any && return nothing
                return (log, run, p |> abspath)
            end |> Filter(x->!isnothing(x)) |> collect

            log_iters = process_all(vals)
            for (logs, logproposal) ∈ log_iters
                for (log, cluster_path) ∈ all_clusters |> Filter(x->in_all(x[1], logs))
                    heightmaps = Dict{Integer, String}()
                    log_heightmaps = all_heightmaps |> Filter(x->in_all(x[1], log)) |> collect
                    for runs ∈ propagate_log_proposal(logproposal, :heightmaps)
                        for (log, run, heightmap_path) ∈ log_heightmaps |> Filter(x->in_all(x[2], runs))
                            heightmaps[run] = heightmap_path
                        end
                    end
                    log_boards = all_boards |> Filter(x->in_all(x[1], log)) |> collect
                    boards = Dict{Integer, Dict{BoardSide, String}}()
                    for (board, side_proposal) ∈ process_all(propagate_log_proposal(logproposal, :boards))
                        for side ∈ side_proposal
                            for (log, board_num, board_side, board_path) ∈ log_boards |> Filter(x->in_all(x[2], board) && in_all(x[3], side))
                                if !haskey(boards, board_num)
                                    boards[board_num] = Dict{BoardSide, String}()
                                end
                                boards[board_num][board_side] = board_path
                            end
                        end
                    end
                    dataset_dict[log] = LogPaths(cluster_path, heightmaps, boards)
                end
            end
            if !isempty(dataset_dict)
                result[dataset] = dataset_dict
            end
            cd("..")
        end
    end
    cd(start_path)
    return result
end

struct ClusterHeightmapBoardPaths
    cluster::AbstractString
    heightmap::AbstractString
    board::AbstractString
end
Base.broadcastable(x::ClusterHeightmapBoardPaths) = (x.cluster, x.heightmap, x.board)

mutable struct ClusterHeightmapBoardDataset{V<:AbstractVector{ClusterHeightmapBoardPaths}}
    triplets::V
    augment_cluster
    augment_heightmap
    augment_board
end

collect_triplets(node::AbstractString) = [node]
collect_triplets(node::LogPaths) = Iterators.map(x->ClusterHeightmapBoardPaths(x...), 
                                                Iterators.product([node.cluster], 
                                                    collect_triplets(node.heightmaps), 
                                                    collect_triplets(node.boards)))
collect_triplets(node::AbstractDict{K, V}) where {K, V} = Iterators.flatten(collect_triplets.(values(node)))
collect_triplets(node::T) where {T}= Iterators.flatten(collect_triplets.(node))

@enum TypeOfData cluster heightmap board
flatten_triplets(dataset::DatasetPaths) = flatten_triplets(dataset, ()) |> collect
flatten_triplets(node::AbstractString, prev_info) = [(prev_info..., node)]
function flatten_triplets(node::LogPaths, prev_info)
    heightmap_paths = flatten_triplets(node.heightmaps, (prev_info..., heightmap))
    board_paths = flatten_triplets(node.boards, (prev_info..., board))
    return [[(prev_info..., cluster, node.cluster)], heightmap_paths, board_paths] |> Cat()
end
flatten_triplets(node::AbstractDict{K, V}, prev_info) where {K, V} = collect(node) |> MapCat(kv->flatten_triplets(kv.second, (prev_info..., kv.first)))

get_flat_image_paths(root_dir::AbstractString, proposal=all; heightmap_suffix=heightmap_suffix()) = flatten_triplets(find_paths(root_dir, proposal, heightmap_suffix=heightmap_suffix))


augment_triple(x::ClusterHeightmapBoardDataset, (cluster, heightmap, board)) = 
            (x.augment_cluster(cluster), 
            x.augment_heightmap(heightmap), 
            x.augment_board(board))

create_dataset(root_dir::AbstractString, 
    proposal=all; 
    augment_cluster=NoOp(), 
    augment_heightmap=NoOp(),
    augment_board=NoOp(),
    heightmap_suffix=heightmap_suffix()) = ClusterHeightmapBoardDataset(
        find_paths(root_dir, proposal, heightmap_suffix=heightmap_suffix) |> collect_triplets |> collect,
        augment_cluster, augment_heightmap, augment_board
    )



DataLoaders.nobs(data::ClusterHeightmapBoardDataset) = length(data.triplets)
DataLoaders.getobs(data::ClusterHeightmapBoardDataset, i::Int) = (data.triplets[i], augment_triple(data, load.(data.triplets[i]))...)

(x::Augmentor.AbstractPipeline)(img) = augment(img, x)
(x::Augmentor.Operation)(img) = augment(img, x)


Base.length(x::ClusterHeightmapBoardDataset) = nobs(x)
Base.firstindex(x::ClusterHeightmapBoardDataset) = firstindex(x.triplets)
Base.lastindex(x::ClusterHeightmapBoardDataset) = lastindex(x.triplets)
Base.getindex(x::ClusterHeightmapBoardDataset, i::Int) = getobs(x, i)
Base.iterate(x::ClusterHeightmapBoardDataset, state=1) = state > length(x) ? nothing : (x[state], state+1)

end