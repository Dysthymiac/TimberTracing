# Serial precompilation avoids a Julia 1.11 deadlock in Flux's CUDA extensions.
get!(ENV, "JULIA_NUM_PRECOMPILE_TASKS", "1")

using TimberTracing            # the local package (run with `julia --project=.`)
using CUDA
using FileIO, Images
isinteractive() && using ImageView
using LazyArrays
using Flux, Augmentor, DataLoaders
import Zygote: withgradient, ignore
import ProgressMeter: Progress
import ProgressMeter
using Random, Statistics
using Parameters: @with_kw, type2dict
using JLSO
using Transducers
using Dates

# ---- Data locations -------------------------------------------------------
# Root directory that holds the `timber_tracing` dataset; model checkpoints and
# result images are written under `out_data/` in the same root. Set the
# TIMBER_TRACING_ROOT environment variable to your data root (see the README).
const DATA_ROOT = get(ENV, "TIMBER_TRACING_ROOT") do
    error("Set the TIMBER_TRACING_ROOT environment variable to your data root " *
          "(the folder containing `timber_tracing/`). See the README.")
end
dataset_path() = joinpath(DATA_ROOT, "timber_tracing")
out_path(parts...) = joinpath(DATA_ROOT, "out_data", parts...)

const all = All()
const WEIGHTED = weighted_aggregation  # used by report_old_metrics (test_old_metrics=true)

(x::Augmentor.AbstractPipeline)(img) = augment(img, x)
(x::Augmentor.Operation)(img) = augment(img, x)

val_boards = [2, 5, 8, 12, 15, 18, 22, 25, 28, 32, 35, 38, 42, 45, 48]
@with_kw mutable struct Args
    # 16 * 2 * (5, 14) = (160, 448)
    # 16 * 2 * (5, 16) = (160, 512)
    get_barcode = get_sum_barcode
    target_size::Tuple{Integer, Integer} = isinteractive() ? (80, 256) : (80, 1024) # (160, 512) # (160, 512) # (160, 512)# (80, 256)
    η::Float64 = 0.001     # learning rate
    opt = Flux.Adam(η) #  Momentum(η) # NADAM(η) #  (Flux 0.16 explicit-optimiser rule)
    batchsize::Int = isinteractive() ? 1 : 3    # batch size
    epochs::Int = isinteractive() ? 1 : 50       # number of epochs
    use_cuda::Bool = true  # use gpu, if cuda is available
    # Which dataset folders (under DATA_ROOT/timber_tracing/) to train / validate on.
    # Keys are folder names — put your own data in folders and name them here, e.g.
    # Dict("train"=>all) / Dict("val"=>all). `all` means "use all logs in that folder".
    # See the "Data format" section of the README.
    train_proposal = Dict("HONKALAHTI2018"=>all, "DIGISAW2021"=>all)
    val_proposal = Dict("HONKALAHTI2019"=>all)
    # Split by log number within one folder instead:
    # train_proposal = Dict("HONKALAHTIOLD"=>not(val_boards)=>all)
    # val_proposal = Dict("HONKALAHTIOLD"=>val_boards.=>all)
    model_kws = Dict(:encoders=>2, :decoders=>1, :skip_action=>cat_skip, :start_filters=>32)
    heightmap_suffix = ""
    loss_terms = [equivariant_term, barcode_term] # [equivariant_term, barcode_term, F_diff_relu_term, l2_norm_term]
    io::IO = stdout
    model_path::AbstractString = out_path("timber_tracing_models")
    model_name::AbstractString = "Unet_barcode"
    save_model::Bool = false
    load_model::Union{Nothing, AbstractString} = nothing
    test_old_metrics::Bool = false
end


function get_augmentors(target_size::Tuple{Integer, Integer})
    augment_start = ConvertEltype(Gray{Float32}) |> Resize(target_size)
    augment_end = SplitChannels() |> PermuteDims((2, 3, 1))
    augment_cluster = augment_start |> augment_end
    augment_heightmap = augment_start |> FlipY(0.5) |>
                        augment_end |> CyclicShiftY(1:target_size[1])
    normalize_invert_from_extrema(x, minmax) = 1 - ((x - minmax[1]) / (minmax[2] - minmax[1]))
    invert_augment = ExtraAugmentors.AggregateThenMapFun(extrema, normalize_invert_from_extrema)  # MapFun(invert_img)
    augment_board = augment_start |> FlipY(0.5) |>
                    augment_end |> invert_augment
    validation_augment = augment_start |> augment_end
    val_cluster=validation_augment
    val_heightmap=validation_augment
    val_board=validation_augment |> invert_augment
    to_image = PermuteDims((3, 1, 2)) |> CombineChannels(Gray{Float32})

    return augment_cluster, augment_heightmap, augment_board, val_cluster, val_heightmap, val_board, to_image
end

process_image((_, path), model, augmentors, device, get_barcode) = path |> load |> augmentors |> Flux.unsqueeze(dims=4) |> device |> model |> get_barcode |> vec
process_image(model, augmentors, device, get_barcode) = x -> process_image(x, model, augmentors, device, get_barcode)

function process_flat_paths(flat_paths, data_type, process_func)
    objs = flat_paths |> Filter(x->x[3] == data_type) |> Map(x->(x[1:2], x[end])) |> collect
    barcodes = objs |> Map(process_func) |> collect
    classes = objs |> Map(first) |> collect
    return barcodes, classes
end

function calculate_matches(model, flat_paths, device, log_augmentors, board_augmentors, get_barcode)
    board_autoencoder = get_board_autoencoder(model)
    log_autoencoder = get_log_autoencoder(model)

    process_board = process_image(board_autoencoder, board_augmentors, device, get_barcode)
    process_log = process_image(log_autoencoder, log_augmentors, device, get_barcode)
    
    board_barcodes, board_classes = process_flat_paths(flat_paths, TimberDatasets.board, process_board)
    log_barcodes, log_classes = process_flat_paths(flat_paths, TimberDatasets.heightmap, process_log)
    
    dist_mat = [cosine_similarity(board, log) |> cpu |> only for board ∈ board_barcodes, log ∈ log_barcodes]
    matches = argmax(dist_mat; dims=2)
    
    real_classes = board_classes
    predicted = vec(log_classes[getindex.(matches, 2)])

    return predicted, real_classes
end

function report_metrics(predicted, real_classes; io=stdout)
    println("Predicted = ", predicted)
    println("Real = ", real_classes)
    pretty, _ = classification_report(predicted, real_classes; io=io)
    return pretty
end

function report_old_metrics(predicted, real_classes; io=stdout)
    agg = [AggregationSet("1-10", WEIGHTED, predicate=x->x[2] ≤ 10), 
    AggregationSet("11-20", WEIGHTED, predicate=x-> 11 ≤ x[2] ≤ 20), 
    AggregationSet("21-30", WEIGHTED, predicate=x-> 21 ≤ x[2] ≤ 30), 
    AggregationSet("31-40", WEIGHTED, predicate=x-> 31 ≤ x[2] ≤ 40), 
    AggregationSet("41-50", WEIGHTED, predicate=x-> 41 ≤ x[2] ≤ 50), 
    AggregationSet("1-30", WEIGHTED, predicate=x-> 1 ≤ x[2] ≤ 30), 
    AggregationSet("31-50", WEIGHTED, predicate=x-> 31 ≤ x[2] ≤ 50), 
    AggregationSet("1-50", WEIGHTED, predicate=x-> 1 ≤ x[2] ≤ 50), 
    AggregationSet("Validation", WEIGHTED, predicate=x-> x[2] ∈ val_boards)]
    println("Predicted = ", predicted)
    println("Real = ", real_classes)
    pretty, _ = classification_report(predicted, real_classes; show_per_class=false, aggregation_sets=agg, io=io)
    return pretty
end

function setup_device(; kws...)
    @unpack_Args Args(; kws...)
    if CUDA.functional() && use_cuda
        println(io, "Training on CUDA GPU")
        CUDA.allowscalar(false)
        return gpu
    else
        println(io, "Training on CPU")
        return cpu
    end
end

function create_model(; kws...)
    @unpack_Args Args(; kws...)
    if isnothing(load_model)
        println(io, "Creating model...")
        return LogBoardAutoencoder(; model_kws...)
    else
        return JLSO.load(load_model)[:model]
    end
end

function print_augmentors(augment_cluster, augment_heightmap, augment_board, val_cluster, val_heightmap, val_board, to_image; io=stdout)
    println(io, "Cluster augmentation: ", augment_cluster)
    println(io, "Heightmap augmentation: ", augment_heightmap)
    println(io, "Board augmentation: ", augment_board)
    println(io, "Validation augmentation: ", val_heightmap)
    println(io, "Validation board augmentation: ", val_board)
    println(io, "To image: ", to_image)
end


function create_dataloaders(augment_cluster, augment_heightmap, augment_board, val_cluster, val_heightmap, val_board, to_image; kws...)
    @unpack_Args Args(; kws...)
    train_dataset = create_dataset(dataset_path(), train_proposal;
                                    augment_cluster=augment_cluster, 
                                    augment_heightmap=augment_heightmap,
                                    augment_board=augment_board, heightmap_suffix=heightmap_suffix)
    val_dataset = create_dataset(dataset_path(), val_proposal;
                                    augment_cluster=val_cluster, 
                                    augment_heightmap=val_heightmap,
                                    augment_board=val_board, heightmap_suffix=heightmap_suffix)
    
    train_loader = DataLoader(train_dataset, batchsize, collate = true)
    val_loader = DataLoader(val_dataset, batchsize, collate = true)
    return train_dataset, train_loader, val_loader
end

generate_showvalues(type, i, loss) = () -> [(:type, type), (:loss,loss), (:epoch, i)]

function epoch(iter_func, loader, type, epoch; io=stdout)
    p = Progress(length(loader), showspeed=true, output=io)
    losses = Float32[]
    i = 0
    update_func() = ProgressMeter.update!(p, i; showvalues = generate_showvalues(type, epoch, mean(losses)))
    callback = isinteractive() ? update_func : Flux.throttle(update_func, 1)
    println(io, "start...")
    for (_, batch...) ∈ loader
        loss = iter_func(batch)
        push!(losses, loss)
        i += 1
        callback()
    end
    ProgressMeter.finish!(p)
    println(io, "...finish")
    return losses
end

function multireverse(input::AbstractArray{T, N}; dims::Union{Integer, Tuple{Integer, Vararg{Integer}}}) where {T, N}
    typeof(dims) <: Integer && return reverse(input; dims=dims)
    output = input
    for dim ∈ dims
        output = reverse(output; dims=dim)
    end
    return output
end

function train_model(iteration; kws...)
    @unpack_Args Args(; kws...)
    foreach(x -> println(x[1], " => ", x[2]), Args(; kws...) |> type2dict |> collect)
    ##########
    println(io, "Setting up the device...")
    device = setup_device(; kws...)
    model = create_model(; kws...) |> device
    opt_state = Flux.setup(opt, model)
    ##########
    println(io, "Preparing training...")
    
    println(io, "Optimizer: ", opt)
    ##########
    println(io, "Defining augmentations...")
    augmentors = get_augmentors(target_size)
    print_augmentors(augmentors...; io=io)
    ###########
    println(io, "Creating datasets...")
    # println(io, "Training dataset:", train_proposal)
    # println(io, "Validation dataset:", val_proposal)

    train_dataset, train_loader, val_loader = create_dataloaders(augmentors...; kws...)
    
    ############
    println("Starting training...")
    mean_loss = 0
    early_stopping = Flux.early_stopping(2; init_score = 3, min_dist = 1f-3) do
        println(io, "Testing early stopping... ")
        println(io, "Loss: ", mean_loss)
        mean_loss
    end

    # loss_func(batch) = model_loss(model, batch)

    create_flip() = let dim=rand([1, 2, (1,2)]); return x->multireverse(x; dims=dim) end
    create_circshift() = let shift=rand(1:target_size[1]); return x->circshift(x, shift) end
    # create_transforms() = let flip=create_flip(); return x -> apply.((flip, create_circshift() ∘ flip, flip), x) end
    
    # create_transforms() = let flip=create_flip(); return (flip, create_circshift() ∘ flip, flip) end
    create_transforms() = let transform=create_circshift() ∘ create_flip(); return (transform, transform, transform) end
# @code_warntype 
    loss_func(m, batch, transforms) = TimberTraining.equivariant_model_loss(m, transforms, batch; get_barcode=get_barcode, terms=loss_terms)
    preprocess_batch(batch) = device.(batch)
    function train_func(batch)
        device_batch = preprocess_batch(batch)
        transforms = create_transforms()
        loss, gs = Flux.withgradient(m -> loss_func(m, device_batch, transforms), model)
        Flux.update!(opt_state, model, gs[1])
        return loss
    end
    val_func(batch) = loss_func(model, preprocess_batch(batch), create_transforms())
    best_val_loss = 1e100
    id = get(ENV, "SLURM_JOB_ID", "result")
    mkpath(out_path("best_models"))
    model_out_path = out_path("best_models", """$(id)_$iteration.jlso""")

    for i ∈ 1:epochs
        println(io, "Iteration ", iteration)
        println("Starting epoch ", i, "...")

        Flux.testmode!(model, false)
        losses = epoch(train_func, train_loader, "train", i; io=io)
        # losses = [1]
        shuffle!(train_dataset.triplets)

        println(io, '◙'^30)
        println(io, "TRAINING LOSS: ", mean(losses))
        println(io, '◙'^30)

        Flux.testmode!(model, true)
        mean_loss = mean(epoch(val_func, val_loader, "val", i; io=io))
        # break
        println(io, '◙'^30)
        println(io, "VALIDATION LOSS: ", mean_loss)
        println(io, '◙'^30)
        if mean_loss < best_val_loss
            best_val_loss = mean_loss
            JLSO.save(model_out_path, :model => (model |> cpu))
        end

        if early_stopping()
            println(io, "No improvement for 3 epochs, stopping early")
            break
        end
    end
    model = JLSO.load(model_out_path)[:model] |> gpu
    # model = JLD.load(model_out_path) |> gpu
    predicted, real = calculate_matches(model, 
        get_flat_image_paths(dataset_path(), val_proposal, heightmap_suffix=heightmap_suffix),
        device, augmentors[5:6]..., get_barcode)
    report_metrics(predicted, real; io=io)

    if test_old_metrics
        predicted, real = calculate_matches(model, 
            get_flat_image_paths(dataset_path(), Dict("HONKALAHTIOLD"=>all), heightmap_suffix=heightmap_suffix),
            device, augmentors[5:6]..., get_barcode)
        report_old_metrics(predicted, real; io=io)
    end

    model = model |> cpu
    
    if save_model
        mkpath(model_path)
        file_path = joinpath(model_path, """$(model_name)_$(Dates.format(now(), dateformat"H-M-S_d.m.Y")).jlso""")
        JLSO.save(file_path, :model=>model)
    end

    #############

    to_image = augmentors[end]
    for (paths, cluster, heightmap, board) ∈ first(val_loader, 5)
        full_to_image = to_image |> ExtraAugmentors.AggregateThenMapFun(extrema, normalize_from_extrema)
        cluster_img = cluster[:,:,:,1] |> to_image
        heightmap_img = heightmap[:,:,:,1] |> to_image
        barcode = get_log_autoencoder(model)(heightmap)
        heightmap_barcode_img = barcode[:,:,:,1] |> full_to_image

        board_img = board[:,:,:,1] |> to_image
        barcode = get_board_autoencoder(model)(board)
        board_barcode_img = barcode[:,:,:,1] |> full_to_image
        heightmap_board_barcode = [heightmap_img; heightmap_barcode_img; cluster_img[1:20, :]; board_barcode_img; board_img]
        println(paths)
        if isinteractive()
            imshow(heightmap_board_barcode)
        else
            id = get(ENV, "SLURM_JOB_ID", "result")
            mkpath(out_path("barcodes_result", id))
            name = """heightmap_board_barcode_$(iteration)_$(Dates.format(now(), dateformat"H-M-S_d.m.Y")).png"""
            out_image_path = out_path("barcodes_result", id, name)
            println()
            println("Out image: ", out_image_path)
            println()
            save(out_image_path, heightmap_board_barcode)
        end
    end
    return model
end


iters = isinteractive() ? 1 : 5
for i ∈ 1:iters
    model = train_model(i)
    println("finished ", i)
end
