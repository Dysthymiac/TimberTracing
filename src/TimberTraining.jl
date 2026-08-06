module TimberTraining

export UnetEncoder, UnetDecoder, LogBoardAutoencoder, get_log_autoencoder, get_board_autoencoder, cosine_similarity
export equivariant_term, barcode_term, equivariant_model_loss
export get_sum_barcode
export cat_skip
using Flux, Images, Statistics, Augmentor, LazyArrays, ImageFiltering, CUDA

# SkipWrapConcat(layer) = SkipConnection(Chain(first, layer), (mx, (_, x...)) -> (mx, mx, x...))
SkipWrapConcat(layer) = SkipConnection(Chain(first, layer), (mx, args) -> [mx, mx, args[2:end]...])

default_cat(x1, x2) = cat(x1, x2; dims=3)

struct PopLayer{T}
    action::T
end
PopLayer() = PopLayer(default_cat)
Flux.@layer PopLayer
# (pop::PopLayer{T})((x1, x2, xs...)) where {T} = (pop.action(x1, x2), xs...)
(pop::PopLayer{T})(args) where {T} = (pop.action(args[1], args[2]), args[3:end]...)

SkipWrapPop(layer; action=default_cat) = Chain(PopLayer(action), SkipWrap(layer))

# SkipWrap(layer) = SkipConnection(Chain(first, layer), (mx, (_, x...)) -> (mx, x...))
SkipWrap(layer) = SkipConnection(Chain(first, layer), (mx, args) -> [mx, args[2:end]...])

SkipWrap(layers...) = SkipWrap(Chain(layers...))


# norm_layer(chs::Integer, λ = identity;  momentum = .99f0, ϵ=0.01f0) = BatchNorm(ch[2], σ; momentum = momentum, ϵ=ϵ)
norm_layer(chs::Integer, λ = identity, G::Integer=chs==1 ? chs : min(chs ÷ 2, 32);  momentum = 0.1f0, ϵ=1f-5) = GroupNorm(chs, G, λ;  eps = ϵ, momentum = momentum)

ConvBlock(ch, filter=(3,3); pad=SamePad(), σ=relu) = Chain(Conv(filter, ch; pad=pad, init=Flux.kaiming_normal),
                                                            norm_layer(ch[2], σ))


function UnetBlock(ch, layers_num=2, filter=(3,3), pad=SamePad())
    ch0 = ch[1] => ch[2][1]
    ch1 = ch[2][1] => ch[2][1]
    ch2 = ch[2]
    return Chain(map(i -> ConvBlock(i == 1 ? ch0 : i==layers_num ? ch2 : ch1, filter; pad=pad), 1:layers_num)...)
end


PoolingBlock(window=(2,2)) = SkipWrap(MaxPool(window))
ConvPoolingBlock(ch, filter=(2,2), pad=0, σ=relu, scale=2) = Chain(Conv(filter, ch; pad=pad, init=Flux.kaiming_normal, stride=scale),
                                                                norm_layer(ch[2], σ))

UpsamplingBlock(scale=2) = x -> upsample_bilinear(x, (scale, scale))

ConvUpsamplingBlock(ch, filter=(2,2), pad=0, σ=relu, scale=2) = Chain(ConvTranspose(filter, ch; pad=pad, init=Flux.kaiming_normal, stride=scale),
                                                                        norm_layer(ch[2], σ))

cat_skip(_) = default_cat


struct UnetEncoder{T}
    encoder::T
end
Flux.@layer UnetEncoder
UnetEncoder(filter::Tuple{Integer, Integer}=(3, 3), blocks=4, block_size=2, start_filters=64, all_conv=false) = UnetEncoder(unet_encoder(filter, blocks, block_size, start_filters, all_conv))


struct UnetDecoder{T}
    decoder::T
end
Flux.@layer UnetDecoder
UnetDecoder(filter::Tuple{Integer, Integer}=(3, 3), blocks=4, block_size=2, start_filters=64, all_conv=false, skip_action=cat_skip) = UnetDecoder(unet_decoder(filter, blocks, block_size, start_filters, all_conv, skip_action))

struct LogBoardAutoencoder
    encoders
    decoders
end
Flux.@layer LogBoardAutoencoder
LogBoardAutoencoder(;encoders=1, decoders=1, filter=(3,3), blocks=4, block_size=2, start_filters=64, all_conv_encoder=false, all_conv_decoder=false, skip_action=cat_skip) =
    LogBoardAutoencoder(
        (UnetEncoder(filter, blocks, block_size, start_filters, all_conv_encoder) for _ ∈ 1:encoders) |> Tuple,
        (UnetDecoder(filter, blocks, block_size, start_filters, all_conv_decoder, skip_action) for _ ∈ 1:decoders) |> Tuple)
get_board_autoencoder(x::LogBoardAutoencoder) = get_board_decoder(x) ∘ get_board_encoder(x)
get_log_autoencoder(x::LogBoardAutoencoder) = get_log_decoder(x) ∘ get_log_encoder(x)
get_board_encoder(x::LogBoardAutoencoder) = x.encoders[end]
get_board_decoder(x::LogBoardAutoencoder) = x.decoders[end]
get_log_encoder(x::LogBoardAutoencoder) = x.encoders[1]
get_log_decoder(x::LogBoardAutoencoder) = x.decoders[1]


(unet::UnetEncoder)(x) = unet.encoder(x)
(unet::UnetDecoder)(x) = unet.decoder(x)
(x::LogBoardAutoencoder)(cluster, log, board) = (cluster, get_log_autoencoder(x)(log), get_board_autoencoder(x)(board))
(x::LogBoardAutoencoder)(tuple::Tuple) = x(tuple...)
encode(x::LogBoardAutoencoder, cluster, log, board) = (cluster, get_log_encoder(x)(log), get_board_encoder(x)(board))
decode(x::LogBoardAutoencoder, cluster, log, board) = (cluster, get_log_decoder(x)(log), get_board_decoder(x)(board))
encode(x::LogBoardAutoencoder, tuple::Tuple) = encode(x, tuple...)
decode(x::LogBoardAutoencoder, tuple::Tuple) = decode(x, tuple...)


function unet_encoder(filter, blocks, block_size, start_filters, all_conv)
    encoder_layers = []
    push!(encoder_layers, tuple)
    prev_f, next_f = 1, start_filters
    for i ∈ 1:blocks
        push!(encoder_layers,
            SkipWrapConcat(
                UnetBlock(prev_f=>next_f=>next_f, block_size, filter)
            ),
            all_conv ? SkipWrap(ConvPoolingBlock(next_f=>next_f)) : PoolingBlock())
        # println("$(repeat(" ", i))Pushing $next_f filters")
        prev_f, next_f = next_f, next_f * 2
    end
    push!(encoder_layers, SkipWrap(UnetBlock(prev_f=>next_f=>next_f, block_size, filter)))

    return Chain(encoder_layers...)
end

function unet_decoder(filter, blocks, block_size, start_filters, all_conv, skip_action = cat_skip)
    prev_f, next_f = 0, start_filters * 2^blocks
    decoder_layers = []
    for i ∈ 1:blocks
        prev_f, next_f = next_f, next_f ÷ 2
        push!(decoder_layers,
            SkipWrap(all_conv ? ConvUpsamplingBlock(prev_f=>prev_f) : UpsamplingBlock(), ConvBlock(prev_f=>next_f, filter)),
            SkipWrapPop(UnetBlock(2next_f=>next_f=>next_f, block_size, filter); action=skip_action(next_f => (next_f ÷ 2))))
    end
    # println("Clip final layer")
    println("Sigmoid final layer")
    push!(decoder_layers, first, ConvBlock(next_f=>1, (1, 1), pad=SamePad(), σ=sigmoid)) # sigmoid # Clip(0, 1)
    return Chain(decoder_layers...)
end

vecnorm(x; dims=1) = .√(sum(x.^2; dims=dims))
function vecnormalize(x; dims=1)
    norm = max.(vecnorm(x; dims=dims), 1f-6)
    return x ./ norm
end

cosine_similarity(x, y) = sum(vecnormalize(x) .* vecnormalize(y); dims=1)

get_sum_barcode(x) = sum(x; dims=1)[1, :, 1, :]


barcode_loss(x, y)::Float32 = 1f0 .- cosine_similarity(x, y) |> sum

barcode_loss(cluster, log, board)::Float32 = barcode_loss(cluster, log) .+ barcode_loss(cluster, board) .+ barcode_loss(board, log)
barcode_loss(tuple::Tuple)::Float32 = barcode_loss(tuple...)


struct EquivariantTerms{T1, T2, T3, T4}
    x::T1
    latent_x::T2
    Fx::T3
    TFx::T3
    barTFx::T4

    Tx::T1
    latent_Tx::T2
    FTx::T3
    barFTx::T4

end
function EquivariantTerms(model::LogBoardAutoencoder, transforms, x; get_barcode=get_sum_barcode)
    latent_x = encode(model, x)
    Fx = decode(model, latent_x)
    TFx = apply.(transforms, Fx)
    barTFx = get_barcode.(TFx)

    Tx = apply.(transforms, x)
    latent_Tx = encode(model, Tx)
    FTx = decode(model, latent_Tx)
    barFTx = get_barcode.(FTx)

    return EquivariantTerms(x, latent_x, Fx, TFx, barTFx, Tx, latent_Tx, FTx, barFTx)
end

preprocess_terms(t::Tuple{<:Real, Function}) = x-> Float32(t[1]) * t[2](x)
preprocess_terms(f) = x-> f(x)

function equivariant_model_loss(model::LogBoardAutoencoder, transforms, batch; terms=[barcode_term, equivariant_term], get_barcode=get_sum_barcode)::Float32
    vals = EquivariantTerms(model, transforms, batch; get_barcode=get_barcode)
    return mapreduce(f->f(vals), +, preprocess_terms.(terms); init=0f0)
end
equivariant_model_loss(model::LogBoardAutoencoder) = (batch, transforms) -> equivariant_model_loss(model, transforms, batch)

equivariant_term(TFx, FTx)::Float32 = mean(abs.(TFx[2] .- FTx[2]) .+ abs.(TFx[3] .- FTx[3]))

barcode_term(terms::EquivariantTerms)::Float32 = barcode_term(terms.barTFx)
equivariant_term(terms::EquivariantTerms)::Float32 = equivariant_term(terms.TFx, terms.FTx)

barcode_term(barcodes...)::Float32 = mean(.+(barcode_loss.(barcodes)...))

end
