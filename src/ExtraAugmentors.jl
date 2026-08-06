module ExtraAugmentors


using Augmentor

export CyclicShift, CyclicShiftY, AggregateThenMapFun, normalize_from_extrema

struct CyclicShift{T<:AbstractVector} <: Augmentor.ImageOperation
    shift::T
    dim::Integer
    function CyclicShift{T}(shift::T; dim=1) where {T<:AbstractVector{S} where S<:Integer}
        length(shift) > 0 || throw(ArgumentError("The number of different shifts passed to \"CyclicShift(...)\" must be non-zero"))
        dim > 0 || throw(ArgumentError("The dimension passed to \"CyclicShift(...)\" should be positive"))
        new{T}(shift, dim)
    end
end
CyclicShift(shift::T; dim=1) where {T<:AbstractVector} = CyclicShift{T}(shift; dim=dim)
CyclicShift(shift::Integer; dim=1) = CyclicShift(shift:shift; dim=dim)
CyclicShiftY(shift::T) where {T<:AbstractVector} = CyclicShift{T}(shift; dim=1)
CyclicShiftY(shift::Integer) = CyclicShift(shift:shift; dim=1)


Augmentor.randparam(op::CyclicShift, _) = Integer(Augmentor.safe_rand(op.shift))

@inline Augmentor.supports_eager(::Type{<:CyclicShift}) = true
@inline Augmentor.supports_stepview(::Type{<:CyclicShift}) = true
@inline Augmentor.supports_affine(::Type{<:CyclicShift}) = false

function get_shifted_idx(sz, param)
    shift = 1 + mod(param, sz)
    return vcat(shift:sz, 1:shift-1)
end

apply_indices(img, dim, param; other_dims=identity) = Tuple(map(x->x[1]==dim ? get_shifted_idx(size(img,dim), param) : other_dims(x[2]), Iterators.enumerate(axes(img))))


Augmentor.applyeager(op::CyclicShift, img::Array, param) = Augmentor.plain_array(getindex(img, apply_indices(img, op.dim, param)...))
Augmentor.applyeager(op::CyclicShift, img::AbstractArray, param) = Augmentor.plain_array(getindex(img, apply_indices(img, op.dim, param)...))
Augmentor.applylazy_fallback(op::CyclicShift, img::AbstractArray, param) = Augmentor.applystepview(op, img, param)

Augmentor.applystepview(op::CyclicShift, img::AbstractArray, param) = Augmentor.indirect_view(img, apply_indices(img, op.dim, param; other_dims=x->1:1:length(x)))


function Augmentor.showconstruction(io::IO, op::CyclicShift)
    print(io, typeof(op).name.name, "(", op.shift, ")")
end

function Base.show(io::IO, op::CyclicShift)
    if get(io, :compact, false)
        if length(op.shift) == 1
            print(io, "Cyclic shift ", first(op.shift), " pixels")
        else
            print(io, "Cyclic shift by s ∈ ", op.shift, " pixels")
        end
    else
        print(io, "Augmentor.")
        Augmentor.showconstruction(io, op)
    end
end


normalize_from_extrema(x, minmax) = (x - minmax[1]) / (minmax[2] - minmax[1])

struct AggregateThenMapFun{A,M} <: Augmentor.Operation
    aggfun::A
    mapfun::M
end

@inline Augmentor.supports_lazy(::Type{<:AggregateThenMapFun}) = true

function Augmentor.applyeager(op::AggregateThenMapFun, img::AbstractArray, param)
    agg = op.aggfun(img)
    Augmentor.plain_array(map(x -> op.mapfun(x, agg), img))
end

function Augmentor.applylazy(op::AggregateThenMapFun, img::AbstractArray, param)
    agg = op.aggfun(img)
    Augmentor.mappedarray(x -> op.mapfun(x, agg), img)
end

function showconstruction(io::IO, op::AggregateThenMapFun)
    print(io, nameof(typeof(op)), '(', op.aggfun, ", ", op.mapfun, ')')
end

function Base.show(io::IO, op::AggregateThenMapFun)
    if get(io, :compact, false)
        print(io, "Map result of \"", op.aggfun, "\" using \"", op.mapfun, "\" over image")
    else
        print(io, "Augmentor.")
        showconstruction(io, op)
    end
end

end
