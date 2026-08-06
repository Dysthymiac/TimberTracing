module DigisawIO

using MLStyle

export parse_board_side, parse_dataset

macro enum_data(typ, def_variants)
    eval(:(@MLStyle.data $typ $def_variants))
    add_subtypes(typ, def_variants) # __module__
end

function add_subtypes(typ::Any, def_variants::Expr)
    ctors = Vector{Symbol}()
    for variant in def_variants.args
        @switch variant begin
            @case :($(case::Symbol)::$ret_ty) || (case::Symbol)
            ctor_name = Symbol(case, "'s constructor")
            push!(ctors, ctor_name)
            continue
            @case ::Any
            continue
        end
    end
    return esc(quote
        get_subtypes(::Type{$typ}) = $(Tuple(eval.(ctors)))
        for subtype ∈ get_subtypes($typ)
            MLStyle.is_enum(::Type{subtype()}) = true
        end
        Base.Pair(a::$typ, b::B) where {B} = Pair{$typ, B}(a, b)
    end)
end

parse_enum(::Type{T}, s::AbstractString) where {T} = reduce((acc, x)->string(x())==s ? x() : acc, get_subtypes(T); init=nothing)

@enum_data LogDataset begin
    HONKALAHTI2018
    HONKALAHTI2019
    TANDEM2020
    TUKKI
    LANKKU
    DIGISAW2021
    HONKALAHTIOLD
end
Base.isless(a::LogDataset, b::LogDataset) = isless(string(a), string(b))
parse_dataset(s::AbstractString) = parse_enum(LogDataset, s)

@enum_data BoardSide begin
    UPPER
    LOWER
end
parse_board_side(s::AbstractString) = parse_enum(BoardSide, s)


macro export_enums(types...)
    esc(quote
        eval(Expr(:export, $types...))
        for type ∈ eval.($types)
            subtypes = map(x->Symbol(x()), get_subtypes(type))
            eval(Expr(:export, subtypes...))
        end
    end)
end

@export_enums LogDataset BoardSide

end
