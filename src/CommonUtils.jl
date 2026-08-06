module CommonUtils

export @fix, always

always(val) = (_...) -> val

function expr_replace_underscores(expr)
    function find_replace!(expr, gensyms)
        if expr == :_ 
            push!(gensyms, gensym("$(length(gensyms))"))
            return gensyms[end]
        elseif expr == :(_...)
            push!(gensyms, :($(gensym("args"))...))
            return gensyms[end]
        elseif expr isa Expr
            for i ∈ eachindex(expr.args)
                expr.args[i] = find_replace!(expr.args[i], gensyms)
            end
        end
        return expr
    end
    gensyms = []
    expr = find_replace!(deepcopy(expr), gensyms)
    return expr, gensyms
end

macro fix(expr)
    expr, gensyms = expr_replace_underscores(expr)
    kwargs = :($(gensym("kwargs"))...)
    if expr.head == :call
        if expr.args[2] isa Expr && expr.args[2].head == :parameters
            push!(expr.args[2].args, kwargs)
        else
            insert!(expr.args, 2, Expr(:parameters, kwargs))
        end
    end
    :(($(gensyms...), ; $kwargs ) -> $expr) |> esc
end

end
