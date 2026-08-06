module ClassificationMetrics

export get_label_set, confusion_matrix, prediction_results, AggregationType, rename_labels
export IoU, support, true_false_ratio, precision, recall, Fᵦ_score, F₁_score, fall_out, specificity, sensitivity, true_positive_rate, false_positive_rate, false_negative_rate, miss_rate, jaccard
export classification_report, all_aggregation
export AggregationSet
export macro_aggregation, micro_aggregation, weighted_aggregation, no_aggregation

using Statistics, OneHotArrays, LinearAlgebra, PrettyTables, Printf
using CommonUtils
import OneHotArrays: onehotbatch, OneHotLike
import Base: show, getindex, vcat

struct ConfusionMatrix{T1, T2}
    label_set::T1
    matrix::T2
end

function create_confusion_matrix(predicted, actual, label_set)
    m = length(label_set)
    reverse_index = Dict(x=>i for (i, x) ∈ enumerate(label_set))
    cm = zeros(Int64, m, m)
    for (p, a) ∈ zip(predicted, actual)
        cm[reverse_index[p], reverse_index[a]] += 1
    end
    return cm
end

confusion_matrix(predicted::OneHotLike, actual::OneHotLike, label_set) = ConfusionMatrix(label_set, predicted * transpose(actual))
confusion_matrix(predicted, actual, ::Nothing=nothing; sort_labels=false) = confusion_matrix(predicted,actual, get_label_set(predicted,actual); sort_labels=sort_labels)
function confusion_matrix(predicted,actual, label_set; sort_labels=false) 
    if sort_labels 
        label_set = sort(label_set)
    end
    return ConfusionMatrix(label_set, create_confusion_matrix(predicted,actual, label_set))
end

struct PredictionResults{T1, T2}
    label_set::T1
    results::T2
end

rename_labels(pr::PredictionResults, new_labels) = PredictionResults(new_labels, pr.results)

prediction_results(cm::ConfusionMatrix) = PredictionResults(cm.label_set, prediction_results(cm.matrix))
prediction_results(predicted,actual; label_set=nothing, sort_labels=false) = prediction_results(confusion_matrix(predicted, actual, label_set, sort_labels=sort_labels))
show(io::IO, cm::PredictionResults) =
    pretty_table(io, cm.results;
        column_labels=["TP", "FN", "FP", "TN"],
        row_labels=cm.label_set,
        alignment=:r,
        fit_table_in_display_horizontally=false, fit_table_in_display_vertically=false)

show(io::IO, cm::ConfusionMatrix) =
    pretty_table(io, cm.matrix;
        column_labels=cm.label_set,
        row_labels=cm.label_set,
        alignment=:c,
        stubhead_label="Predictions ↓",
        highlighters =
        [TextHighlighter((_, i, j) -> i==j, Crayon(foreground = :green)),
        TextHighlighter((_, i, j) -> i != j, Crayon(foreground = :red))],
        fit_table_in_display_horizontally=false, fit_table_in_display_vertically=false)


get_label_set(x) = unique(x)
get_label_set(x...) = ∪(get_label_set.(x)...)

onehot_prepare(args...; label_set) = map(x->onehotbatch(x, label_set), args)

function prediction_results(confusion_matrix) 
    TP = diag(confusion_matrix)
    FN = vec(sum(confusion_matrix; dims=1)) .- TP
    FP = vec(sum(confusion_matrix; dims=2)) .- TP
    TN = sum(confusion_matrix) .- TP .- FN .- FP
    return hcat(TP, FN, FP, TN)
end

Base.getindex(pr::PredictionResults, I) = let res=pr.results[I, :]; PredictionResults(pr.label_set[I], ndims(res)==1 ? transpose(res) : res); end
Base.vcat(pr1::PredictionResults, pr2::PredictionResults) = PredictionResults(vcat(pr1.label_set, pr2.label_set), vcat(pr1.results, pr2.results))
Base.vcat(pr1::PredictionResults, prs::PredictionResults...) = reduce(vcat, prs; init=pr1)


get_TP(pr::PredictionResults) = @view pr.results[:, 1]
get_FN(pr::PredictionResults) = @view pr.results[:, 2]
get_FP(pr::PredictionResults) = @view pr.results[:, 3]
get_TN(pr::PredictionResults) = @view pr.results[:, 4]
get_support(pr::PredictionResults) = round.(Integer, sum(@view(pr.results[:, 1:2]), dims=2))

nonzero_mean(vals) = mean(vals[abs.(vals) .> eps()])

no_aggregation(func, vals...) = func.(vals...)
micro_aggregation(func, vals...) = nonzero_mean(func.(vals...))
macro_aggregation(func, vals...) = func(sum.(vals)...)
weighted_aggregation(func, vals...; weights=nothing) = let weights=@something weights vals[1].+vals[2]; sum(weights.*func.(vals...) ./ sum(weights)) end

default_aggregation = weighted_aggregation

apply_metric(TP, FN, FP, TN; aggregate=default_aggregation, metric=accuracy, kws...) = 
            aggregate(@fix(metric(_...; kws...)), TP, FN, FP, TN)
apply_metric(pr::PredictionResults; aggregate=default_aggregation, metric=accuracy, kws...) = 
            apply_metric(eachcol(pr.results)...; aggregate=aggregate, metric=metric, kws...)
apply_metric(confusion_matrix::ConfusionMatrix; aggregate=default_aggregation, metric=accuracy, kws...) = 
            apply_metric(prediction_results(confusion_matrix); aggregate=aggregate, metric=metric, kws...)
apply_metric(predicted, classes; aggregate=default_aggregation, metric=accuracy, label_set=nothing, sort_labels=false, kws...) = 
            apply_metric(confusion_matrix(predicted, classes, label_set; sort_labels=sort_labels); aggregate=aggregate, metric=metric, kws...)

macro metric(funcs...)
    exprs = (esc(quote
        $(func)(TP::T, FN::T, FP::T, TN::T; aggregate=default_aggregation, kws...) where {T<:AbstractArray} = 
                         apply_metric(TP, FN, FP, TN; aggregate=aggregate, metric=$(func), kws...)
        $(func)(pr::PredictionResults; aggregate=default_aggregation, kws...) = apply_metric(pr; aggregate=aggregate, metric=$(func), kws...)
        $(func)(cm::ConfusionMatrix; aggregate=default_aggregation, kws...) = apply_metric(cm; aggregate=aggregate, metric=$(func), kws...)
        $(func)(predicted, classes; aggregate=default_aggregation, label_set=nothing, sort_labels=false, kws...) = 
                apply_metric(predicted, classes; aggregate=aggregate, metric=$(func), label_set=label_set, sort_labels=sort_labels, kws...)
    end) for func ∈ funcs)
    return Expr(:block, exprs...)
end

safe_div(a, b) = b==zero(b) ? zero(b) : a/b

binary_accuracy(TP, FN, FP, TN) = safe_div(TP + TN, TP + FP + TN + FN)
precision(TP, FN, FP, TN) = safe_div(TP, TP + FP)
recall(TP, FN, FP, TN) = safe_div(TP, TP + FN)
fall_out(TP, FN, FP, TN) = safe_div(FP, FP + TN)
miss_rate(TP, FN, FP, TN) = safe_div(FN, TP + FN)
specificity(TP, FN, FP, TN) = safe_div(TN, FP + TN)
jaccard(TP, FN, FP, TN) = safe_div(TP, TP + FN + FP)
IoU(args...; kwargs...) = jaccard(args...; kwargs...)
support(TP, FN, FP, TN) = TP + FN
true_false_ratio(TP, FN, FP, TN) = safe_div(TP + FN, TP + FP + TN + FN)
Fᵦ_score(TP, FN, FP, TN; β=1) = let β²=β^2; safe_div((1 + β²)TP, (1 + β²)TP + β²*FN + FP) end
@generated Fᵦ_score(β::Real) = quote 
    β = eval(β)
    rounded_β = isinteger(β) ? β : round(β; digits=2)
    name = Symbol(replace("F$(rounded_β)_score", "."=>"_"))
    eval(:($name(args...; kwargs...) = Fᵦ_score(args...; kwargs..., β = $β)))
end
F₁_score(args...; kwargs...) = Fᵦ_score(args...; kwargs..., β=1)

sensitivity(args...; kwargs...) = recall(args...; kwargs...)
true_positive_rate(args...; kwargs...) = recall(args...; kwargs...)
false_positive_rate(args...; kwargs...) = fall_out(args...; kwargs...)
false_negative_rate(args...; kwargs...) = miss_rate(args...; kwargs...)

@metric binary_accuracy precision recall fall_out specificity Fᵦ_score miss_rate jaccard support true_false_ratio

accuracy(predicted, actual) = mean(predicted .== actual)

all_aggregation() = [AggregationSet("Micro", micro_aggregation), AggregationSet("Macro", macro_aggregation), AggregationSet("Weighted", weighted_aggregation)]

struct AggregationSet{T1, T2, T3}
    name::T1
    predicate::T2
    aggregation::T3
end

AggregationSet(name, aggregation=default_aggregation; predicate=always(true)) = AggregationSet(name, predicate, aggregation)


classification_report(predicted, actual; 
    label_set=nothing, sort_labels=true,
    kws...) = classification_report(prediction_results(predicted, actual; label_set=label_set, sort_labels=sort_labels); kws...)


function classification_report(prediction_results::PredictionResults; 
    metrics=[precision, recall, F₁_score], 
    show_per_class=true, 
    aggregation_sets = all_aggregation(), io=stdout, include_support=true, backend=:text)

    pretty_name(m) = uppercasefirst(replace(string(m), r"(\d+)_(\d+)"=>s"\1.\2", "_"=>" "))
    names = pretty_name.(metrics)
    if include_support
        names = push!(names, "Support")
    end

    full_labels = [ag.name for ag ∈ aggregation_sets]
    
    support = get_support(prediction_results)

    all_inds = [findall(ag.predicate.(prediction_results.label_set)) for ag ∈ aggregation_sets]
    
    support_col = [sum(support[inds]) for inds ∈ all_inds]
    
    agg_types = [ag.aggregation for ag ∈ aggregation_sets]

    full_results = [map(((inds, agg),)->m(prediction_results[inds]; aggregate=agg), zip(all_inds, agg_types)) for m ∈ metrics]

    if show_per_class
        full_labels = vcat(prediction_results.label_set, full_labels)
        support_col = vcat(support, support_col)
        full_results = vcat.(map(m->m(prediction_results; aggregate=no_aggregation), metrics), full_results)
    end
    table = reduce(hcat, full_results)
    if include_support
        table = hcat(table, support_col)
    end
    # PrettyTables 3.x: disable display cropping only for the text backend.
    optional_kws = backend == :text ?
        (; fit_table_in_display_horizontally=false, fit_table_in_display_vertically=false) :
        (;)
    pretty = pretty_table(io, table;
            column_labels=names,
            row_labels=full_labels,
            formatters=[(v,i,j)-> (j==size(table,2) && include_support) ? round(Integer, v) : @sprintf("%.4f", v)],
            backend=backend,
            optional_kws...)
    return pretty, table
end


end