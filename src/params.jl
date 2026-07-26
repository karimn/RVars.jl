# MCMC samplers have no notion of an array-valued parameter: a model declaring
# `x[trial, patient]` reaches the chain as one flat column per element, labelled
# `Symbol("x[1,1]")`, `Symbol("x[2,1]")`, ... . The code below undoes that flattening,
# turning the flat named vector RVar that `from_chains` builds into one RVar per model
# parameter, each with the shape the model declared.

# Split a flattened element name into its base parameter and integer index:
# `Symbol("x[2,3]") -> (:x, (2, 3))`. A name that is not a base followed by a bracketed,
# all-integer index — a plain scalar like `:sigma`, or an exotic label like
# `Symbol("x[a]")` — comes back with `nothing` for the index and is left alone as a
# scalar, so no name is ever silently reinterpreted as something it isn't.
function _parse_param_name(s::Symbol)
    str = String(s)
    (lastindex(str) > 2 && last(str) == ']') || return (s, nothing)
    open_at = findfirst('[', str)
    # `open_at == 1` means there is no base name to group under, e.g. `Symbol("[1]")`.
    (open_at === nothing || open_at == 1) && return (s, nothing)
    base = SubString(str, 1, prevind(str, open_at))
    inner = SubString(str, nextind(str, open_at), prevind(str, lastindex(str)))
    isempty(inner) && return (s, nothing)
    parts = split(inner, ',')
    idx = Vector{Int}(undef, length(parts))
    for (k, p) in enumerate(parts)
        v = tryparse(Int, strip(p))
        v === nothing && return (s, nothing)
        idx[k] = v
    end
    return (Symbol(base), (idx...,))
end

# Bucket flat column positions by base parameter name, keeping first-appearance order so
# the resulting NamedTuple lists parameters in the order the sampler reported them.
function _group_param_names(nms::AbstractVector{Symbol})
    order = Symbol[]
    cols = Dict{Symbol, Vector{Int}}()
    locs = Dict{Symbol, Vector{Union{Nothing, Tuple{Vararg{Int}}}}}()
    for (j, nm) in enumerate(nms)
        base, idx = _parse_param_name(nm)
        if !haskey(cols, base)
            push!(order, base)
            cols[base] = Int[]
            locs[base] = Union{Nothing, Tuple{Vararg{Int}}}[]
        end
        push!(cols[base], j)
        push!(locs[base], idx)
    end
    return order, cols, locs
end

_idx_str(loc) = join(loc, ",")

# Gather the columns belonging to one base name into a single RVar. `store` is the
# (ndraws, nvars) flat draw matrix; `cols[k]` is the column holding element `locs[k]`.
# Placement is driven entirely by the parsed indices, so the result does not depend on
# the order the sampler happened to emit the elements in.
function _assemble_param(base::Symbol, store::AbstractMatrix{T}, cols::Vector{Int},
                         locs::Vector, nchains::Int) where {T}
    if any(isnothing, locs)
        # An unindexed name is a scalar parameter, and a scalar owns its name outright:
        # it can neither be repeated nor share a base with indexed elements.
        length(locs) == 1 || error("Parameter :$base appears $(length(locs)) times; an " *
                                   "unindexed name must be unique and cannot be mixed " *
                                   "with indexed entries")
        data = store[:, cols[1]]
        return RVar{T, 0, typeof(data)}(data, nchains)
    end

    ranks = unique(map(length, locs))
    if length(ranks) != 1
        rank_str = join(sort(ranks), " and ")
        error("Parameter :$base has entries of differing dimensionality ($rank_str indices)")
    end
    M = ranks[1]
    for loc in locs
        if !all(>=(1), loc)
            error("Parameter :$base has non-positive index [$(_idx_str(loc))]; " *
                  "only 1-based indices are supported")
        end
    end
    dims = ntuple(d -> maximum(loc -> loc[d], locs), M)

    out = similar(store, T, (size(store, 1), dims...))
    filled = falses(dims)
    for (k, loc) in enumerate(locs)
        filled[loc...] && error("Parameter :$base has duplicate index [$(_idx_str(loc))]")
        filled[loc...] = true
        out[:, loc...] = @view store[:, cols[k]]
    end
    if !all(filled)
        # findfirst over a BitVector yields an Int rather than a CartesianIndex, so walk
        # CartesianIndices instead to keep one code path for every rank.
        gap = first(I for I in CartesianIndices(filled) if !filled[I])
        shape_str = join(dims, "x")
        error("Parameter :$base is missing index [$(_idx_str(Tuple(gap)))]: got " *
              "$(length(locs)) entries, but its indices span a $shape_str array " *
              "needing $(prod(dims))")
    end
    return RVar{T, M, typeof(out)}(out, nchains)
end

"""
    rvars(x::RVar)
    rvars(array, param_names)

Regroup per-element parameter draws into one random variable per model parameter,
restoring the shape the model declared.

Samplers flatten an array-valued parameter into one entry per element, named `x[i]`,
`x[i,j]`, and so on. `rvars` reads those names and reverses the flattening, returning a
`NamedTuple` keyed by base parameter name: a scalar parameter becomes a `RVar{T, 0}`, a
vector parameter a `RVar{T, 1}`, an `(m, n)` matrix parameter a `RVar{T, 2}` of
`size == (m, n)`, and so on. `nchains` is carried through unchanged, and elements are
placed by their parsed indices rather than by column order.

`x` must be a vector random variable carrying names (see [`from_chains`](@ref)); passing
`array` and `param_names` builds that value first. Names without a bracketed integer index
are kept as scalars under their own name.

Throws if a parameter's element names do not describe a complete, 1-based, rectangular
array — a missing or duplicated index, mixed dimensionality, or a name used both with and
without an index.

# Examples
```julia
p = rvars(chn)          # with the MCMCChains extension loaded
p.xyz                   # RVar{Float64, 2}, size (ntrials, npatients)
p.xyz[1, 3]             # RVar{Float64, 0} — every draw for trial 1, patient 3
(; xyz, sigma) = rvars(chn)
```
"""
function rvars(x::RVar{T, 1}; dims::NamedTuple=NamedTuple(),
               labels::NamedTuple=NamedTuple()) where {T}
    nms = variables(x)
    nms === nothing && error("rvars needs parameter names, and this random variable has " *
                             "none; build it with from_chains(array, param_names) or from a Chains object")
    store = x.draws
    order, cols, locs = _group_param_names(nms)

    # Reject unknown keys rather than silently ignoring them — a typo'd parameter or
    # dimension name would otherwise leave the axes quietly unnamed.
    for k in keys(dims)
        k in order || error("dims refers to :$k, which is not a parameter in this fit; " *
                            "available: $(join(string.(order), ", "))")
    end

    # Unlike `dims`, extra keys in `labels` are fine: labels are keyed by dimension and a
    # whole data frame's worth of levels (see `recover_types`) is a legitimate thing to
    # pass, most of whose columns are not dimensions of this fit.
    vals = Any[_attach_dims(_assemble_param(b, store, cols[b], locs[b], x.nchains),
                            b, dims, labels) for b in order]
    return NamedTuple{(order...,)}((vals...,))
end

function rvars(x::RVar{T, N}; kwargs...) where {T, N}
    error("rvars expects a vector random variable of per-element parameter draws (N == 1), got N == $N")
end

# Attach the dimension names declared for `base`, plus any labels registered for those
# dimension names. Labels are keyed by dimension rather than by parameter, so :arm means
# the same thing in every parameter that has an :arm axis and is declared once.
function _attach_dims(v::RVar{T, M}, base::Symbol, dims::NamedTuple,
                      labels::NamedTuple) where {T, M}
    haskey(dims, base) || return v
    spec = dims[base]
    dn = spec isa Symbol ? (spec,) : Tuple(Symbol.(spec))
    M == 0 && error("Parameter :$base is a scalar, so it has no dimensions to name")
    length(dn) == M || error("Got $(length(dn)) dimension names for parameter :$base, " *
                             "which has $M dimension(s): size $(size(v))")
    dl = ntuple(d -> get(labels, dn[d], nothing), M)
    for d in 1:M
        labs = dl[d]
        labs === nothing && continue
        length(labs) == size(v, d) || error(
            "Dimension :$(dn[d]) of parameter :$base has length $(size(v, d)), " *
            "but $(length(labs)) labels were supplied")
    end
    dl_final = any(l -> l !== nothing, dl) ? dl : nothing
    return RVar{T, M, typeof(v.draws)}(v.draws, v.nchains, nothing, dn, dl_final)
end

"""
    recover_types(data; sorted=true)

Derive dimension labels from the data the model was fitted to, in the spirit of
tidybayes' `recover_types`. `data` is anything `pairs` yields `name => column` from — a
`NamedTuple` of vectors, a `Dict`, or `eachcol(df)` for a `DataFrame`. Each column becomes
one entry mapping the column's name to its distinct values, ready to pass as `labels`:

```julia
labs = recover_types((arm = ["control", "drug", "control"], site = ["a", "b", "a"]))
# (arm = ["control", "drug"], site = ["a", "b"])

p = rvars(chn; dims = (a = (:trial, :arm),), labels = labs)
```

Columns that are not dimensions of the fit are simply ignored, so passing a whole table is
fine.

!!! warning
    The *order* of the recovered labels must match the integer coding the model used, and
    that coding is not recorded in the fit. `sorted=true` (the default) sorts the distinct
    values, matching how R factors and `CategoricalArray`s number their levels by default;
    `sorted=false` keeps first-appearance order. If your model indexed some other way,
    pass the labels explicitly rather than recovering them.
"""
function recover_types(data; sorted::Bool=true)
    ks = Symbol[]
    vs = Any[]
    for (k, col) in pairs(data)
        col isa AbstractVector || continue
        # A continuous column is a measurement, not a dimension; every row would otherwise
        # come back as its own "level".
        eltype(col) <: AbstractFloat && continue
        lv = unique(col)
        if sorted
            # Not everything is orderable; an unsortable column keeps appearance order.
            try
                lv = sort(lv)
            catch
            end
        end
        push!(ks, Symbol(k))
        push!(vs, lv)
    end
    return NamedTuple{(ks...,)}((vs...,))
end

function rvars(array::AbstractArray{<:Any, 3}, param_names::AbstractVector{Symbol}; kwargs...)
    return rvars(from_chains(array, param_names; flat=true); kwargs...)
end

function rvars(array::AbstractArray{<:Any, 3}, param_names::AbstractVector{String}; kwargs...)
    return rvars(array, Symbol.(param_names); kwargs...)
end
