# tidybayes' gather_draws analogue: turn the hidden-draws-axis representation into a long
# table (a NamedTuple of equal-length vectors, already a valid Tables.jl column table)
# with one row per draw x element and one column per dimension.

const _GATHER_RESERVED = (:chain, :draw, :value)

function _check_reserved_collision(nm::Symbol, reserved::Tuple{Vararg{Symbol}})
    nm in reserved || return
    error("Dimension name :$nm collides with the reserved gather_draws column :$nm; " *
          "rename the dimension before calling gather_draws")
end

"""
    gather_draws(x::RVar)

Flatten `x` into a long table: one row per draw x element, as a `NamedTuple` of
equal-length vectors (a valid Tables.jl column table with no Tables.jl dependency).

Columns are one per dimension (named from [`dimnames`](@ref), falling back to `:dim1`,
`:dim2`, ... when `x` carries none), plus `:chain` (1-based chain index), `:draw` (the
1-based iteration within its chain), and `:value` (the scalar draw). A dimension's column
holds [`dimlabels`](@ref) at each position when the axis is labelled, otherwise the
integer position — label element types are preserved, never stringified. A rank-0 `x` has
no dimension columns, just `(chain, draw, value)`.

Row count is `ndraws(x) * length(x)`.

A flat vector `RVar` carrying element [`variables`](@ref) (e.g. `from_chains(...; flat=true)`)
does not use those element names as labels here — they describe elements, not the axis;
attach `dimlabels` (see [`rvars`](@ref)) if you want them in the table.

See also [`gather_draws(::NamedTuple)`](@ref) to gather several parameters at once.

# Examples
```julia
p = RVar(chn; dims = (a = (:trial, :arm),), labels = (arm = ["control", "drug"],))
gather_draws(p.a)
# (trial = [...], arm = [...], chain = [...], draw = [...], value = [...])
```
"""
function gather_draws(x::RVar{T, N}) where {T, N}
    nit = niterations(x)
    d = draws(x)
    nrows = length(d)

    dn = dimnames(x)
    dim_names = dn === nothing ? ntuple(i -> Symbol(:dim, i), N) : dn
    for nm in dim_names
        _check_reserved_collision(nm, _GATHER_RESERVED)
    end

    dl = dimlabels(x)
    labs = dl === nothing ? ntuple(i -> nothing, N) : dl

    dim_cols = ntuple(N) do k
        labs[k] === nothing ? Vector{Int}(undef, nrows) : Vector{eltype(labs[k])}(undef, nrows)
    end
    chain_col = Vector{Int}(undef, nrows)
    draw_col = Vector{Int}(undef, nrows)
    value_col = Vector{T}(undef, nrows)

    for (r, I) in enumerate(CartesianIndices(d))
        idx = Tuple(I)
        draw_lin = idx[1]
        draw_col[r] = (draw_lin - 1) % nit + 1
        chain_col[r] = (draw_lin - 1) ÷ nit + 1
        value_col[r] = d[I]
        for k in 1:N
            pos = idx[k + 1]
            dim_cols[k][r] = labs[k] === nothing ? pos : labs[k][pos]
        end
    end

    return merge(NamedTuple{dim_names}(dim_cols), (chain=chain_col, draw=draw_col, value=value_col))
end

"""
    gather_draws(nt::NamedTuple)

Gather several parameters (as returned by [`rvars`](@ref) or `RVar(chn)`) into a single
long table, with an extra `:variable` column holding each row's parameter name.

Parameters may differ in rank and in which dimensions they carry: the result takes the
union of all dimension columns across parameters, filling `missing` for rows whose
parameter lacks that dimension. Two parameters sharing a dimension name must carry
identical [`dimlabels`](@ref) for it (both unlabelled, or the same labels) — this is
already guaranteed if a shared `labels` argument was passed to [`rvars`](@ref); a genuine
disagreement is an error rather than a silent mismatch. Naming a dimension `:variable`
(colliding with the added column) is likewise an error, as are dimensions named `:chain`,
`:draw`, or `:value` (checked per-parameter, see [`gather_draws(::RVar)`](@ref)).

# Examples
```julia
p = RVar(chn; dims = (a = (:trial, :arm),), labels = (arm = ["control", "drug"],))
gather_draws(p)
# (variable = [...], trial = [...], arm = [...], chain = [...], draw = [...], value = [...])
```
"""
function gather_draws(nt::NamedTuple)
    isempty(nt) && error("gather_draws needs at least one parameter, got an empty NamedTuple")

    ks = keys(nt)
    tables = Vector{Any}(undef, length(ks))
    dim_names_per = Vector{Vector{Symbol}}(undef, length(ks))
    label_by_dim = Dict{Symbol, Any}()

    for (i, k) in enumerate(ks)
        v = nt[k]
        v isa RVar || error("gather_draws expects a NamedTuple of RVars; :$k is a $(typeof(v))")
        tables[i] = gather_draws(v)

        N = ndims(v)
        dn = dimnames(v)
        names_ = collect(dn === nothing ? ntuple(j -> Symbol(:dim, j), N) : dn)
        dim_names_per[i] = names_

        dl = dimlabels(v)
        labs = dl === nothing ? ntuple(j -> nothing, N) : dl
        for (j, nm) in enumerate(names_)
            lab = labs[j]
            if haskey(label_by_dim, nm)
                isequal(label_by_dim[nm], lab) ||
                    error("Parameters disagree on labels for dimension :$nm; " *
                          "gather_draws requires every parameter sharing a dimension " *
                          "name to carry identical labels for it")
            else
                label_by_dim[nm] = lab
            end
        end
    end

    all_dim_names = Symbol[]
    for names_ in dim_names_per, nm in names_
        nm in all_dim_names || push!(all_dim_names, nm)
    end
    _check_reserved_collision.(all_dim_names, Ref((:variable,)))

    nrows_each = [length(t.value) for t in tables]
    total = sum(nrows_each)

    variable_col = Vector{Symbol}(undef, total)
    chain_col = Vector{Int}(undef, total)
    draw_col = Vector{Int}(undef, total)
    value_type = mapreduce(t -> eltype(t.value), promote_type, tables)
    value_col = Vector{value_type}(undef, total)

    dim_cols = Dict{Symbol, Vector}()
    for nm in all_dim_names
        present = [i for i in eachindex(ks) if nm in dim_names_per[i]]
        et = eltype(getproperty(tables[first(present)], nm))
        dim_cols[nm] = length(present) == length(ks) ?
            Vector{et}(undef, total) : Vector{Union{Missing, et}}(undef, total)
    end

    pos = 1
    for (i, k) in enumerate(ks)
        tbl = tables[i]
        n = nrows_each[i]
        rng = pos:(pos + n - 1)
        variable_col[rng] .= k
        chain_col[rng] = tbl.chain
        draw_col[rng] = tbl.draw
        value_col[rng] = tbl.value
        for nm in all_dim_names
            if nm in dim_names_per[i]
                dim_cols[nm][rng] = getproperty(tbl, nm)
            else
                dim_cols[nm][rng] .= missing
            end
        end
        pos += n
    end

    dim_named_tuple = NamedTuple{(all_dim_names...,)}(Tuple(dim_cols[nm] for nm in all_dim_names))
    return merge((variable=variable_col,), dim_named_tuple,
                 (chain=chain_col, draw=draw_col, value=value_col))
end
