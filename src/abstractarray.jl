Base.IndexStyle(::Type{<:RandomDraw}) = IndexLinear()

function Base.getindex(x::RandomDraw{T, N}, i::Int) where {T, N}
    d = x.draws
    n_draws = size(d, 1)
    total = length(x)
    if i < 1 || i > total
        throw(BoundsError(x, i))
    end
    data = d[:, i]
    RandomDraw{T, 0, typeof(data)}(data, x.nchains)
end

function Base.getindex(x::RandomDraw{T, N}, I::Vararg{Int, N}) where {T, N}
    d = x.draws
    n_draws = size(d, 1)
    sz = size(d)
    indices = map((i, dim_sz) -> min(i, dim_sz), I, sz[2:end])
    linear_idx = 1
    for (dim_idx, idx) in enumerate(indices)
        stride = prod(sz[(dim_idx+2):end])
        linear_idx += (idx - 1) * stride
    end
    if linear_idx < 1 || linear_idx > prod(sz[2:end])
        throw(BoundsError(x, I))
    end
    data = d[:, linear_idx]
    RandomDraw{T, 0, typeof(data)}(data, x.nchains)
end

function Base.getindex(x::RandomDraw{T, N}, idx::AbstractArray{Bool}) where {T, N}
    if length(idx) != length(x)
        throw(DimensionMismatch("logical index length $(length(idx)) != length $(length(x))"))
    end
    d = x.draws
    flat_idx = vec(idx)
    result_draws = d[:, flat_idx]
    sz = size(result_draws)
    RandomDraw{eltype(result_draws), length(sz) - 1, typeof(result_draws)}(result_draws, x.nchains)
end

function Base.setindex!(x::RandomDraw{T, N}, val::Number, i::Int) where {T, N}
    d = x.draws
    total = length(x)
    if i < 1 || i > total
        throw(BoundsError(x, i))
    end
    d[:, i] .= val
    return x
end

function Base.setindex!(x::RandomDraw, val::RandomDraw, i::Int)
    vd = draws(val)
    if size(vd, 1) == 1
        vd = repeat(vd, size(x.draws, 1))
    end
    total = length(x)
    if i < 1 || i > total
        throw(BoundsError(x, i))
    end
    x.draws[:, i] .= vec(vd)
    return x
end

function Base.reshape(x::RandomDraw{T, N}, dims::Dims) where {T, N}
    new_dims = (size(x.draws, 1), dims...)
    reshaped = reshape(x.draws, new_dims)
    RandomDraw{T, length(dims), typeof(reshaped)}(reshaped, x.nchains)
end

function Base.reshape(x::RandomDraw, dims::Int...)
    reshape(x, dims)
end

function Base.similar(x::RandomDraw{T, N}, ::Type{S}, dims::Dims) where {T, N, S}
    new_dims = (size(x.draws, 1), dims...)
    data = similar(x.draws, S, new_dims)
    RandomDraw{S, length(dims), typeof(data)}(data, x.nchains)
end

function Base.copy(x::RandomDraw{T, N}) where {T, N}
    RandomDraw{T, N, typeof(x.draws)}(copy(x.draws), x.nchains)
end

function Base.show(io::IO, x::RandomDraw{T, N}) where {T, N}
    nd = ndraws(x)
    nc = nchains(x)
    if N == 0
        print(io, "RandomDraw{$T}<$(nd),$(nc)>[1] mean ± sd:")
        if nd > 0
            m = Statistics.mean(x)
            s = Statistics.std(x)
            print(io, "\n[1] ", round(m; digits=2), " ± ", round(s; digits=2))
        end
    else
        sz_str = join(size(x), ",")
        print(io, "RandomDraw{$T}<$(nd),$(nc)>[$(sz_str)] mean ± sd:")
        n_show = min(length(x), 6)
        m_vals = Statistics.mean(x)
        s_vals = Statistics.std(x)
        if N == 1
            for i in 1:n_show
                print(io, "\n[$i] ", round(m_vals[i]; digits=2), " ± ", round(s_vals[i]; digits=2))
            end
        else
            for i in 1:min(size(x, 1), 4)
                print(io, "\n[$i,:] ")
                print(io, round.(m_vals[i, :]; digits=2), " ± ", round.(s_vals[i, :]; digits=2))
            end
            if size(x, 1) > 4
                print(io, "\n...")
            end
        end
        if length(x) > 6
            print(io, "\n...")
        end
    end
end
