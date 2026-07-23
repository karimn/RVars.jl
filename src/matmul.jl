import LinearAlgebra: *, dot

function _prep_matmul(x::RandomDraw, y::RandomDraw)
    nx, ny = ndraws(x), ndraws(y)
    if nx != ny && nx != 1 && ny != 1
        error("Incompatible number of draws: $nx vs $ny")
    end
    nc = _combine_nchains(x, y)
    dx = nx == ny ? draws(x) : _broadcast_draws(x, max(nx, ny))
    dy = nx == ny ? draws(y) : _broadcast_draws(y, max(nx, ny))
    dx, dy, nc
end

function *(x::RandomDraw{T, 2}, y::RandomDraw{S, 2}) where {T, S}
    dx, dy, nc = _prep_matmul(x, y)
    n, m, k = size(dx, 1), size(dx, 2), size(dx, 3)
    p = size(dy, 3)
    result = similar(dx, promote_type(T, S), (n, m, p))
    for i in 1:n
        view(result, i, :, :) .= view(dx, i, :, :) * view(dy, i, :, :)
    end
    RandomDraw{promote_type(T, S), 2, typeof(result)}(result, nc)
end

function *(x::RandomDraw{T, 2}, y::AbstractMatrix{S}) where {T, S}
    dx = draws(x)
    n, m, k = size(dx, 1), size(dx, 2), size(dx, 3)
    p = size(y, 2)
    result = similar(dx, promote_type(T, S), (n, m, p))
    for i in 1:n
        view(result, i, :, :) .= view(dx, i, :, :) * y
    end
    RandomDraw{promote_type(T, S), 2, typeof(result)}(result, nchains(x))
end

function *(x::AbstractMatrix{T}, y::RandomDraw{S, 2}) where {T, S}
    dy = draws(y)
    n, k, p = size(dy, 1), size(dy, 2), size(dy, 3)
    m = size(x, 1)
    result = similar(dy, promote_type(T, S), (n, m, p))
    for i in 1:n
        view(result, i, :, :) .= x * view(dy, i, :, :)
    end
    RandomDraw{promote_type(T, S), 2, typeof(result)}(result, nchains(y))
end

function *(x::RandomDraw{T, 1}, y::RandomDraw{S, 2}) where {T, S}
    dx, dy, nc = _prep_matmul(x, y)
    n = size(dx, 1)
    m = size(dx, 2)
    p = size(dy, 3)
    result = similar(dx, promote_type(T, S), (n, p))
    for i in 1:n
        result[i, :] .= view(dx, i, :) * view(dy, i, :, :)
    end
    RandomDraw{promote_type(T, S), 1, typeof(result)}(result, nc)
end

function *(x::RandomDraw{T, 2}, y::RandomDraw{S, 1}) where {T, S}
    dx, dy, nc = _prep_matmul(x, y)
    n = size(dx, 1)
    m = size(dx, 2)
    result = similar(dx, promote_type(T, S), (n, m))
    for i in 1:n
        result[i, :] .= view(dx, i, :, :) * view(dy, i, :)
    end
    RandomDraw{promote_type(T, S), 1, typeof(result)}(result, nc)
end

function *(x::RandomDraw{T, 1}, y::AbstractVector{S}) where {T, S}
    dx = draws(x)
    n = size(dx, 1)
    m = size(dx, 2)
    result = similar(dx, promote_type(T, S), (n,))
    for i in 1:n
        result[i] = dot(view(dx, i, :), y)
    end
    RandomDraw{promote_type(T, S), 0, typeof(result)}(result, nchains(x))
end

function *(x::AbstractVector{T}, y::RandomDraw{S, 1}) where {T, S}
    dy = draws(y)
    n = size(dy, 1)
    m = size(dy, 2)
    result = similar(dy, promote_type(T, S), (n,))
    for i in 1:n
        result[i] = dot(x, view(dy, i, :))
    end
    RandomDraw{promote_type(T, S), 0, typeof(result)}(result, nchains(y))
end

function dot(x::RandomDraw{T, 1}, y::RandomDraw{S, 1}) where {T, S}
    dx, dy, nc = _prep_matmul(x, y)
    n = size(dx, 1)
    m = size(dx, 2)
    result = similar(dx, promote_type(T, S), (n,))
    for i in 1:n
        result[i] = dot(view(dx, i, :), view(dy, i, :))
    end
    RandomDraw{promote_type(T, S), 0, typeof(result)}(result, nc)
end
