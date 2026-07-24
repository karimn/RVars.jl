import LinearAlgebra: *, dot

function _prep_matmul(x::RVar, y::RVar)
    nx, ny = ndraws(x), ndraws(y)
    if nx != ny && nx != 1 && ny != 1
        error("Incompatible number of draws: $nx vs $ny")
    end
    nc = _combine_nchains(x, y)
    dx = nx == ny ? draws(x) : _broadcast_draws(x, max(nx, ny))
    dy = nx == ny ? draws(y) : _broadcast_draws(y, max(nx, ny))
    dx, dy, nc
end

function *(x::RVar{T, 2}, y::RVar{S, 2}) where {T, S}
    dx, dy, nc = _prep_matmul(x, y)
    n, m, k = size(dx, 1), size(dx, 2), size(dx, 3)
    p = size(dy, 3)
    result = similar(dx, promote_type(T, S), (n, m, p))
    for i in 1:n
        view(result, i, :, :) .= view(dx, i, :, :) * view(dy, i, :, :)
    end
    RVar{promote_type(T, S), 2, typeof(result)}(result, nc)
end

function *(x::RVar{T, 2}, y::AbstractMatrix{S}) where {T, S}
    dx = draws(x)
    n, m, k = size(dx, 1), size(dx, 2), size(dx, 3)
    p = size(y, 2)
    result = similar(dx, promote_type(T, S), (n, m, p))
    for i in 1:n
        view(result, i, :, :) .= view(dx, i, :, :) * y
    end
    RVar{promote_type(T, S), 2, typeof(result)}(result, nchains(x))
end

function *(x::AbstractMatrix{T}, y::RVar{S, 2}) where {T, S}
    dy = draws(y)
    n, k, p = size(dy, 1), size(dy, 2), size(dy, 3)
    m = size(x, 1)
    result = similar(dy, promote_type(T, S), (n, m, p))
    for i in 1:n
        view(result, i, :, :) .= x * view(dy, i, :, :)
    end
    RVar{promote_type(T, S), 2, typeof(result)}(result, nchains(y))
end

function *(x::RVar{T, 1}, y::RVar{S, 2}) where {T, S}
    dx, dy, nc = _prep_matmul(x, y)
    n = size(dx, 1)
    m = size(dx, 2)
    k = size(dy, 2)
    p = size(dy, 3)
    m == k || throw(DimensionMismatch("vector length $m does not match matrix rows $k"))
    result = similar(dx, promote_type(T, S), (n, p))
    for i in 1:n
        # Row-vector × matrix: (1×m)(m×p) -> length-p row, per draw.
        result[i, :] .= vec(view(dx, i, :)' * view(dy, i, :, :))
    end
    RVar{promote_type(T, S), 1, typeof(result)}(result, nc)
end

function *(x::RVar{T, 2}, y::RVar{S, 1}) where {T, S}
    dx, dy, nc = _prep_matmul(x, y)
    n = size(dx, 1)
    m = size(dx, 2)
    result = similar(dx, promote_type(T, S), (n, m))
    for i in 1:n
        result[i, :] .= view(dx, i, :, :) * view(dy, i, :)
    end
    RVar{promote_type(T, S), 1, typeof(result)}(result, nc)
end

function *(x::RVar{T, 1}, y::AbstractVector{S}) where {T, S}
    dx = draws(x)
    n = size(dx, 1)
    m = size(dx, 2)
    result = similar(dx, promote_type(T, S), (n,))
    for i in 1:n
        result[i] = dot(view(dx, i, :), y)
    end
    RVar{promote_type(T, S), 0, typeof(result)}(result, nchains(x))
end

function *(x::AbstractVector{T}, y::RVar{S, 1}) where {T, S}
    dy = draws(y)
    n = size(dy, 1)
    m = size(dy, 2)
    result = similar(dy, promote_type(T, S), (n,))
    for i in 1:n
        result[i] = dot(x, view(dy, i, :))
    end
    RVar{promote_type(T, S), 0, typeof(result)}(result, nchains(y))
end

# Mixed vector-RV / plain-matrix products. Without these the generic AbstractArray
# fallbacks either throw (a vector RV is seen as an m×1 column) or, for dot, recurse
# until the stack overflows.

function *(x::RVar{T, 1}, y::AbstractMatrix{S}) where {T, S}
    dx = draws(x)
    n, m = size(dx, 1), size(dx, 2)
    k, p = size(y, 1), size(y, 2)
    m == k || throw(DimensionMismatch("vector length $m does not match matrix rows $k"))
    result = similar(dx, promote_type(T, S), (n, p))
    for i in 1:n
        result[i, :] .= vec(view(dx, i, :)' * y)
    end
    RVar{promote_type(T, S), 1, typeof(result)}(result, nchains(x))
end

function *(x::AbstractMatrix{T}, y::RVar{S, 1}) where {T, S}
    dy = draws(y)
    n, k = size(dy, 1), size(dy, 2)
    m = size(x, 1)
    size(x, 2) == k || throw(DimensionMismatch("matrix columns $(size(x, 2)) do not match vector length $k"))
    result = similar(dy, promote_type(T, S), (n, m))
    for i in 1:n
        result[i, :] .= x * view(dy, i, :)
    end
    RVar{promote_type(T, S), 1, typeof(result)}(result, nchains(y))
end

function *(x::RVar{T, 2}, y::AbstractVector{S}) where {T, S}
    dx = draws(x)
    n, m, k = size(dx, 1), size(dx, 2), size(dx, 3)
    k == length(y) || throw(DimensionMismatch("matrix columns $k do not match vector length $(length(y))"))
    result = similar(dx, promote_type(T, S), (n, m))
    for i in 1:n
        result[i, :] .= view(dx, i, :, :) * y
    end
    RVar{promote_type(T, S), 1, typeof(result)}(result, nchains(x))
end

function *(x::AbstractVector{T}, y::RVar{S, 2}) where {T, S}
    dy = draws(y)
    n, k, p = size(dy, 1), size(dy, 2), size(dy, 3)
    length(x) == k || throw(DimensionMismatch("vector length $(length(x)) does not match matrix rows $k"))
    result = similar(dy, promote_type(T, S), (n, p))
    for i in 1:n
        result[i, :] .= vec(x' * view(dy, i, :, :))
    end
    RVar{promote_type(T, S), 1, typeof(result)}(result, nchains(y))
end

function dot(x::RVar{T, 1}, y::RVar{S, 1}) where {T, S}
    dx, dy, nc = _prep_matmul(x, y)
    n = size(dx, 1)
    m = size(dx, 2)
    result = similar(dx, promote_type(T, S), (n,))
    for i in 1:n
        result[i] = dot(view(dx, i, :), view(dy, i, :))
    end
    RVar{promote_type(T, S), 0, typeof(result)}(result, nc)
end

function dot(x::RVar{T, 1}, y::AbstractVector{S}) where {T, S}
    dx = draws(x)
    n = size(dx, 1)
    size(dx, 2) == length(y) || throw(DimensionMismatch("lengths $(size(dx, 2)) and $(length(y)) do not match"))
    result = similar(dx, promote_type(T, S), (n,))
    for i in 1:n
        result[i] = dot(view(dx, i, :), y)
    end
    RVar{promote_type(T, S), 0, typeof(result)}(result, nchains(x))
end

function dot(x::AbstractVector{T}, y::RVar{S, 1}) where {T, S}
    dy = draws(y)
    n = size(dy, 1)
    length(x) == size(dy, 2) || throw(DimensionMismatch("lengths $(length(x)) and $(size(dy, 2)) do not match"))
    result = similar(dy, promote_type(T, S), (n,))
    for i in 1:n
        result[i] = dot(x, view(dy, i, :))
    end
    RVar{promote_type(T, S), 0, typeof(result)}(result, nchains(y))
end
