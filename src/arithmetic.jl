function _binop_scalar(f::Function, x::RandomDraw, y::Number)
    d = draws(x)
    result = f.(d, y)
    RandomDraw(result, nchains=nchains(x))
end

function _binop_scalar(f::Function, x::Number, y::RandomDraw)
    d = draws(y)
    result = f.(x, d)
    RandomDraw(result, nchains=nchains(y))
end

function _binop_rv(f::Function, x::RandomDraw, y::RandomDraw)
    nx, ny = ndraws(x), ndraws(y)
    if nx != ny && nx != 1 && ny != 1
        error("Incompatible number of draws: $nx vs $ny")
    end
    dx = nx == ny ? draws(x) : _broadcast_draws(x, max(nx, ny))
    dy = nx == ny ? draws(y) : _broadcast_draws(y, max(nx, ny))
    sx, sy = size(dx), size(dy)
    nd = max(length(sx), length(sy))
    pad_sx = ntuple(i -> i <= length(sx) ? sx[i] : 1, nd)
    pad_sy = ntuple(i -> i <= length(sy) ? sy[i] : 1, nd)
    for i in 1:nd
        if pad_sx[i] != pad_sy[i] && pad_sx[i] != 1 && pad_sy[i] != 1
            error("Dimension mismatch: $(sx[2:end]) vs $(sy[2:end])")
        end
    end
    result = f.(dx, dy)
    RandomDraw(result, nchains=_combine_nchains(x, y))
end

Base.:+(x::RandomDraw, y::RandomDraw) = _binop_rv(+, x, y)
Base.:+(x::RandomDraw, y::Number) = _binop_scalar(+, x, y)
Base.:+(x::Number, y::RandomDraw) = _binop_scalar(+, x, y)

Base.:-(x::RandomDraw, y::RandomDraw) = _binop_rv(-, x, y)
Base.:-(x::RandomDraw, y::Number) = _binop_scalar(-, x, y)
Base.:-(x::Number, y::RandomDraw) = _binop_scalar(-, x, y)

Base.:*(x::RandomDraw, y::RandomDraw) = _binop_rv(*, x, y)
Base.:*(x::RandomDraw, y::Number) = _binop_scalar(*, x, y)
Base.:*(x::Number, y::RandomDraw) = _binop_scalar(*, x, y)

Base.:/(x::RandomDraw, y::RandomDraw) = _binop_rv(/, x, y)
Base.:/(x::RandomDraw, y::Number) = _binop_scalar(/, x, y)
Base.:/(x::Number, y::RandomDraw) = _binop_scalar(/, x, y)

Base.:\(x::RandomDraw, y::RandomDraw) = _binop_rv(\, x, y)
Base.:\(x::RandomDraw, y::Number) = _binop_scalar(\, x, y)
Base.:\(x::Number, y::RandomDraw) = _binop_scalar(\, x, y)

Base.:^(x::RandomDraw, y::RandomDraw) = _binop_rv(^, x, y)
Base.:^(x::RandomDraw, y::Number) = _binop_scalar(^, x, y)
Base.:^(x::Number, y::RandomDraw) = _binop_scalar(^, x, y)

Base.:%(x::RandomDraw, y::RandomDraw) = _binop_rv(%, x, y)
Base.:%(x::RandomDraw, y::Number) = _binop_scalar(%, x, y)
Base.:%(x::Number, y::RandomDraw) = _binop_scalar(%, x, y)

Base.:&(x::RandomDraw, y::RandomDraw) = _binop_rv(&, x, y)
Base.:&(x::RandomDraw, y::Number) = _binop_scalar(&, x, y)
Base.:&(x::Number, y::RandomDraw) = _binop_scalar(&, x, y)

Base.:|(x::RandomDraw, y::RandomDraw) = _binop_rv(|, x, y)
Base.:|(x::RandomDraw, y::Number) = _binop_scalar(|, x, y)
Base.:|(x::Number, y::RandomDraw) = _binop_scalar(|, x, y)

Base.:(==)(x::RandomDraw, y::RandomDraw) = _binop_rv(==, x, y)
Base.:(==)(x::RandomDraw, y::Number) = _binop_scalar(==, x, y)
Base.:(==)(x::Number, y::RandomDraw) = _binop_scalar(==, x, y)

Base.:<(x::RandomDraw, y::RandomDraw) = _binop_rv(<, x, y)
Base.:<(x::RandomDraw, y::Number) = _binop_scalar(<, x, y)
Base.:<(x::Number, y::RandomDraw) = _binop_scalar(<, x, y)

Base.:<=(x::RandomDraw, y::RandomDraw) = _binop_rv(<=, x, y)
Base.:<=(x::RandomDraw, y::Number) = _binop_scalar(<=, x, y)
Base.:<=(x::Number, y::RandomDraw) = _binop_scalar(<=, x, y)

Base.:>(x::RandomDraw, y::RandomDraw) = _binop_rv(>, x, y)
Base.:>(x::RandomDraw, y::Number) = _binop_scalar(>, x, y)
Base.:>(x::Number, y::RandomDraw) = _binop_scalar(>, x, y)

Base.:>=(x::RandomDraw, y::RandomDraw) = _binop_rv(>=, x, y)
Base.:>=(x::RandomDraw, y::Number) = _binop_scalar(>=, x, y)
Base.:>=(x::Number, y::RandomDraw) = _binop_scalar(>=, x, y)

Base.:!(x::RandomDraw) = RandomDraw(.!(draws(x)), nchains=nchains(x))
Base.:-(x::RandomDraw) = RandomDraw(-(draws(x)), nchains=nchains(x))

for f in [:sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh,
          :asinh, :acosh, :atanh,
          :exp, :log, :log2, :log10, :log1p, :sqrt,
          :abs, :sign, :floor, :ceil, :trunc, :round]
    @eval begin
        Base.$f(x::RandomDraw) = RandomDraw($f.(draws(x)), nchains=nchains(x))
    end
end

function Base.isapprox(x::RandomDraw, y::RandomDraw; kwargs...)
    nx, ny = ndraws(x), ndraws(y)
    dx = nx == ny ? draws(x) : _broadcast_draws(x, max(nx, ny))
    dy = nx == ny ? draws(y) : _broadcast_draws(y, max(nx, ny))
    return isapprox(dx, dy; kwargs...)
end
