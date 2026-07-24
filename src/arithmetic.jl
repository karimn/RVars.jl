function _binop_scalar(f::Function, x::RVar, y::Number)
    d = draws(x)
    result = f.(d, y)
    _maybe_names(RVar(result, nchains=nchains(x)), x.names)
end

function _binop_scalar(f::Function, x::Number, y::RVar)
    d = draws(y)
    result = f.(x, d)
    _maybe_names(RVar(result, nchains=nchains(y)), y.names)
end

function _binop_rv(f::Function, x::RVar, y::RVar)
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
    _maybe_names(RVar(result, nchains=_combine_nchains(x, y)), _combine_names(x, y))
end

Base.:+(x::RVar, y::RVar) = _binop_rv(+, x, y)
Base.:+(x::RVar, y::Number) = _binop_scalar(+, x, y)
Base.:+(x::Number, y::RVar) = _binop_scalar(+, x, y)

Base.:-(x::RVar, y::RVar) = _binop_rv(-, x, y)
Base.:-(x::RVar, y::Number) = _binop_scalar(-, x, y)
Base.:-(x::Number, y::RVar) = _binop_scalar(-, x, y)

Base.:*(x::RVar, y::RVar) = _binop_rv(*, x, y)
Base.:*(x::RVar, y::Number) = _binop_scalar(*, x, y)
Base.:*(x::Number, y::RVar) = _binop_scalar(*, x, y)

Base.:/(x::RVar, y::RVar) = _binop_rv(/, x, y)
Base.:/(x::RVar, y::Number) = _binop_scalar(/, x, y)
Base.:/(x::Number, y::RVar) = _binop_scalar(/, x, y)

Base.:\(x::RVar, y::RVar) = _binop_rv(\, x, y)
Base.:\(x::RVar, y::Number) = _binop_scalar(\, x, y)
Base.:\(x::Number, y::RVar) = _binop_scalar(\, x, y)

Base.:^(x::RVar, y::RVar) = _binop_rv(^, x, y)
Base.:^(x::RVar, y::Number) = _binop_scalar(^, x, y)
Base.:^(x::Number, y::RVar) = _binop_scalar(^, x, y)

Base.:%(x::RVar, y::RVar) = _binop_rv(%, x, y)
Base.:%(x::RVar, y::Number) = _binop_scalar(%, x, y)
Base.:%(x::Number, y::RVar) = _binop_scalar(%, x, y)

Base.:&(x::RVar, y::RVar) = _binop_rv(&, x, y)
Base.:&(x::RVar, y::Number) = _binop_scalar(&, x, y)
Base.:&(x::Number, y::RVar) = _binop_scalar(&, x, y)

Base.:|(x::RVar, y::RVar) = _binop_rv(|, x, y)
Base.:|(x::RVar, y::Number) = _binop_scalar(|, x, y)
Base.:|(x::Number, y::RVar) = _binop_scalar(|, x, y)

Base.:(==)(x::RVar, y::RVar) = _binop_rv(==, x, y)
Base.:(==)(x::RVar, y::Number) = _binop_scalar(==, x, y)
Base.:(==)(x::Number, y::RVar) = _binop_scalar(==, x, y)

Base.:<(x::RVar, y::RVar) = _binop_rv(<, x, y)
Base.:<(x::RVar, y::Number) = _binop_scalar(<, x, y)
Base.:<(x::Number, y::RVar) = _binop_scalar(<, x, y)

Base.:<=(x::RVar, y::RVar) = _binop_rv(<=, x, y)
Base.:<=(x::RVar, y::Number) = _binop_scalar(<=, x, y)
Base.:<=(x::Number, y::RVar) = _binop_scalar(<=, x, y)

Base.:>(x::RVar, y::RVar) = _binop_rv(>, x, y)
Base.:>(x::RVar, y::Number) = _binop_scalar(>, x, y)
Base.:>(x::Number, y::RVar) = _binop_scalar(>, x, y)

Base.:>=(x::RVar, y::RVar) = _binop_rv(>=, x, y)
Base.:>=(x::RVar, y::Number) = _binop_scalar(>=, x, y)
Base.:>=(x::Number, y::RVar) = _binop_scalar(>=, x, y)

Base.:!(x::RVar) = _maybe_names(RVar(.!(draws(x)), nchains=nchains(x)), x.names)
Base.:-(x::RVar) = _maybe_names(RVar(-(draws(x)), nchains=nchains(x)), x.names)

for f in [:sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh,
          :asinh, :acosh, :atanh,
          :exp, :log, :log2, :log10, :log1p, :sqrt,
          :abs, :sign, :floor, :ceil, :trunc, :round]
    @eval begin
        Base.$f(x::RVar) =
            _maybe_names(RVar($f.(draws(x)), nchains=nchains(x)), x.names)
    end
end

function Base.isapprox(x::RVar, y::RVar; kwargs...)
    nx, ny = ndraws(x), ndraws(y)
    dx = nx == ny ? draws(x) : _broadcast_draws(x, max(nx, ny))
    dy = nx == ny ? draws(y) : _broadcast_draws(y, max(nx, ny))
    return isapprox(dx, dy; kwargs...)
end
