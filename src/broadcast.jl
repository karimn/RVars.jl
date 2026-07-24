Base.broadcastable(x::RandomDraw) = x

# Custom broadcast style to ensure output is RandomDraw
struct RandomDrawStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end

function Base.BroadcastStyle(::Type{<:RandomDraw{T, N}}) where {T, N}
    RandomDrawStyle{N}()
end

function Base.BroadcastStyle(s1::RandomDrawStyle{M}, s2::RandomDrawStyle{N}) where {M, N}
    RandomDrawStyle{max(M, N)}()
end

function Base.BroadcastStyle(s::RandomDrawStyle{M}, ::Base.Broadcast.DefaultArrayStyle{N}) where {M, N}
    RandomDrawStyle{max(M, N)}()
end

function Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{M}, s::RandomDrawStyle{N}) where {M, N}
    RandomDrawStyle{max(M, N)}()
end

function Base.BroadcastStyle(s::RandomDrawStyle{M}, ::Base.Broadcast.Style{Tuple}) where {M}
    RandomDrawStyle{M}()
end

# Recycle a 1-draw constant up to `target_nd` draws so it can be broadcast against
# operands that carry a full sample.
function _pad_draws(d, target_nd)
    nd = size(d, 1)
    nd == target_nd && return d
    ndims_d = ndims(d)
    factor = cld(target_nd, nd)
    inner = (factor, ntuple(i -> 1, max(0, ndims_d - 1))...)
    result = repeat(d; inner=inner)
    if ndims_d == 1
        return result[1:target_nd]
    end
    result[1:target_nd, ntuple(i -> Colon(), ndims_d - 1)...]
end

function Base.similar(bc::Base.Broadcast.Broadcasted{RandomDrawStyle{N}}, ::Type{T}) where {N, T}
    # A fused expression (x .* 2 .+ 1) nests Broadcasted objects inside bc.args, hiding the
    # RandomDraw operands. Flatten first so the args are leaves and the scan below sees them.
    bcf = Base.Broadcast.flatten(bc)
    src_draws = nothing
    n_draws = 1
    for a in bcf.args
        a isa RandomDraw || continue
        nd = size(a.draws, 1)
        # The draws axis must hold the full broadcast result; a 1-draw constant recycles.
        # Track the widest operand so `similar` inherits its array type, not a constant's.
        if src_draws === nothing || nd > n_draws
            src_draws = a.draws
        end
        n_draws = max(n_draws, nd)
    end
    axs = axes(bc)
    if src_draws === nothing
        return Array{T}(undef, axs...)
    end
    nc = _combine_nchains(bcf.args...)
    sz = (n_draws, length.(axs)...)
    data = similar(src_draws, T, sz)
    return _maybe_names(RandomDraw{T, N, typeof(data)}(data, nc), _combine_names(bcf.args...))
end

function Base.copy(bc::Base.Broadcast.Broadcasted{RandomDrawStyle{N}}) where {N}
    # Flatten so bcf.args holds only leaf operands and bcf.f is the composed function;
    # otherwise a nested Broadcasted would be passed straight through to the function
    # below, and the RandomDraw operands it hides would never be recycled or aligned.
    bcf = Base.Broadcast.flatten(bc)
    sample_args = map(a -> a isa AbstractArray ? a[1] : a, bcf.args)
    sample_result = bcf.f(sample_args...)
    T_inner = eltype(typeof(sample_result))
    dest = similar(bcf, T_inner)
    max_nd = size(dest.draws, 1)
    # Operate on the backing arrays in one fused broadcast rather than materialising a
    # scalar RandomDraw per logical element. Each operand is reshaped so axis 1 is the
    # draws axis and axes 2.. are the logical shape — the layout ordinary broadcast
    # expansion already knows how to align:
    #   - a RandomDraw contributes its store, recycled up to `max_nd` draws;
    #   - a plain array is draw-invariant, so it enters as (1, shape...) and repeats
    #     along the draws axis for free;
    #   - a scalar passes through untouched.
    prepared = map(bcf.args) do a
        if a isa RandomDraw
            d = _pad_draws(draws(a), max_nd)
            ndims(d) == 1 ? reshape(d, size(d, 1), 1) : d
        elseif a isa AbstractArray
            reshape(a, 1, size(a)...)
        else
            a
        end
    end
    dest.draws .= bcf.f.(prepared...)
    return dest
end
