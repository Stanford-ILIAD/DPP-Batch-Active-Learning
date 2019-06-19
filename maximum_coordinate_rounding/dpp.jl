import Base.length
using LinearAlgebra
using StatsBase
using Plots
using Random
using DelimitedFiles

abstract type KernelSpace end

struct RBFSpace <: KernelSpace
    x::Matrix{Float64}
    sigma::Float64
end

function length(K::RBFSpace)::Int64
    return size(K.x, 2)
end

function(K::RBFSpace)(i::Int64)::Vector{Float64}
    return K.x[:, i]
end

function (K::RBFSpace)(p::Vector{Float64}, q::Vector{Float64})::Float64
    l2 = sum((p-q).^2)
    return exp(-l2/(2*K.sigma^2))
end

function (K::RBFSpace)(i::Int64, j::Int64)::Float64
    return K(K(i), K(j))
end

function(K::RBFSpace)(i::Int64, p::Vector{Float64})::Float64
    return K(K(i), p)
end

function(K::RBFSpace)(p::Vector{Float64}, i::Int64)::Float64
    return K(p, K(i))
end

function GridSpace(xs, sigma::Float64)::RBFSpace
    return RBFSpace(hcat([[p...] for p in Iterators.product(xs...)]...), sigma)
end

struct Sample
    K::KernelSpace
    S::Vector{Int64}
    M::Matrix{Float64} # Inverse of K[S, S]
    U::Matrix{Float64} # K[S, :]
    L::Vector{Float64} # K[., .]-K[., :] M K[:, .]
end

function (s::Sample)(p::Vector{Float64})
    u = s.K.(s.S, [p])
    return s.K(p, p)-u'*s.M*u
end

function value(s::Sample)::Float64
    return det(s.K.(s.S, s.S'))
end

@userplot PointPlot
@recipe function f(inp::PointPlot)
    s = inp.args[1]
    grid := false
    space = hcat(s.K.(1:length(s.K))...)
    showaxis := false
    label := ""
    if size(space, 1)>=1
        xlims := [min(space[1, :]...), max(space[1, :]...)]
    end
    if size(space, 1)>=2
        ylims := [min(space[2, :]...), max(space[2, :]...)]
    end
    if size(space, 1)>=3
        zlims := [min(space[3, :]...), max(space[3, :]...)]
    end
    @series begin
        seriestype := :scatter
        data = hcat(s.K.(s.S)...)
        Tuple(data[i, :] for i=1:size(data, 1))
    end
end

@userplot HeatPlot
@recipe function f(inp::HeatPlot)
    s = inp.args[1]
    space = hcat(s.K.(1:length(s.K))...)
    xrange = range(min(space[1, :]...), max(space[1, :]...), length=100)
    yrange = range(min(space[1, :]...), max(space[1, :]...), length=100)
    @series begin
        seriestype := :heatmap
        xrange, yrange, (x, y)->s([x, y])
    end
end

function length(s::Sample)
    return length(s.S)
end

function initial(K::KernelSpace, S::Vector{Int64}=Int64[])::Sample
    U = K.(S, (1:length(K))')
    M = inv(U[:, S])
    L = K.(1:length(K), 1:length(K))-sum((M*U).*U, dims=1)[1, :]
    return Sample(K, S, M, U, L)
end

function sane(s::Sample)::Sample
    return initial(s.K, s.S)
end

function sanity(s::Sample)::Float64
    t = sane(s)
    eM = maximum(abs.(s.M-t.M))
    eU = maximum(abs.(s.U-t.U))
    eL = maximum(abs.(s.L-t.L))
    return max(eM, eU, eL)
end

function added(s::Sample, ind::Int64)::Sample
    k = length(s)
    S = [s.S; ind]
    U = [s.U; s.K.(ind, 1:length(s.K))']
    M1 = [s.M zeros(k, 1); zeros(1, k) 0]
    v = [-s.M*s.U[:, ind]; 1.]
    M = M1 + (v.*v')/s.L[ind]
    L = s.L-(((v'*U).^2)/s.L[ind])[1, :]
    return Sample(s.K, S, M, U, L)
end

function removed(s::Sample, i::Int64)::Sample
    S = [s.S[1:i-1]; s.S[i+1:end]]
    U = [s.U[1:i-1, :]; s.U[i+1:end, :]]
    v = s.M[:, i]
    M1 = s.M-(v.*v')./v[i]
    M = [M1[1:i-1, 1:i-1] M1[1:i-1, i+1:end]; M1[i+1:end, 1:i-1] M1[i+1:end, i+1:end]]
    L = s.L+(((v'*s.U).^2)./v[i])[1, :]
    return Sample(s.K, S, M, U, L)
end

function greedy_next(s::Sample)::Int64
    return argmax(s.L)
end

function random_next(s::Sample)::Int64
    return sample(1:length(s.K), Weights(max.(s.L, 1e-15)))
end

function output_points(filename::String, s::Sample)
    open(filename, "w") do io
        writedlm(io, ["x" "y"])
        writedlm(io, s.K.(s.S))
    end
end

function output_heatmap(filename::String, s::Sample, samples::Int64=21)
    space = hcat(s.K.(1:length(s.K))...)
    xrange = range(min(space[1, :]...), max(space[1, :]...), length=samples)
    yrange = range(min(space[1, :]...), max(space[1, :]...), length=samples)
    open(filename, "w") do io
        writedlm(io, ["x" "y" "z"])
        for x=xrange
            for y=yrange
                writedlm(io, [x y s([x, y])])
            end
            write(io, "\n")
        end
    end
end

function greedy(K::KernelSpace, k::Int64)::Sample
    s = initial(K)
    while length(s)<k
        s = added(s, greedy_next(s))
    end
    return s
end

function convex(K::KernelSpace, k::Int64)::Sample
    base = initial(K)
    while length(base)<k
        z = Float64[1. for i=1:length(K)]
        for i=base.S
            z[i] = 0.
        end
        s = initial(K, base.S)
        while length(s)<k
            s = added(s, argmax(s.L.*z))
        end
        for iter=1:(length(K)*log(k+1))
            z ./= sum(z)/length(z)
            s = removed(s, rand((length(base)+1):length(s)))
            probs = max.(s.L.*z, 1e-15)
            probs ./= sum(probs)
            ind = sample(1:length(K), Weights(probs))
            s = added(s, ind)
            if iter>2*k*log(k+1)
                z .+= probs
            end
        end
        ind = argmax(z)
        z[ind] = 0.
        base = added(base, ind)
    end
    return base
end

function run3()
  for iters=1:100
    N = 200
    D = 2
    k = 3
    sigma = 1.
    K = RBFSpace(rand(D, N), sigma)

    G = greedy(K, k)
    println(value(G))
    C = convex(K, k)
    println(value(C))
    println()
  end
end

function run20()
  for iters=1:100
    N = 200
    D = 2
    k = 20
    sigma = .2
    K = RBFSpace(rand(D, N), sigma)

    G = greedy(K, k)
    println(value(G))
    C = convex(K, k)
    println(value(C))
    println()
  end
end

run20()

#Random.seed!(0)


#pointplot(C)

#pointplot(convex(K, 10))
# println(greedy(K, 10))

# s = initial(K)
# for i=1:150
#     while true
#         j = rand(1:length(K))
#         if !(j in s.S)
#             global s=added(t, j)
#             break
#         end
#     end
# end
#
# for t=1:1000
#     global s = removed(s, rand(1:length(s)))
#     global s = added(s, random_next(s))
#     output_points("data/points_$(t).dat", s)
#     output_heatmap("data/heat_$(t).dat", s, 51)
# end
