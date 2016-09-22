using Knet,BenchmarkTools
using Knet: libknet8, @cuda

function sqrttest{T}(BLK::Int,THR::Int,CNT::Int,x::KnetArray{T},y::KnetArray{T})
    if     T<:Float32; ccall(("sqrt_32",libknet8),Void,(Cint,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat}),BLK,THR,CNT,x,y)
    elseif T<:Float64; ccall(("sqrt_64",libknet8),Void,(Cint,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble}),BLK,THR,CNT,x,y)
    else   error("$T not supported"); end
    @cuda(cudart,cudaDeviceSynchronize,())
    @cuda(cudart,cudaGetLastError,())  # @cuda(cudart,cudaPeekAtLastError,())
end

function runtests()
    data = Float32[]

    for T in [ Float32, Float64 ],
        CNT in vcat([ 10^i for i=1:6 ], [ 2^4i for i=1:5 ])
        global x = KnetArray(rand(T,CNT))
        global y = similar(x)
        for BLK in [ 2^i for i=5:10 ],
            THR in [ 2^i for i=5:10 ]
            bm = @benchmarkable sqrttest($BLK,$THR,$CNT,$x,$y) samples=1000
            bt = run(bm).times
            println((sizeof(T), CNT, BLK, THR, minimum(bt),median(bt),mean(bt),maximum(bt)))
            push!(data, sizeof(T), CNT, BLK, THR, median(bt))
        end
        println((:type,T,:size,CNT))
    end

    data = reshape(data, (5, div(length(data),5)))
end

function table(data, elsize, arsize)
    r = Array(Float32,6,6)
    for i=1:size(data,2)
        data[1,i] == elsize || continue
        data[2,i] == arsize || continue
        blk = round(Int,log(data[3,i]/16)/log(2))
        thr = round(Int,log(data[4,i]/16)/log(2))
        r[blk,thr] = data[5,i]
    end
    r ./= minimum(r)
    cols = [ 2^i for i=5:10 ]
    r = vcat(cols', r)
    rows = vcat(0, [ 2^i for i=5:10 ])
    r = hcat(rows, r)
    display(r)
end
