# Define some new primitives: conv4 and pool

"""

`conv4(w,x;kwargs...)` executes convolutions or cross-correlations
using filters specified with `w` over tensor `x`.  Currently 4 or 5
dimensional KnetArrays with Float32 or Float64 entries are supported.

If `w` has dimensions (W1,W2,...,I,O) and `x` has dimensions
(X1,X2,...,I,N), the result y will have dimensions (Y1,Y2,...,O,N)
where

    Yi=1+floor((Xi+2*padding[i]-Wi)/stride[i])

Here I is the number of input channels, O is the number of output
channels, N is the number of instances, and Wi,Xi,Yi are spatial
dimensions.  Padding and stride are keyword arguments that can be
specified as a single number (in which case they apply to all
dimensions), or an array/tuple with entries for each spatial
dimension.

Here is a description of all available keyword arguments:

* padding: the number of extra zeros implicitly concatenated at the start and at the end of each dimension. Default=floor((filterSize-1)/2) which preserves the input size when filterSize is odd and stride=1.
* stride: the number of elements to slide to reach the next filtering window. Default=1.
* upscale: upscale factor for each dimension. Default=1.
* mode: 0 for convolution and 1 for cross-correlation.  Default=0.
* alpha,beta: blend or scale the result: dstValue = alpha*srcValue + beta*priorDstValue. Default=1,0.
* algo: specifies which convolution algorithm shoud be used to compute the results. Default=0. See the CUDNN User Guide for details.
* workSpace: data pointer to GPU memory to a workspace needed to able to execute the specified algorithm. Default=C_NULL.
* workSpaceSizeInBytes: the size in bytes of the provided workSpace. Default=0.
* handle: handle to a previously created cuDNN context. Default=Knet allocated context.

"""
function conv4{T}(w::KnetArray{T},x::KnetArray{T};
                  handle=cudnnhandle, alpha=one(T), beta=zero(T),
                  algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, o...)
    y = similar(x, cdims(w,x;o...))
    @cuda(cudnn, cudnnConvolutionForward,
          (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,   UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
          handle,Ref(T(alpha)),TD(x),x,FD(w),w,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(T(beta)),TD(y),y)
    return y
end

function conv4x{T}(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T};
                   handle=cudnnhandle, alpha=one(T), beta=zero(T),
                   algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, o...)
    dx = similar(x)
    if cudnnVersion >= 4000
        @cuda(cudnn,cudnnConvolutionBackwardData,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),FD(w),w,TD(dy),dy,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(T(beta)),TD(dx),dx)
    elseif cudnnVersion >= 3000
        @cuda(cudnn,cudnnConvolutionBackwardData_v3,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),FD(w),w,TD(dy),dy,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(T(beta)),TD(dx),dx)
    else
        @cuda(cudnn,cudnnConvolutionBackwardData,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,       Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),FD(w),w,TD(dy),dy,CD(w,x;o...),Ref(T(beta)),TD(dx),dx)
    end
    return dx
end

function conv4w{T}(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T};
                   handle=cudnnhandle, alpha=one(T), beta=zero(T),
                   algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, o...)
    dw = similar(w)
    if cudnnVersion >= 4000
        @cuda(cudnn,cudnnConvolutionBackwardFilter,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),TD(x),x,TD(dy),dy,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(T(beta)),FD(dw),dw)
    elseif cudnnVersion >= 3000
        @cuda(cudnn,cudnnConvolutionBackwardFilter_v3,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),TD(x),x,TD(dy),dy,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(T(beta)),FD(dw),dw)
    else
        @cuda(cudnn,cudnnConvolutionBackwardFilter,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,       Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),TD(x),x,TD(dy),dy,CD(w,x;o...),Ref(T(beta)),FD(dw),dw)
    end
    return dw
end


@primitive conv4(w,x; o...),dy  conv4w(w,x,dy;o...)  conv4x(w,x,dy;o...)
@zerograd  conv4x(w,x,dy;o...)
@zerograd  conv4w(w,x,dy;o...)


"""

`pool(x;kwargs...)` computes pooling of input values (i.e., the
maximum or average of several adjacent values) to produce an output
with smaller height and/or width.  Currently 4 or 5 dimensional
KnetArrays with Float32 or Float64 entries are supported.

If `x` has dimensions (X1,X2,...,I,N), the result y will have
dimensions (Y1,Y2,...,I,N) where

   Yi=1+floor((Xi+2*padding[i]-window[i])/stride[i])

Here I is the number of input channels, N is the number of instances,
and Xi,Yi are spatial dimensions.  Window, padding and stride are
keyword arguments that can be specified as a single number (in which
case they apply to all dimensions), or an array/tuple with entries for
each spatial dimension.

Here is a description of all available keyword arguments:

* window: the pooling window size for each dimension. Default=2.
* padding: the number of extra zeros implicitly concatenated at the start and at the end of each dimension. Default=0.
* stride: the number of elements to slide to reach the next pooling window. Default=same as window.
* mode: 0 for max, 1 for average including padded values, 2 for average excluding padded values.  Default=0.
* maxpoolingNanOpt: Nan numbers are not propagated if 0, they are propagated if 1. Default=0.
* alpha: can be used to scale the result. Default=1.
* handle: Handle to a previously created cuDNN context. Default=Knet allocated context.

"""
function pool{T}(x::KnetArray{T}; handle=cudnnhandle, alpha=one(T), beta=zero(T), o...)
    y = similar(x, pdims(x; o...))
    @cuda(cudnn, cudnnPoolingForward,
          (Cptr, Cptr,      Ptr{T},    Cptr,Ptr{T},Ptr{T},   Cptr,Ptr{T}),
          handle,PD(x;o...),Ref(T(alpha)),TD(x),x,    Ref(T(beta)),TD(y),y)
    return y
end

function poolx{T}(x::KnetArray{T},y::KnetArray{T},dy::KnetArray{T};
                  handle=cudnnhandle, alpha=one(T), beta=zero(T), o...)
    dx = similar(x)
    @cuda(cudnn,cudnnPoolingBackward,
          (Cptr,Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Ptr{T},Cptr,Ptr{T}),
          handle,PD(x;o...),Ref(T(alpha)),TD(y),y,TD(dy),dy,TD(x),x,Ref(T(beta)),TD(dx),dx)
    return dx
end

@primitive pool(x;o...),dy,y  poolx(x,y,dy;o...)
@zerograd  poolx(x,y,dy;o...)

# cudnn descriptors

type TD; ptr
    function TD(a::KnetArray)
        d = Cptr[0]
        @cuda(cudnn,cudnnCreateTensorDescriptor,(Ptr{Cptr},),d)
        n = ndims(a)
        sz = [Cint(size(a,n-i+1)) for i=1:n]
        st = [Cint(stride(a,n-i+1)) for i=1:n]
        @cuda(cudnn,cudnnSetTensorNdDescriptor,
              (Cptr,UInt32,Cint,Ptr{Cint},Ptr{Cint}),
              d[1], DT(a), n, sz, st)
        td = new(d[1])
        finalizer(td, x->@cuda(cudnn,cudnnDestroyTensorDescriptor,(Cptr,),x.ptr))
        return td
    end
end

type FD; ptr
    function FD(a::KnetArray)
        d = Cptr[0]
        @cuda(cudnn,cudnnCreateFilterDescriptor,(Ptr{Cptr},),d)
        n = ndims(a)
        sz = [Cint(size(a,n-i+1)) for i=1:n]
        if cudnnVersion >= 5000
            @cuda(cudnn,cudnnSetFilterNdDescriptor,
                  (Cptr,UInt32,UInt32,Cint,Ptr{Cint}),
                  d[1], DT(a), 0,     n,   sz)
        elseif cudnnVersion >= 4000
            @cuda(cudnn,cudnnSetFilterNdDescriptor_v4,
                  (Cptr,UInt32,UInt32,Cint,Ptr{Cint}),
                  d[1], DT(a), 0,     n,   sz)
        else
            @cuda(cudnn,cudnnSetFilterNdDescriptor,
                  (Cptr,UInt32,Cint,Ptr{Cint}),
                  d[1], DT(a),    n,   sz)
        end
        fd = new(d[1])
        finalizer(fd, x->@cuda(cudnn,cudnnDestroyFilterDescriptor,(Cptr,),x.ptr))
        return fd
    end
end

type CD; ptr
    function CD(w::KnetArray,x::KnetArray; padding=padsize(w), stride=1, upscale=1, mode=0)
        d = Cptr[0]
        @cuda(cudnn,cudnnCreateConvolutionDescriptor,(Ptr{Cptr},),d)
        nd = ndims(x)-2
        if cudnnVersion >= 4000
            @cuda(cudnn,cudnnSetConvolutionNdDescriptor,
                  (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32,UInt32),
                  d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),mode,DT(x))
        elseif cudnnVersion >= 3000
            @cuda(cudnn,cudnnSetConvolutionNdDescriptor_v3,
                  (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32,UInt32),
                  d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),mode,DT(x))
        else
            @cuda(cudnn,cudnnSetConvolutionNdDescriptor,
                  (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32),
                  d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),mode)
        end
        cd = new(d[1])
        finalizer(cd, x->@cuda(cudnn,cudnnDestroyConvolutionDescriptor,(Cptr,),x.ptr))
        return cd
    end
end

type PD; ptr
    function PD(x::KnetArray; window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0)
        d = Cptr[0]
        @cuda(cudnn,cudnnCreatePoolingDescriptor,(Ptr{Cptr},),d)
        nd = ndims(x)-2
        if cudnnVersion >= 5000
            @cuda(cudnn,cudnnSetPoolingNdDescriptor,
                  (Cptr,UInt32,UInt32,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),
                  d[1],mode,maxpoolingNanOpt,nd,cdsize(window,nd),cdsize(padding,nd),cdsize(stride,nd))
        elseif cudnnVersion >= 4000
            @cuda(cudnn,cudnnSetPoolingNdDescriptor_v4,
                  (Cptr,UInt32,UInt32,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),
                  d[1],mode,maxpoolingNanOpt,nd,cdsize(window,nd),cdsize(padding,nd),cdsize(stride,nd))
        else
            @cuda(cudnn,cudnnSetPoolingNdDescriptor,
                  (Cptr,UInt32,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),
                  d[1],mode,nd,cdsize(window,nd),cdsize(padding,nd),cdsize(stride,nd))
        end
        pd = new(d[1])
        finalizer(pd, x->@cuda(cudnn,cudnnDestroyPoolingDescriptor,(Cptr,),x.ptr))
        return pd
    end
end

import Base: unsafe_convert
unsafe_convert(::Type{Cptr}, td::TD)=td.ptr
unsafe_convert(::Type{Cptr}, fd::FD)=fd.ptr
unsafe_convert(::Type{Cptr}, cd::CD)=cd.ptr
unsafe_convert(::Type{Cptr}, pd::PD)=pd.ptr

function cdsize(w, nd)
    if isa(w,Integer)
        fill(Cint(w),nd)
    elseif length(w)==nd 
        [ Cint(w[nd-i+1]) for i=1:nd ]
    else
        throw(DimensionMismatch("$w $nd"))
    end
end

DT(::KnetArray{Float32})=UInt32(0)
DT(::KnetArray{Float64})=UInt32(1)
DT(::KnetArray{Float16})=UInt32(2)

function cdims(w,x; padding=padsize(w), stride=1, o...)
    N = ndims(w)
    ntuple(N) do i
        if i < N-1
            pi = (if isa(padding,Number); padding; else padding[i]; end)
            si = (if isa(stride,Number); stride; else stride[i]; end)
            1 + div(size(x,i) - size(w,i) + 2*pi, si)
        elseif i == N-1
            size(w,N)
        else # i == N
            size(x,N)
        end
    end
end

function pdims(x; window=2, padding=0, stride=window, o...)
    N = ndims(x)
    ntuple(N) do i
        if i < N-1
            wi = (if isa(window,Number); window; else window[i]; end)
            pi = (if isa(padding,Number); padding; else padding[i]; end)
            si = (if isa(stride,Number); stride; else stride[i]; end)
            1 + div(size(x,i) + 2*pi - wi, si)
        else
            size(x,i)
        end
    end
end

# convolution padding size that preserves the input size when filter size is odd and stride=1
padsize(w)=ntuple(i->div(size(w,i)-1,2), ndims(w)-2)

"""

mat(x) reshapes x into a two-dimensional matrix.  For 1-D inputs mat
returns `reshape(x, (length(x),1))`.  For inputs with more than two
dimensions of size (X1,X2,...,XD), mat returns

    reshape(x, (X1*X2*...*X[D-1],XD))

"""
function mat(x)
    if ndims(x) > 2
        xn = size(x,ndims(x))
        reshape(x, (div(length(x),xn),xn))
    elseif ndims(x)==2
        x
    elseif ndims(x)==1
        reshape(x, (length(x),1))
    else
        throw(MethodError(mat,x))
    end
end


### CPU implementations: originally by Onur Kuru, 2016; adapted to Knet8 by Deniz Yuret, 2017

function _conv2{T}(x::Array{T,2}, w::Array{T,2}; padding=(0,0), stride=(1,1), mode=0)
    pad = Int[0,0]
    for i=1:2
        pad[i] = size(w,i)-1-padding[i]
        if pad[i] < 0; error("cpu conv4 does not support padding >= w"); end
    end
    if mode==1; w=rot180(w); end
    y = conv2(x, w)
    return y[1+pad[1]:stride[1]:end-pad[1], 1+pad[2]:stride[2]:end-pad[2]]
end

function conv4{T}(w::Array{T,4},x::Array{T,4}; alpha=1, beta=0,
                  padding=padsize(w), stride=(1,1), upscale=1, mode=0, # 0=conv, 1=xcorr
                  o...) # handle=cudnnhandle, algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0
    if beta != 0; error("cpu conv4 only supports beta=0"); end
    if upscale != 1; error("cpu conv4 only supports upscale=1"); end
    if isa(padding,Number); padding=(padding,padding); end
    if isa(stride,Number); stride=(stride,stride); end
    # To support stride!=1 we need a weird operation in conv4w which scatters dy into a larger array
    if stride != (1,1); error("cpu conv4 only supports stride=1"); end
    y = fill!(similar(x, cdims(w,x; padding=padding, stride=stride)), 0)
    Wx,Hx,Cx,N = size(x)
    Ww,Hw,Cw,K = size(w)
    if !(Cx==Cw && Hx>=Hw && Wx>=Ww); throw(DimensionMismatch()); end
    @inbounds for n in 1:N, k in 1:K, c in 1:Cx
        y[:,:,k,n] += _conv2(x[:,:,c,n], w[:,:,c,k]; padding=padding, stride=stride, mode=mode)
    end
    if alpha!=1; y = alpha*y; end
    return y
end

# dw = rot180(xcorr(x,dy)) if y=conv(w,x)
# dw = xcorr(x,dy) if y=xcorr(w,x)
# TODO: stride != 1 not working
function conv4w{T}(w::Array{T,4}, x::Array{T,4}, dy::Array{T,4}; alpha=1,
                  padding=padsize(w), stride=(1,1), mode=0, # 0=conv, 1=xcorr
                  o...) # handle=cudnnhandle, algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, upscale=1, beta=0
    Wx,Hx,C,Nx = size(x)
    Wy,Hy,K,Ny = size(dy)
    dw = fill!(similar(w),0)
    @inbounds for c in 1:C, k in 1:K, n in 1:Ny
        dw1 = _conv2(x[:,:,c,n], dy[:,:,k,n]; padding=padding, stride=stride, mode=1)
        if mode==0; dw1=rot180(dw1); end
        dw[:,:,c,k] += dw1
    end
    if alpha!=1; dw = alpha*dw; end
    return dw
end

# dx = xcorr(dy, w, 'full')
function conv4x{T}(w::Array{T,4}, x::Array{T,4}, dy::Array{T,4}; alpha=1,
                   padding=padsize(w), stride=(1,1), mode=0, # 0=conv, 1=xcorr
                   o...) # handle=cudnnhandle, algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, upscale=1, beta=0
    Wy,Hy,Ky,N = size(dy)
    Ww,Hw,C,Kw = size(w)
    pad1 = Ww - 1 - padding[1]
    pad2 = Hw - 1 - padding[2]
    dx = fill!(similar(x),0)
    @inbounds for n in 1:N, c in 1:C, k in 1:Kw
        dx1 = _conv2(dy[:,:,k,n], w[:,:,c,k]; padding=(pad1,pad2), stride=stride, mode=1-mode)
        dx[:,:,c,n] += dx1
    end
    if alpha!=1; dx = alpha*dx; end
    return dx
end

function pool{T}(x::Array{T,4}; alpha=1, beta=0,
                 window=(2,2), stride=window, padding=0, mode=0, # 0:max, 1:avg+pad, 2:avg-pad
                 maxpoolingNanOpt=0, o...) # handle=cudnnhandle
    if beta != 0; error("cpu pool only supports beta=0"); end
    if mode != 0; error("cpu pool only supports mode=0"); end # TODO
    if padding != 0; error("cpu pool only supports padding=0"); end # TODO
    if stride != window; error("cpu pool only supports stride=window"); end # TODO
    if maxpoolingNanOpt != 0; error("cpu pool only supports maxpoolingNanOpt=0"); end # TODO: check what this means
    if isa(window,Number); window=(window,window); end
    y = fill!(similar(x,pdims(x;window=window,padding=padding,stride=stride)),0)
    Wx,Hx,C,Nx = size(x);
    Wy,Hy,K,Ny = size(y);
    @inbounds for n in 1:Nx, c in 1:C, i in 0:stride[1]:Wx-stride[1], j in 0:stride[2]:Hx-stride[2]
        iy, jy = div(i,stride[1])+1, div(j,stride[2])+1
        wx_end = min(i+window[1], Wx)
        hx_end = min(j+window[2], Hx)
        y[iy,jy,c,n] = maximum(x[i+1:wx_end,j+1:hx_end,c,n])
    end
    if alpha != 1; y = alpha * y; end
    return y
end

function poolx{T}(x::Array{T,4}, y::Array{T,4}, dy::Array{T,4}; alpha=1, beta=0,
                 window=(2,2), stride=window, padding=0, mode=0, # 0:max, 1:avg+pad, 2:avg-pad
                 maxpoolingNanOpt=0, o...) # handle=cudnnhandle
    dx = fill!(similar(x), 0)
    Wx,Hx,C,Nx = size(x);
    Wy,Hy,K,Ny = size(y);
    @inbounds for n in 1:Nx, c in 1:C, i in 0:stride[1]:Wx-stride[1], j in 0:stride[2]:Hx-stride[2]
        iy, jy = div(i,stride[1])+1, div(j,stride[2])+1
        wx_end = min(i+window[1], Wx)
        hx_end = min(j+window[2], Hx)
        a = x[i+1:wx_end,j+1:hx_end,c,n]
        di,dj = ind2sub(a,indmax(a))
        dx[i+di,j+dj,c,n] += dy[iy,jy,c,n]
    end
    if alpha!=1; dx = alpha * dx; end
    return dx
end

# function cudnnGetConvolutionNdForwardOutputDim{T}(x::Array{T,4}, w::Array{T,4}; padding=padding,stride=stride)
#     Wx,Hx,Cx,N = size(x)
#     Ww,Hw,Cw,K = size(w)
#     @assert Cx==Cw
#     Wy,Hy = floor(Int, 1 + (Int[Wx,Hx] + 2*padding - Int[Ww,Hw]) / stride)
#     return (Wy,Hy,K,N)
# end

# function cudnnGetPoolingNdForwardOutputDim{T}(x::Array{T,4}; window=2, padding=0, stride=1, mode=CUDNN_POOLING_MAX)
#     # @assert padding==0 && stride==1 && mode==CUDNN_POOLING_MAX
#     dims = [size(x)...]
#     # (mode, pdims, window, padding, stride) = cudnnGetPoolingNdDescriptor(pd)
#     for i=1:length(dims)-2
#         # dims[i] = 1 + floor((dims[i] + 2*padding - window) / stride)
#         dims[i] = 1 + ceil((dims[i] + 2*padding - window) / stride)
#     end
#     tuple(dims...)
# end

# if GPU
#     import CUDNN: cudnnConvolutionForward, cudnnConvolutionBackwardFilter, cudnnConvolutionBackwardData, cudnnPoolingForward, cudnnPoolingBackward
#     using CUDNN: CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, CUDNN_CONVOLUTION_FWD_ALGO_FFT, CUDNN_POOLING_MAX, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
# else
#     const CUDNN_CONVOLUTION = (UInt32)(0)
#     const CUDNN_CROSS_CORRELATION = (UInt32)(1)
#     const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = (UInt32)(0)
#     const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = (UInt32)(1)
#     const CUDNN_CONVOLUTION_FWD_ALGO_GEMM = (UInt32)(2)
#     const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = (UInt32)(3)
#     const CUDNN_CONVOLUTION_FWD_ALGO_FFT = (UInt32)(4)
#     const CUDNN_POOLING_MAX = (UInt32)(0)
#     const CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = (UInt32)(1)
#     const CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = (UInt32)(2)
# end

# function cudnnConvolutionForward{T}(x::Array{T,4}, w::Array{T,4}, y::Array{T,4}; padding=0, stride=1, 
#                                     upscale=1, mode=CUDNN_CONVOLUTION, cd=nothing,
#                                     algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
#                                     workSpace=0, workSpaceSizeInBytes=0, alpha=1, beta=1,im2col=1)
#     # x: (W,H,C,N)
#     # w: (W,H,C,K) 
#     # y: (W,H,K,N) 
#     fill!(y,0)
#     @assert (padding==0 && stride==1 && upscale==1 && mode==CUDNN_CONVOLUTION && algorithm == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM) "$((padding,stride,upscale,mode,algorithm))"
#     Wx,Hx,Cx,N = size(x)
#     Ww,Hw,Cw,K = size(w)
#     @assert (Cx==Cw && Hx>=Hw && Wx>=Ww) "$((Wx,Hw,Ww,Hw))"

#     @inbounds for n in 1:N, k in 1:K, c in 1:Cx
#         y[:,:,k,n] += _conv2_gemm(x[:,:,c,n], w[:,:,c,k]; pad=padding, stride=stride, xcorr=mode!=0)
#     end
#     return y
# end

# # Alternative to _conv2 but 5x slower.
# # Does not handle tuple padding
# # Does not handle strides != 1
# function _conv2_gemm{T}(x0::Array{T,2}, w0::Array{T,2}; padding=0, stride=1, mode=0)
#     if padding > 0
#         x=zeros(eltype(x0),map(m->2*padding+m,size(x0))) 
#         x[padding+1:end-padding,padding+1:end-padding] = x0
#     else
#         x=x0
#     end
#     if mode==1
#         w = vec(w0)
#     else
#         w = reverse(vec(w0))
#     end
#     rwindow, cwindow = size(w0)
#     row_extend = size(x,1)-rwindow+1
#     col_extend = size(x,2)-cwindow+1
#     widx = [(j-1)*size(x,1)+i for i in 1:row_extend, j in 1:col_extend]
#     oidx = [(j-1)*size(x,1)+i for i in 1:rwindow, j in 1:cwindow]
#     destidx = [i+(j-1) for i in vec(widx), j in vec(oidx)]
#     return reshape(x[destidx]*w,row_extend,col_extend)
# end

