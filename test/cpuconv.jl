using Knet, BenchmarkTools
using Knet: conv4w,_conv2,padsize,conv4x,poolx

function testconv4()
    for t in (Float32, Float64)
        cx = rand(t, (9,8,7,6))
        cw = rand(t, (3,4,7,5))
        gx = KnetArray(cx)
        gw = KnetArray(cw)
        for p in ((0,0), (1,1), (2,2))
            for s in ((1,1),) #  2) TODO stride!=1
                for u in (1,)       # u!=1 does not work on gpu or cpu
                    for m in (0, 1)
                        for a in (1, 2)
                            for b in (0,) # b!=0 does not make sense when allocating y
                                println((t,:p,p,:s,s,:u,u,:m,m,:a,a,:b,b))
                                cy = conv4(cw,cx; padding=p, stride=s, upscale=u, mode=m, alpha=a, beta=b)
                                gy = conv4(gw,gx; padding=p, stride=s, upscale=u, mode=m, alpha=a, beta=b)
                                tst1 = isapprox(cy, Array(gy))
                                cdw = conv4w(cw,cx,cy; padding=p, stride=s, upscale=u, mode=m, alpha=a, beta=b)
                                gdw = conv4w(gw,gx,gy; padding=p, stride=s, upscale=u, mode=m, alpha=a, beta=b)
                                tst2 = isapprox(cdw, Array(gdw))
                                cdx = conv4x(cw,cx,cy; padding=p, stride=s, upscale=u, mode=m, alpha=a, beta=b)
                                gdx = conv4x(gw,gx,gy; padding=p, stride=s, upscale=u, mode=m, alpha=a, beta=b)
                                tst3 = isapprox(cdx, Array(gdx))
                                println((tst1,tst2,tst3))
                            end
                        end
                    end
                end
            end
        end
    end
end

function timeconv4()
    for t in (Float32, Float64)
        println(t)
        cx = rand(t, (28,28,1,100)) # (256,256,3,10))
        cw = rand(t, (5,5,1,20))    # (5, 5, 3, 20))
        @time cy = conv4(cw,cx;padding=(0,0))
        @time cdw = conv4w(cw,cx,cy;padding=(0,0))
        @time cdx = conv4x(cw,cx,cy;padding=(0,0))
        @time cy = conv4(cw,cx;padding=(0,0))
        @time cdw = conv4w(cw,cx,cy;padding=(0,0))
        @time cdx = conv4x(cw,cx,cy;padding=(0,0))
    end
end

function testpool4()
    for t in (Float32, Float64)
        cx = rand(t, (9,8,7,6))
        gx = KnetArray(cx)
        for w in ((2,2), (3,3), (2,3))
            # for p in ((0,0), (1,1), (2,2))
            p = 0 # TODO
            # for s in (w, (1,1))
            s = w # TODO
            # for m in (0, 1, 2)
            m = 0 # TODO
            n = 0 # TODO
            # for b in (0,) # b!=0 does not make sense when allocating y
            b = 0
            # for a in (1, 2) # a!=1 is broken in CUDNN back
            a = 1
            println((t,:w,w,:p,p,:s,s,:m,m,:n,n,:a,a,:b,b))
            cy = pool(cx; window=w, padding=p, stride=s, mode=m, maxpoolingNanOpt=n, alpha=a, beta=b)
            gy = pool(gx; window=w, padding=p, stride=s, mode=m, maxpoolingNanOpt=n, alpha=a, beta=b)
            tst1 = isapprox(cy, Array(gy))
            cdx = poolx(cx,cy,cy; window=w, padding=p, stride=s, mode=m, maxpoolingNanOpt=n, alpha=a, beta=b)
            gdx = poolx(gx,gy,gy; window=w, padding=p, stride=s, mode=m, maxpoolingNanOpt=n, alpha=a, beta=b)
            tst2 = isapprox(cdx, Array(gdx))
            println((tst1,tst2))
        end
    end
end


function _conv3{T}(x::Array{T,2}, w::Array{T,2}; padding=(0,0), stride=(1,1), mode=0)
    pad = Int[0,0]
    for i=1:2
        pad[i] = size(w,i)-1-padding[i]
        if pad[i] < 0; error("cpu conv4 does not support padding >= w"); end
    end
    if mode==1; w=rot180(w); end
    @show y = conv2(x, w)
    return y[1+pad[1]:stride[1]:end-pad[1], 1+pad[2]:stride[2]:end-pad[2]]
end

# dx = xcorr(dy, w, 'full')
function _conv4x{T}(w::Array{T,4}, x::Array{T,4}, dy::Array{T,4}; alpha=1,
                   padding=padsize(w), stride=(1,1), mode=0, # 0=conv, 1=xcorr
                   o...) # handle=cudnnhandle, algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, upscale=1, beta=0
    Wy,Hy,Ky,N = size(dy)
    Ww,Hw,C,Kw = size(w)
    pad1 = Ww - 1 - padding[1]
    pad2 = Hw - 1 - padding[2]
    dx = fill!(similar(x),0)
    @inbounds for n in 1:N, c in 1:C, k in 1:Kw
        w1 = w[:,:,c,k]
        # if mode==1; w1=rot180(w1); end
        dx1 = _conv2(dy[:,:,k,n], w1; padding=(pad1,pad2), stride=stride, mode=1-mode)
        dx[:,:,c,n] += dx1
    end
    if alpha!=1; dx = alpha*dx; end
    return dx
end

# o = Dict(:mode=>1,:stride=>(1,1),:padding=>(1,0))
# o = Dict(:mode=>0,:stride=>(1,1),:padding=>(1,0))
# cw = reshape([1.0:3.0...], (3,1,1,1))
# cx = reshape([10.0:10.0:50.0...], (5,1,1,1))
# cy = conv4(cw,cx;o...)
# gw = KnetArray(cw)
# gx = KnetArray(cx)
# gy = conv4(gw,gx;o...)
# cy1 = ones(cy)
# gy1 = KnetArray(cy1)
# cdw = conv4w(cw,cx,cy;o...)
# gdw = conv4w(gw,gx,gy;o...)
# cdx = conv4x(cw,cx,cy;o...)
# gdx = conv4x(gw,gx,gy;o...)
# p = Dict(:window=>(2,1),:alpha=>2)
# cy = pool(cx; p...)
# gy = pool(gx; p...)
# cdx = poolx(cx,cy,cy; p...)
# gdx = poolx(gx,gy,gy; p...)
# isapprox(cdx, Array(gdx))
