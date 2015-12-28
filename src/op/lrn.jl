type LRN <: Op; lrnN; lrnAlpha; lrnBeta; lrnK; end
lrn(x,y;lrnN=5,lrnAlpha=1e-4,lrnBeta=0.75,lrnK=2.0,o...)=(LRN(lrnN,lrnAlpha,lrnBeta,lrnK),x,y)
ninputs(::LRN)=1
overwrites(::LRN)=false
back_reads_x(::LRN)=true
back_reads_y(::LRN)=true
infersize(::LRN,xdims,ydims)=infersize(Sigm(),xdims,ydims)
forw(l::LRN, x, y; o...)=
    (cudnnLRNCrossChannelForward(x,y; lrnN=l.lrnN, lrnAlpha=l.lrnAlpha, lrnBeta=l.lrnBeta, lrnK=l.lrnK); gpusync(); y)
back(l::LRN, dy, dx; x=nothing, y=nothing, o...)=
    (dx!=nothing && cudnnLRNCrossChannelBackward(y,dy,x,dx; lrnN=l.lrnN, lrnAlpha=l.lrnAlpha, lrnBeta=l.lrnBeta, lrnK=l.lrnK); gpusync(); dx)

type DivN <: Op; lrnN; lrnAlpha; lrnBeta; lrnK; end
divn(x,y;lrnN=5,lrnAlpha=1e-4,lrnBeta=0.75,lrnK=2.0,o...)=(DivN(lrnN,lrnAlpha,lrnBeta,lrnK),x,y)
ninputs(::DivN)=1
overwrites(::DivN)=false
back_reads_x(::DivN)=false      # different from LRN
back_reads_y(::DivN)=true
infersize(::DivN,xdims,ydims)=infersize(Sigm(),xdims,ydims)
forw(l::DivN, x, y; o...)=
    (cudnnDivisiveNormalizationForward(x,y; lrnN=l.lrnN, lrnAlpha=l.lrnAlpha, lrnBeta=l.lrnBeta, lrnK=l.lrnK); gpusync(); y)
back(l::DivN, dy, dx; y=nothing, o...)=
    (dx!=nothing && cudnnDivisiveNormalizationBackward(y,dy,dx; lrnN=l.lrnN, lrnAlpha=l.lrnAlpha, lrnBeta=l.lrnBeta, lrnK=l.lrnK); gpusync(); dx) # different from LRN

