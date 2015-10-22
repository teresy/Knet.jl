using Knet
@show hidden = 1024
@show winit = Gaussian(0,.01)
@show lr = 2.0
@show gclip = 10.0
maxnorm = zeros(2)
losscnt = zeros(2)

#data=S2SData("ptb.train.txt";dict="ptb.train.txt")
sgen1=SketchEngine(`xzcat sdfkl32KCsd_enTenTen12.vert.xz`; dict="wdict10022")
sgen2=SketchEngine(`xzcat sdfkl32KCsd_enTenTen12.vert.xz`; dict="wdict10022")
data=S2SData(sgen1,sgen2)
@show vocab = maxtoken(data,2)

model = S2S(lstm; hidden=hidden, vocab=vocab, winit=winit)
setopt!(model; lr=lr)
train(model, data, softloss; gclip=gclip, maxnorm=maxnorm, losscnt=losscnt)

:ok

