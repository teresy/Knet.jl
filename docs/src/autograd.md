# AutoGrad Reference

**Contents**
```@contents
Pages = ["autograd.md"]
```

## Taking gradients

```@docs
AutoGrad.grad
AutoGrad.gradloss
```

## Checking gradients
```@docs
AutoGrad.gradcheck
```

## Defining new primitives

```@docs
AutoGrad.@primitive
AutoGrad.@zerograd
```

## How it works

```@docs
AutoGrad
```

### Forward and backward passes
```@docs
AutoGrad.forward_pass
AutoGrad.backward_pass
```

### Recording operations
```@docs
AutoGrad.recorder
AutoGrad.Rec
AutoGrad.Node
AutoGrad.Tape
AutoGrad.complete!
```

### Defining derivatives
```@docs
AutoGrad.Grad
AutoGrad.getval
```

### Higher order gradients
```@docs
AutoGrad.higher_order_gradients
```

## Function Index

```@index
Pages = ["autograd.md"]
```
