function firstn(a,n)
    b = Any[]
    for i in a
        n == 0 && break
        push!(b,i)
        n -= 1
    end
    b
end
