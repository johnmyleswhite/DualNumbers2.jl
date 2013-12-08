module DualNumbers2

importall Base


export DualNumber,
    Dual1,
    value,
    deriv1,
    dual1,
    autodiff1,
    autodiff1_wrapper
    

abstract DualNumber <: FloatingPoint

immutable Dual1{T<:FloatingPoint} <: DualNumber
    a::T
    b::T
end

#immutable Dual2{T<:FloatingPoint} <: DualNumber
#    a::T
#    b::T
#    c::T
#end

value(x::Dual1) = x.a
deriv1(x::Dual1) = x.b
dual1(x::Number) = Dual1(x,zero(x))

@vectorize_1arg DualNumber value
@vectorize_1arg DualNumber deriv1
@vectorize_1arg Number dual1


#convert(::Type{Dual1},x::Number) = Dual1(x, zero(x))

# TODO: promote to different types of dual numbers (e.g. Dual1{Int} -> Dual1{Float64})
promote(x::Dual1, y::Dual1) = (x,y)
promote(x::Dual1, y::Number) = (x,Dual1(y,zero(y)))
promote(x::Number, y::Dual1) = (Dual1(x,zero(x)),y)

(+)(x::Dual1, y::Dual1) = Dual1(x.a+y.a, x.b+y.b)

(-)(x::Dual1, y::Dual1) = Dual1(x.a-y.a, x.b-y.b)
(-)(x::Dual1) = Dual1(-x.a, -x.b)

(*)(x::Dual1, y::Dual1) = Dual1(x.a*y.a, x.a*y.b+x.b*y.a)

(/)(x::Dual1, y::Dual1) = Dual1(x.a/y.a, (x.b*y.a - x.a*y.b)/b.a^2)

# f(a+bε) = f(a) + bf'(a)ε
# TODO: automate this with metaprogramming (Calculus.jl?)

exp(x::Dual1) = (c = exp(x.a); Dual1(c, x.b*c))
sin(x::Dual1) = Dual1(sin(x.a), x.b*cos(x.a))
cos(x::Dual1) = Dual1(cos(x.a), -x.b*sin(x.a))


function autodiff1(f, x, gradient_output, dualvec)
    # Assume f doesn't modify the input
    for i in 1:length(x)
        dualvec[i] = Dual1(x[i], zero(x[i]))
    end
    for i in 1:length(x)
        dualvec[i] = Dual1(dualvec[i].a,one(eltype(x)))
        result = f(dualvec)
        gradient_output[i] = result.b
        dualvec[i] = dual1(dualvec[i].a) # zero out derivative indicator
    end

end

function autodiff1(f,x)
    gradient_output = similar(x)
    dualvec = similar(x,Dual1{eltype(x)})
    autodiff1(f,x,gradient_output, dualvec)
    return gradient_output
end

# generates a function that computes the gradient of f(x)
# assuming that f takes a Vector{T} of length n
function autodiff1_wrapper(f,T,n)
    dualvec = Array(Dual1{T},n)
    function gradient!(x, gradient_output)
        autodiff1(f,x, gradient_output, dualvec)
    end
end


end # module
