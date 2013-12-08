using Base.Test

using DualNumbers

function rosenbrock(x::Vector)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function rosenbrock_gradient(x::Vector)
    return [-2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1], 200.0 * (x[2] - x[1]^2)]
end

@test_approx_eq autodiff1(rosenbrock,[5.6,4.3]) rosenbrock_gradient([5.6,4.3])

using Optim

function rosenbrock_gradient!(x::Vector, storage::Vector)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
end

res1 = optimize(rosenbrock, rosenbrock_gradient!, [0.0, 0.0], method = :l_bfgs)

rosenbrock_autogradient! = autodiff1_wrapper(rosenbrock,Float64,2)
res2 = optimize(rosenbrock, rosenbrock_autogradient!, [0.0, 0.0], method = :l_bfgs)

@test_approx_eq res1.minimum res2.minimum
