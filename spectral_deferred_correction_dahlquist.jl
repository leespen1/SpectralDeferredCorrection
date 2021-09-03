"""
Specialization of the spectral integration script, for the case of f(t, u) =
λu. Will work on this problem, then generalize to other forms in the main script.
"""

import PyCall
import Plots
SymPy = PyCall.pyimport("sympy")
import FastGaussQuadrature as FGQ # So I don't have to calculate Gauss-Legendre nodes myself, although I should be able to
#import BenchmarkTools

struct Parameters{T}
    λ::Float64
    Δt::Float64
    t_n::Float64
    c_nodes::Vector{Float64}
    times::Vector{Float64}
    Δt_0::Float64
    Δtimes::Vector{Float64}
    S::Matrix{Float64}
    u_n::T
end

function Parameters(λ::Float64, Δt::Float64, t_n::Float64, c_nodes::Vector{Float64})
    return Parameters(
        λ,
        Δt,
        t_n,
        c_nodes,
        map(x -> t_n + Δt*x, c_nodes),# Times
        c_nodes[1]*Δt,# Δt_0
        map(i -> (c_nodes[i+1]-c_nodes[i])*Δt, 1:length(c_nodes)-1),# Δtimes
        spectral_integration_matrix(c_nodes),# S
        exp(λ*t_n),# u_n
    )
end

"""
Main
"""
function main(λ::Float64, Δt::Float64, Δt_n::Float64, c_nodes::Vector{Float64})
    u(t) = exp(λ*t)
    f(t, u) = λ*u

    para = Parameters(λ, Δt, t_n, c_nodes)

    # Validation
    validate_collocation_nodes(para)
    validate_time_differences(para)

    p = no_of_points = length(c_nodes)
    no_of_corrections = 4*p

    # Prediction Phase - Standard backward Euler to fill first column
    u_vec_1 = Vector{Float64}(undef, p) # Will need to change to accomodate higher dimensions
    u_vec_1[1] = backward_euler_step(para.λ, para.Δt_0, para.u_n)
    for i in 1:length(u_vec_1)-1
        u_vec_1[i+1] = backward_euler_step(para.λ, para.Δtimes[i], u_vec_1[i])
    end

    u_vec_k = u_vec_1[:] # Copy first approximation
    # Correction Phase
    u_np1_saves = correction_phase!(u_vec_k, para, 2*p, saves=1:2*p)
    pushfirst!(u_np1_saves, (0, u_vec_1[end]))

    # Get exact values for comparison purposes
    u_exact = u(t_n+Δt)
    # Compute errors
    u_np1_saves_errors = Vector{Tuple{Int64, Float64}}(undef, length(u_np1_saves))
    for (i, (j, u_np1)) in enumerate(u_np1_saves)
        u_np1_saves_errors[i] = (j, abs(u_np1-u_exact))
    end

    # Plotting
    plot = Plots.plot(
        getindex.(u_np1_saves_errors, 1), # No of Corrections
        getindex.(u_np1_saves_errors, 2), # Error
        label="Approximation of u(t_n+Δt)"
    )
    Plots.plot!(
        title="Errors Over 1 Timestep Vs No of Corrections \nλ=$λ, Δt=$Δt, t_n=$t_n, p=$p (Gauss-Lobatto)",
        xlabel="# of Corrections (0 ⟹ Backward Euler Prediction)",
        ylabel="Abs(E)",
        #yaxis=:log,
    )

    Plots.savefig(plot, "spectral_deferred_correction_evolution.png")
    display(plot)
    return
end


"""
Returns the spectral integration matrix for the given array of collocation nodes
"""
function spectral_integration_matrix(c_nodes)::Array{Float64}
    mat_size = length(c_nodes)
    x = SymPy.Symbol("x")
    integrated_funcs = [SymPy.integrate(legrange_func(c_nodes, q), (x, 0, 1))
                        for q=1:mat_size]
    return [c_nodes[m]*integrated_funcs[q] for m=1:mat_size, q=1:mat_size]
end


"""
Returns expression (using SymPy, in terms of x) for legrange basis function

Using a SymPy expression will allow me to integrate analytically (which is
sensible, since the expression is a polynomial)
"""
function legrange_func(x_list, j)
    x = SymPy.Symbol("x")
    func = 1
    x_j = x_list[j]
    for (k, x_k) in enumerate(x_list)
        if k != j
            func = func*(x-x_k)/(x_j-x_k)
        end
    end
    return func
end


"""
Advance using implicit/backward euler.

Assumes function Dahlquist equation, y' = λy. Solving backward euler yields:
    u_n+1 = u_n + dt*λ*u_n+1
=>  u_n+1 =u_n/(1-dt*λ)
"""
function backward_euler_step(λ::Float64, dt::Float64, u)
    return u/(1-dt*λ)
end


"""
Mutable version of backward euler (assuming u is a vector)
"""
function backward_euler_step!(λ::Float64, dt::Float64, u)
    factor = 1/(1-dt*λ)
    for (i, u_entry) in enumerate(u)
        u[i] = factor*u_entry
    end
end


"""
Use equations (B15) and (B16) to make a correction (k → k+1) to the current
provided set of approximate solutions (u_vec_k).
"""
function make_correction!(u_vec_k, u_vec_kp1, u_n, λ::Float64, S::Matrix{Float64}, Δt_0::Float64, Δtimes::Vector{Float64}, Δt::Float64)
    eq_b15!(u_vec_k, u_vec_kp1, λ, S, Δt_0, Δt, u_n)
    for m in 1:length(u_vec_k)-1
        Δt_m = Δtimes[m]
        eq_b16!(u_vec_k, u_vec_kp1, λ, S, Δt_m, Δt, m)
    end
end

function make_correction!(u_vec_k, u_vec_kp1, p::Parameters)
    make_correction!(u_vec_k, u_vec_kp1, p.u_n, p.λ, p.S, p.Δt_0, p.Δtimes, p.Δt)
end


"""
Implementation of equation (B15)
"""
function eq_b15!(u_vec_k, u_vec_kp1, λ::Float64, S::Matrix{Float64}, Δt_0::Float64, Δt::Float64, u_n)
    summand(q) = S[1,q]*λ*u_vec_k[q]
    factor1 = (1/(1-Δt_0*λ))
    factor2 = u_n - Δt_0*λ*u_vec_k[1] + Δt*sum(summand,1:length(u_vec_k))
    u_vec_kp1[1] = factor1*factor2
end

function eq_b15!(u_vec_k, u_vec_kp1, p::Parameters)
    eq_b15!(u_vec_k, u_vec_kp1, p.λ, p.S, p.Δt_0, p.Δt, p.u_n)
end


"""
Implementation of equation (B16)
"""
function eq_b16!(u_vec_k, u_vec_kp1, λ::Float64, S::Matrix{Float64}, Δt_m::Float64, Δt::Float64, m::Int64)
    summand(q) = (S[m+1,q]-S[m,q])*λ*u_vec_k[q]
    factor1 = (1/(1-Δt_m*λ))
    factor2 = u_vec_kp1[m] - Δt_m*λ*u_vec_k[m+1] + Δt*sum(summand,1:length(u_vec_k))
    u_vec_kp1[m+1] = factor1*factor2
end

function eq_b16!(u_vec_k, u_vec_kp1, p::Parameters, m::Int64)
    eq_b16!(u_vec_k, u_vec_kp1, p.λ, p.S, p.Δtimes[m], p.Δt, m)
end


"""
Make sure collocation nodes have proper scale and order.
"""
function validate_collocation_nodes(c_nodes)
    for node in c_nodes
        if node < 0 || node > 1
            throw(DomainError(node, "Collocation nodes must be ∈ [0,1]."))
        end
    end
    for i in 1:length(c_nodes)-1
        if c_nodes[i+1] <= c_nodes[i]
            throw(DomainError(node, "Collocation nodes must be obey c_m+1 > c_m ∀ m = 1, ... , p."))
        end
    end
end

function validate_collocation_nodes(p::Parameters)
    validate_collocation_nodes(p.c_nodes)
end


"""
Warn user if choice of collocation nodes will cause division by zero. This
function is specific to the Dahlquist equation.
"""
function validate_time_differences(λ::Float64, Δt_0::Float64, Δtimes::Vector{Float64})
    # Should change this to a 'warning'. Julia can handle division by zero, so
    # I may actually want to use bad collocation nodes to see what happens
    if 1/(1-Δt_0*λ) in (Inf, -Inf, NaN)
        throw(DomainError(Δt_0, "Time differential causes error. Adjust collocation nodes"))
    end
    for Δt in Δtimes
        if 1/(1-Δt*λ) in (Inf, -Inf, NaN)
            throw(DomainError(Δt, "Time differential causes error. Adjust collocation nodes"))
        end
    end
end

function validate_time_differences(p::Parameters)
    validate_time_differences(p.λ, p.Δt_0, p.Δtimes)
end


"""
Swap contents of two vectors (or iterables in general) quickly and without
allocating memory. Assumes arguments are of equal length.
"""
function swap!(a_vec, b_vec)
    for i in eachindex(a_vec, b_vec)
        a_vec[i], b_vec[i] = b_vec[i], a_vec[i]
    end
end

function correction_phase!(
        u_vec_k, u_n,
        λ::Float64, S::Matrix{Float64}, Δt_0::Float64, Δtimes::Vector{Float64}, Δt::Float64,
        no_of_corrections::Int64; saves=[])

    u_vec_kp1 = u_vec_k[:]

    return_vec = []
    # Correction Phase - No animation
    for i in 1:no_of_corrections
        make_correction!(u_vec_k, u_vec_kp1, u_n, λ, S, Δt_0, Δtimes, Δt)
        swap!(u_vec_k, u_vec_kp1)
        # Save a selection of the correction process
        if i in saves
            push!(return_vec, (i, u_vec_k[end]))
        end
    end
    if !(no_of_corrections in saves) # At least return the latest correction
        push!(return_vec, (no_of_corrections, u_vec_k[end]))
    end
    return return_vec
end

function correction_phase!(u_vec_k, p::Parameters,
        no_of_corrections::Int64; saves=[])

    correction_phase!(
        u_vec_k,
        p.u_n,
        p.λ,
        p.S,
        p.Δt_0,
        p.Δtimes,
        p.Δt,
        no_of_corrections,
        saves=saves
    )
end


λ = -1.0
Δt = 1.0
t_n = 0.0
c_nodes, weights =  FGQ.gausslobatto(7)
# Shift from [-1,1] the the standard interval [0,1]
for (i, node) in enumerate(c_nodes)
    c_nodes[i] = (node+1)/2
end

main(λ, Δt, t_n, c_nodes)

