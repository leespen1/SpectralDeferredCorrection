"""
Specialization of the spectral integration script, for the case of f(t, u) =
λu. Will work on this problem, then generalize to other forms in the main script.
"""

import PyCall
import Plots
SymPy = PyCall.pyimport("sympy")
import FastGaussQuadrature as FGQ # So I don't have to calculate Gauss-Legendre nodes myself, although I should be able to
#import BenchmarkTools


"""
Main
"""
function main()
    # Function Paramaters
    λ = 0.5
    u(t) = exp(λ*t)
    f(t, u) = λ*u

    # Initial Conditions / Approximation Parameters
    #c_nodes = range(0, 1, length=10) # Old method, equally spaced nodes
    c_nodes, weights =  FGQ.gausslegendre(10)
    # Shift from [-1,1] the the standard interval [0,1]
    for (i, node) in enumerate(c_nodes)
        c_nodes[i] = (node+1)/2
    end
    validate_collocation_nodes(c_nodes)

    start_t = 0.5
    u_n = u(start_t)
    Δt = 7.0


    times = map(x -> start_t + Δt*x, c_nodes)
    Δt_0 = c_nodes[1]*Δt
    Δtimes = map(i -> (c_nodes[i+1]-c_nodes[i])*Δt, 1:length(c_nodes)-1)
    validate_time_differences(λ, Δt_0, Δtimes)

    p = no_of_points = length(c_nodes)
    no_of_corrections = 4*p

    S = spectral_integration_matrix(c_nodes)

    # Get exact values for comparison purposes
    u_vec_exact = [u(t) for t in times]

    # Prediction Phase - Standard backward Euler to fill first column
    u_vec_1 = Vector{Float64}(undef, length(c_nodes))
    u_vec_2 = Vector{Float64}(undef, length(c_nodes))
    u_vec_p = Vector{Float64}(undef, length(c_nodes))
    u_vec_2p = Vector{Float64}(undef, length(c_nodes))
    u_vec_4p = Vector{Float64}(undef, length(c_nodes))

    u_vec_1[1] = u(start_t)
    for i in 1:length(u_vec_1)-1
        u_vec_1[i+1] = backward_euler_step(λ, Δtimes[i], u_vec_1[i])
    end

    u_vec_k = u_vec_1[:] # Copy first approximation
    u_vec_kp1 = Vector{Float64}(undef, length(c_nodes))

    # Correction Phase - No animation
    for i in 1:no_of_corrections
        make_correction!(u_vec_k, u_vec_kp1, u_n, λ, S, Δt_0, Δtimes, Δt)
        swap!(u_vec_k, u_vec_kp1)
        if i == 2
            u_vec_2 .= u_vec_k[:]
        elseif i == p
            u_vec_p .= u_vec_k[:]
        elseif i == 2*p
            u_vec_2p .= u_vec_k[:] # Save Copy of 2pth Correction
        elseif i == 4*p
            u_vec_4p .= u_vec_k[:]
        end
    end

    ## Correction Phase - With animation
    #animation = Plots.@animate for i in 1:no_of_corrections
    #    make_correction!(u_vec_k, u_vec_kp1, u_n, λ, S, Δt_0, Δtimes, Δt)
    #    swap!(u_vec_k, u_vec_kp1)

    #    errors_pred = [abs(u) for u in u_vec_exact - u_vec_1]
    #    errors_corr = [abs(u) for u in u_vec_exact - u_vec_k]
    #    plot = Plots.plot(
    #        errors_pred[2:end],
    #        label="Prediction",
    #    )
    #    Plots.plot!(
    #        errors_corr[2:end],
    #        label = "Correction - K = $i"
    #    )
    #    if i == 2
    #        u_vec_2 .= u_vec_k[:]
    #    elseif i == p
    #        u_vec_p .= u_vec_k[:]
    #    elseif i == 2*p
    #        u_vec_2p .= u_vec_k[:] # Save Copy of 2pth Correction
    #    elseif i == 4*p
    #        u_vec_4p .= u_vec_k[:]
    #    end
    #end
    #fps = floor(no_of_corrections/5) # fps needed for a 5 second animation
    #fps = Int(min(fps, 30)) # Don't exceed 30 fps (to keep easy to interpret, should maybe use lower floor)
    #Plots.gif(animation, "spectral_deferred_correction_evolution.gif", fps=fps)

    # At this point, u_vec_1 holds the prediction, and u_vec_k holds the most
    # recent revision. If I want to analyze the evolution of the revisions, I
    # would have to save them during the correction phase.
    errors_pred = [abs(u) for u in u_vec_exact - u_vec_1]
    errors_corr_2 = [abs(u) for u in u_vec_exact - u_vec_2]
    errors_corr_p = [abs(u) for u in u_vec_exact - u_vec_p]
    errors_corr_2p = [abs(u) for u in u_vec_exact - u_vec_2p]
    errors_corr_4p = [abs(u) for u in u_vec_exact - u_vec_4p]

    ## In case I want to look at the fractional error.
    #frac_errors_pred = [abs((u_est-u_ex)/u_ex) for (u_est, u_ex) in zip(u_vec_1, u_vec_exact)]
    #frac_errors_corr = [abs((u_est-u_ex)/u_ex) for (u_est, u_ex) in zip(u_vec_2p, u_vec_exact)]

    ## For looking at some of the results by hand
    ##sample(u) = u[1:Int(floor(p/5)):end]
    #sample(u) = u
    ##println("\nS - Spectral Integration Matrix")
    ##display(S)
    #println("\nCollocation Nodes")
    #display(c_nodes)
    #println("\nExact Values")
    #display(sample(u_vec_exact))
    #println("\nFirst Prediction")
    #display(sample(u_vec_1))
    #println("\nFinal Revision")
    #display(sample(u_vec_k))
    #println("\nErrors Prediction")
    #display(sample(errors_pred))
    #println("\nErrors Correction")
    #display(sample(errors_corr_2p))
    #println()
    
    Δtimes_plot = map(t -> t - start_t, times[2:end])
    #pushfirst!(Δtimes_plot, Δt_0)
    errors_to_include = [
        errors_pred,
        errors_corr_2,
        errors_corr_p,
        errors_corr_2p,
        #errors_corr_4p,
    ]
    errors_to_include = map(x -> x[2:end], errors_to_include)

    display(Δtimes_plot)
    display(errors_to_include)

    plot2 = Plots.plot(
        Δtimes_plot,
        errors_to_include,
        #label=["Prediction", "K=1", "K=p", "K=2p", "K=4p"]
    )

    #plot = Plots.plot(
    #    #Δtimes_plot,
    #    errors_pred[2:end],
    #    label="Prediction",
    #)
    #Plots.plot!(
    #    errors_corr_2[2:end],
    #    label="Corrected Values (K=2)",
    #)
    #Plots.plot!(
    #    errors_corr_p[2:end],
    #    label="Corrected Values (K=p)",
    #)
    #Plots.plot!(
    #    errors_corr_2p[2:end],
    #    label="Corrected Values (K=2p)",
    #)
    #Plots.plot!(
    #    errors_corr_4p[2:end],
    #    label="Corrected Values (K=4p)"
    #)

    all_errors = append!(
        [],
        errors_pred[2:end],
        errors_corr_2[2:end],
        errors_corr_p[2:end],
        errors_corr_2p[2:end],
        errors_corr_4p[2:end]
    )

    #Plots.plot!(
    #    plot,
    #    title="Error Analysis - Prediction vs Corrections",
    #    #xticks=Δtimes_plot,
    #    xlabel="Δt",
    #    ylabel="Abs(E)",
    #    #ylabel="\$\\left|\\frac{u_{est}-u_{ex}}{u_{ex}}\\right|\$",
    #    ylim=(minimum(all_errors), maximum(all_errors)),
    #    yaxis=:log,
    #    marker=:cross,
    #)

    #savefig(plot, "spectral_deferred_correction_evolution.png")
    #display(plot2)
    display(plot2)
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


"""
Implementation of equation (B15)
"""
function eq_b15!(u_vec_k, u_vec_kp1, λ::Float64, S::Matrix{Float64}, Δt_0::Float64, Δt::Float64, u_n)
    summand(q) = S[1,q]*λ*u_vec_k[q]
    factor1 = (1/(1-Δt_0*λ))
    factor2 = u_n - Δt_0*λ*u_vec_k[1] + Δt*sum(summand,1:length(u_vec_k))
    u_vec_kp1[1] = factor1*factor2
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


"""
Swap contents of two vectors (or iterables in general) quickly and without
allocating memory. Assumes arguments are of equal length.
"""
function swap!(a_vec, b_vec)
    for i in eachindex(a_vec, b_vec)
        a_vec[i], b_vec[i] = b_vec[i], a_vec[i]
    end
end

main()
