"""
What I want to do:

Step 1: Implement SDC using explicit methods, rather than implicit
- Try to do it in a clear and clean way
- Reuse data when possible

Step 2: Implement room for implicit methods later on

Step 3: Implement implicit


"""

import PyCall
import Plots
SymPy = PyCall.pyimport("sympy")
import FastGaussQuadrature as FGQ # So I don't have to calculate Gauss-Legendre nodes myself, although I should be able to
#import BenchmarkTools

λ=-1.0

function u(t::Float64)
    exp(λ*t)
end

"""
The main definition of f. Behavior of all f-related functions (f_exp, f_imp,
etc, will refer back to this one)
"""
function f(t::Float64, u)
    λ*u
    #λ*exp(λ*t)
end


function f(u)
    f(0.0, u) # Time doesn't matter for dahlquist equation
end


function f_exp(t::Float64, u)
    f(t, u)
end


function f_exp(u)
    f(u)
end


function Δf_exp(t::Float64, u_kp1, u_k)
    f_exp(t, u_kp1)-f_exp(t,u_k)
end


function Δf_exp(u_kp1, u_k)
    f_exp(u_kp1)-f_exp(u_k)
end


function prediction_step(t::Float64, u, dt::Float64)
    u + dt*f_exp(t, u)
end


function prediction_step(u, dt::Float64)
    u + dt*f_exp(u)
end


function gauss_lobatto_nodes(n::Int64)
    c_nodes, weights =  FGQ.gausslobatto(n)
    # Shift from [-1,1] the the standard interval [0,1]
    for (i, node) in enumerate(c_nodes)
        c_nodes[i] = (node+1)/2
    end
    return c_nodes
end


struct Timing
    t_n::Float64
    Δt::Float64
    Δt_0::Float64
    Δtimes::Vector{Float64}
    times::Vector{Float64}
end


function Timing(t_n::Float64, Δt::Float64, c_nodes::Vector{Float64})
    Timing(
        t_n,
        Δt,
        c_nodes[1]*Δt,#Δt_0
        map(i -> Δt*(c_nodes[i+1]-c_nodes[i]), 1:length(c_nodes)-1),#Δtimes
        map(i -> t_n+Δt*c_nodes[i], 1:length(c_nodes)),#times
    )
end


function perform_timestep(u_n, T::Timing, S::Matrix{Float64}; get_interstep=false)
    p = no_of_points = length(T.Δtimes)+1
    no_of_corrections = 2*p

    # Prediction Phase
    u_vec_1 = Vector{Float64}(undef, p) # Will need to change to accomodate higher dimensions
    u_vec_1[1] = prediction_step(T.t_n, u_n, T.Δt_0)
    for i in 1:length(u_vec_1)-1
        u_vec_1[i+1] = prediction_step(T.times[i], u_vec_1[i], T.Δtimes[i])
    end

    u_vec_k = u_vec_1[:] # Copy first approximation

    # Correction Phase
    if get_interstep
        u_np1_saves, corr_matrix = correction_phase!(u_vec_k, u_n, S, T, 2*p, saves=1:2*p, get_interstep=get_interstep)
    else
        u_np1_saves = correction_phase!(u_vec_k, u_n, S, T, 2*p, saves=1:2*p, get_interstep=get_interstep)
    end
    pushfirst!(u_np1_saves, (0, u_vec_1[end])) # Put prediction at start of saves

    if get_interstep
        return u_np1_saves, corr_matrix
    else
        return u_np1_saves
    end
end


function graph(u_np1_saves, T::Timing;
              graph_name::String="SDC_explicit_evolution.png",
              display_plot::Bool=false)

    # Get exact value for comparison purposes
    u_exact = u(T.t_n + T.Δt)
    # Compute errors
    u_np1_saves_errors = Vector{Tuple{Int64, Float64}}(undef, length(u_np1_saves))
    for (i, (j, u_np1)) in enumerate(u_np1_saves)
        #u_np1_saves_errors[i] = (j, abs(u_np1-u_exact))
        u_np1_saves_errors[i] = (j, u_np1 - u_exact)
    end

    # Plotting
    plot = Plots.plot(
        getindex.(u_np1_saves_errors, 1), # No of Corrections
        getindex.(u_np1_saves_errors, 2), # Error
        label="Approximation of u(t_n+Δt)"
    )
    p = length(T.Δtimes)+1
    Plots.plot!(
        plot,
        title="Error Over 1 Timestep Vs No of Corrections \nλ=$λ, Δt=$(T.Δt), t_n=$(T.t_n), p=$p (Gauss-Lobatto)",
        xlabel="# of Corrections (0 ⟹ Initial Prediction)",
        ylabel="Error",
        xlim=(0, 2*p),
        #yaxis=:log,
    )
    # Make a line on y=0, for easier comprehension
    Plots.hline!(plot, [0], label="")

    Plots.savefig(plot, graph_name)

    if display_plot
        display(plot)
    end
    return nothing
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
Swap contents of two vectors (or iterables in general) quickly and without
allocating memory. Assumes arguments are of equal length.
"""
function swap!(a_vec, b_vec)
    for i in eachindex(a_vec, b_vec)
        a_vec[i], b_vec[i] = b_vec[i], a_vec[i]
    end
end


function correction_phase!(
        u_vec_k, u_n, S::Matrix{Float64}, T::Timing,
        no_of_corrections::Int64; saves=[], get_interstep=false)

    if get_interstep
        corr_matrix = zeros(Float64, length(u_vec_k), no_of_corrections+1)
        corr_matrix[:, 1] = u_vec_k[:]
    end

    u_vec_kp1 = u_vec_k[:]

    return_vec = []
    # Correction Phase - No animation
    for i in 1:no_of_corrections
        make_correction!(u_vec_k, u_vec_kp1, u_n,S, T)
        swap!(u_vec_k, u_vec_kp1)


        # Save a selection of the correction process
        if i in saves
            push!(return_vec, (i, u_vec_k[end]))
            if get_interstep
                corr_matrix[:, i+1] = u_vec_k[:] # For debugging
            end
        end
    end
    if !(no_of_corrections in saves) # At least return the latest correction
        push!(return_vec, (no_of_corrections, u_vec_k[end]))
    end

    if get_interstep
        return  return_vec, corr_matrix
    else
        return return_vec
    end
end


"""
Use equations (B15) and (B16) to make a correction (k → k+1) to the current
provided set of approximate solutions (u_vec_k).
"""
function make_correction!(u_vec_k, u_vec_kp1, u_n, S::Matrix{Float64}, T::Timing)
    eq_b15!(u_vec_k, u_vec_kp1, u_n, T.Δt, T.times, S)
    for m in 1:length(u_vec_k)-1
        eq_b16!(u_vec_k, u_vec_kp1, T.Δt, T.times, T.Δtimes, S, m)
    end
end



"""
Implementation of equation (B15)
"""
function eq_b15!(u_vec_k, u_vec_kp1, u_n, Δt::Float64, times, S::Matrix{Float64})
    u_vec_kp1[1] = begin
        summand(q) = S[1,q]*f(times[q], u_vec_k[q])
        u_n + Δt*sum(summand,1:size(S)[2]) # sum over first row of S
    end
end


"""
Implementation of equation (B16)
"""
function eq_b16!(u_vec_k, u_vec_kp1, Δt::Float64, times, Δtimes::Vector{Float64},
                 S::Matrix{Float64}, m::Int64)
    u_vec_kp1[m+1] = begin
        summand(q) = (S[m+1,q]-S[m,q])*f(times[q], u_vec_k[q])
        term1 = u_vec_kp1[m] + Δtimes[m]*(Δf_exp(times[m], u_vec_kp1[m], u_vec_k[m]))
        term1 + Δt*sum(summand, 1:size(S)[2])
    end
end


function main()
    n = 6
    t_n, Δt = 0.0, 1.0
    u_n = u(t_n)
    #c_nodes = gauss_lobatto_nodes(n)
    c_nodes = [0.0, 0.5, 1.0]
    T = Timing(t_n, Δt, c_nodes)
    S = spectral_integration_matrix(c_nodes)

    #println("Collocation Nodes:")
    #display(c_nodes)
    println("Spectral Integration Matrix")
    display(S)

    #println("Timing Struct:")
    #display(T)

    u_np1_saves = perform_timestep(u_n, T, S, get_interstep=false)
    u_np1_saves, corr_matrix = perform_timestep(u_n, T, S, get_interstep=true)
    true_values = [u(t) for (i, t) in enumerate(T.times)]
    err_matrix = corr_matrix[:, :]
    for i in 1:size(err_matrix)[2]
        err_matrix[:, i] = corr_matrix[:, i] - true_values
    end

    println("\nInternode Approximations")
    display(corr_matrix)
    println("\nInternode Errors")
    display(err_matrix)

    #println("Results:")
    #display(u_np1_saves)

    graph(u_np1_saves, T, display_plot=true)
end
#c_nodes = [0.0, 0.5, 1.0]
#dir_name = "FigureSaves"
#if !isdir(dir_name)
#    mkdir(dir_name)
#end
#no_of_runs = 50
#rand_λs = -5*rand(Float64, no_of_runs)
#rand_Δts = 2*rand(Float64, no_of_runs)
#rand_t_ns = 5*rand(Float64, no_of_runs)
#for (i, (λ, Δt, t_n)) in enumerate(zip(rand_λs, rand_Δts, rand_t_ns))
#    myround(x) = round(x, digits=2)
#    λ, Δt, t_n = myround(λ), myround(Δt), myround(t_n)
#    main(λ, Δt, t_n, c_nodes, graph_name="$dir_name/figure$i.png", display_plot=false)
#end

main()
