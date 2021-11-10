module SpectralDeferredCorrection
export SDC, make_SDC, make_spectral_int_matrix, gauss_lobatto_nodes

#using Polynomials, Plots, LinearAlgebra, FastGaussQuadrature,InvertedIndices
import FastGaussQuadrature
#import BenchmarkTools

#"""
#Solves the equation:
#    w + alpha*fI(t,w) = b
#For w given α, t, and b
#Do I need a best guess for w, as well?
#"""
#function implicit_solve(α, t, b)
#end

"""
Get n Gauss-Lobatto quadrature nodes, on interval [0-1].
"""
function gauss_lobatto_nodes(n::Int)
    nodes, weights =  FastGaussQuadrature.gausslobatto(n)
    return 0.5*(nodes .+ 1.0)
end

"""
Get the spectral integration matrix
"""
function make_spectral_int_matrix(n::Int)
    nodes = gauss_lobatto_nodes(n) # Perhaps add option to change quadrature in the future
    S = zeros(n,n)

    for j = 1:n
        Lt = fromroots(nodes[Not(j)])
        Li = (1.0/Lt(nodes[j]))*Lt
        for i = 1:n
            S[i,j] = integrate(Li,0.0,nodes[i])
        end
    end
    return S
end

"""
Notes for improvement:

The main usage for this function will probably be something like:
for i in 1:number_of_timesteps
    SDC(...)

So I don't want to waste resources recomputing S everytime, or setting up all
the equations.

Perhaps I should have a create_SDC function? This would be provided all
parameters except the initial guess, and create an SDC function which only
needs the initial values as arguments.

From a UX perspective, I like that they will have an SDC function whose only
arguments are the initial data. That would be very easy to use. But I don't
like that they have to call a function to create a function. That seems like it
might be confusing.

Actually, I think it would be good to also have Δt as an argument to the
user-SDC function. I can imagine a user might want to use variable timestep
size (but they probably don't need to vary the number of collocation nodes,
implicit/explicit f, etc).
"""
function SDC(n, fE, fI, t, Δt, u, implicit_solver)
    nodes = gauss_lobatto_nodes(n)
    n_cor = 2*n-3
    S = make_spectral_int_matrix(n) # nodes will be computed twice; should make more efficient
    subtimes = (Δt*nodes).+ t 
    subvals = Vector{typeof(u)}(undef, n)

    subvals[1] = u
    # Prediction
    for m in 1:n-1
        subvals[m+1] = begin
            u_m = subvals[m]
            t_m = subtimes[m]
            t_mp1 = subtimes[m+1]
            Δt_m = t_mp1 - t_m
            RHS = u_m + Δt_m*fE(t_m,u_m)
            implicit_solver(-Δt_m, t_m, RHS)
        end
    end

    f(t, u) = fE(t, u) + fI(t, u)
    # Correction
    for k in 1:n_cor
        # B15 - Not needed for gauss lobatto nodes, just leaves subvals as they were
        # But B15 will be needed if start point is not a node
        f_subvals = [f(subtimes[i], subvals[i]) for i in 1:n]
        u_m_k = subvals[1]
        for m in 1:n-1
            #B16
            u_m_k_next = subvals[m+1] # Should figure out better notation for this
            subvals[m+1] = begin
                u_m_kp1 = subvals[m]
                t_m = subtimes[m]
                t_mp1 = subtimes[m+1]
                Δt_m = t_mp1 - t_m
                u_mp1_k = subvals[m+1]
                ΔfE = fE(t_m, u_m_kp1) - fE(t_m, u_m_k)
                RHS = subvals[m] + Δt_m*(ΔfE - f(t_mp1, u_mp1_k))
                RHS += sum(q -> (S[m+1,q]-S[m,q])*f_subvals[q], 1:size(S)[2])
                implicit_solver(-Δt_m, fI, RHS)
            end
            u_m_k = u_m_k_next
        end
    end
    # Will be different if endpoint is a node, need to use B6
    return subvals[end]
end


"""
This function returns a function: an SDC solver which needs only the initial
conditions as the arguments. Moreover, the quadrature and spectral integration
matrix are built-in to the function, not calculated each run.
"""
function make_SDC(n, fE, fI, implicit_solver)
    nodes = gauss_lobatto_nodes(n)
    n_cor = 2*n-3
    S = make_spectral_int_matrix(n) # nodes will be computed twice; should make more efficient

    return function small_arg_list_SDC(u, t, Δt)
        subtimes = (Δt*nodes).+ t 
        subvals = Vector{typeof(u)}(undef, n)
        subvals[1] = u
        # Prediction
        for m in 1:n-1
            subvals[m+1] = begin
                u_m = subvals[m]
                t_m = subtimes[m]
                t_mp1 = subtimes[m+1]
                Δt_m = t_mp1 - t_m
                RHS = u_m + Δt_m*fE(t_m,u_m)
                implicit_solver(-Δt_m, fI, RHS)
            end
        end

        f(t, u) = fE(t, u) + fI(t, u)
        # Correction
        for k in 1:n_cor
            # B15 - Not needed for gauss lobatto nodes, just leaves subvals as they were
            # But B15 will be needed if start point is not a node
            f_subvals = [f(subtimes[i], subvals[i]) for i in 1:n]
            u_m_k = subvals[1]
            for m in 1:n-1
                #B16
                u_m_k_next = subvals[m+1] # Should figure out better notation for this
                subvals[m+1] = begin
                    u_m_kp1 = subvals[m]
                    t_m = subtimes[m]
                    t_mp1 = subtimes[m+1]
                    Δt_m = t_mp1 - t_m
                    u_mp1_k = subvals[m+1]
                    ΔfE = fE(t_m, u_m_kp1) - fE(t_m, u_m_k)
                    RHS = subvals[m] + Δt_m*(ΔfE - f(t_mp1, u_mp1_k))
                    RHS += sum(q -> (S[m+1,q]-S[m,q])*f_subvals[q], 1:size(S)[2])
                    implicit_solver(-Δt_m, fI, RHS)
                end
                u_m_k = u_m_k_next
            end
        end
        # Will be different if endpoint is a node, need to use B6
        return subvals[end]
    end
end

#function graph(u_np1_saves, T::Timing;
#              graph_name::String="SDC_evolution.png",
#              display_plot::Bool=false)
#
#    # Get exact value for comparison purposes
#    u_exact = u(T.t_n + T.Δt)
#    # Compute errors
#    u_np1_saves_errors = Vector{Tuple{Int64, Float64}}(undef, length(u_np1_saves))
#    for (i, (j, u_np1)) in enumerate(u_np1_saves)
#        #u_np1_saves_errors[i] = (j, abs(u_np1-u_exact))
#        u_np1_saves_errors[i] = (j, u_np1 - u_exact)
#    end
#
#    # Plotting
#    plot = Plots.plot(
#        getindex.(u_np1_saves_errors, 1), # No of Corrections
#        getindex.(u_np1_saves_errors, 2), # Error
#        label="Approximation of u(t_n+Δt)"
#    )
#    p = length(T.Δtimes)+1
#    Plots.plot!(
#        plot,
#        title="Error Over 1 Timestep Vs No of Corrections \nλ=$λ, Δt=$(T.Δt), t_n=$(T.t_n), p=$p (Gauss-Lobatto)",
#        xlabel="# of Corrections (0 ⟹ Initial Prediction)",
#        ylabel="Error",
#        xlim=(0, 2*p),
#        #yaxis=:log,
#    )
#    # Make a line on y=0, for easier comprehension
#    Plots.hline!(plot, [0], label="")
#
#    Plots.savefig(plot, graph_name)
#
#    if display_plot
#        display(plot)
#    end
#    return nothing
#end


#"""
#Make sure collocation nodes have proper scale and order.
#"""
#function validate_collocation_nodes(c_nodes)
#    for node in c_nodes
#        if node < 0 || node > 1
#            throw(DomainError(node, "Collocation nodes must be ∈ [0,1]."))
#        end
#    end
#    for i in 1:length(c_nodes)-1
#        if c_nodes[i+1] <= c_nodes[i]
#            throw(DomainError(node, "Collocation nodes must be obey c_m+1 > c_m ∀ m = 1, ... , p."))
#        end
#    end
#end


end

