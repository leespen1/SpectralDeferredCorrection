using Polynomials, Plots, LinearAlgebra, FastGaussQuadrature,InvertedIndices


function computeS(n::Int)
nodes, weights = gausslobatto(n);
nodes = 0.5*(nodes .+ 1.0)

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


n = 4
S  = computeS(n)