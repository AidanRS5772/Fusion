function group_combination_count(n::Int, k::Int)
    # Step 1: Compute total number of valid group subsets of size ≥ m
    T = sum(binomial(n, r) for r in 3:n)

    # Step 2: Number of multisets (combinations with repetition)
    return binomial(T + k - 1, k)
end

v = 8
f = 6
println(2^(f - 1))
println(group_combination_count(v, f))
