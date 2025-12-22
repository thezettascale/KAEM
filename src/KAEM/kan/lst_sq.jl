module LstSqSolver

export regularize, forward_elimination, backward_substitution

using Lux

function eyeG(G)
    return 1:G .== (1:G)' |> Lux.f32
end

function regularize(B_i, y_i, basis; ε = 1.0f-4, init = false)
    J, O, G = basis.I, basis.O, basis.G
    S = init ? size(B_i, 2) : basis.S

    B_perm = reshape(B_i, G, 1, S, 1, J)
    B_perm_transpose = reshape(B_perm, 1, G, S, 1, J)
    A = dropdims(
        sum(
            B_perm .* B_perm_transpose; dims = 3
        ); dims = 3
    ) # G x G x 1 x J

    y_perm = reshape(y_i, 1, 1, S, O, J)
    b = dropdims(
        sum(
            B_perm .* y_perm; dims = 3
        ); dims = 3
    ) # G x 1 x O x J

    A = A .+ ε .* eyeG(G)
    return A .* 1.0f0, b .* 1.0f0
end

function eliminator(
        k,
        A,
        b,
        lower_mask_all,
        upper_mask_all,
    )
    lower_mask = lower_mask_all[:, k]
    upper_mask = upper_mask_all[:, k]

    pivot = A[k:k, k:k, :, :]
    pivot_row = A[k:k, :, :, :]
    pivot_col = A[:, k:k, :, :]

    factors = pivot_col .* lower_mask ./ pivot

    # Rank-1 update
    elimination_mask = lower_mask * upper_mask'
    A = A .- (factors .* pivot_row) .* elimination_mask

    pivot_b = b[k:k, :, :, :]
    b = b .- factors .* pivot_b
    return A, b
end

function backsubber(
        k,
        coef,
        A,
        b,
        k_mask_all,
        upper_mask_all,
        J, O, G
    )
    k_mask = k_mask_all[:, k]
    upper_mask = upper_mask_all[:, k]

    diag_elem = A[k:k, k:k, :, :]
    rhs_elem = b[k:k, :, :, :]

    upper_row = A[k:k, :, :, :]
    upper_coef = PermutedDimsArray(coef .* upper_mask, (2, 1, 3, 4))
    sum_term = sum(upper_row .* upper_coef; dims = 2)

    new_coef_k = (rhs_elem .- sum_term) ./ diag_elem
    coef = coef .+ new_coef_k .* k_mask
    return coef
end

function pivot_onehot(k, A, lower_mask_all, G; tie_bias = 1.0f-6)
    """Returns a one-hot vector selecting the pivot position p ∈ {k..G}"""
    cand = lower_mask_all[:, k]

    diag_vals = sum(A .* eyeG(G); dims = 2)
    score = dropdims(maximum(abs.(diag_vals); dims = 4); dims = 4)

    # Mask out i < k and add bias so argmax is unique (prefer smallest i on ties)
    idx = (1:G) |> Lux.f32
    score_biased = score .* cand .- tie_bias .* idx

    # Make one-hot by equality with max (unique because of bias)
    m = maximum(score_biased)
    e_p = (score_biased .== m) |> Lux.f32
    return e_p
end

function swap_rows_onehot(A, k, e_p, G)
    """Swaps rows k and p in the first dimension of A using one-hot e_p"""
    e_k = ((1:G) .== k) |> Lux.f32

    row_k = A[k:k, :, :, :]
    row_p = sum(A .* e_p; dims = 1)

    # A_new = A + (row_p-row_k)*e_k + (row_k-row_p)*e_p
    A_new = A .+
        (row_p .- row_k) .* e_k .+
        (row_k .- row_p) .* e_p

    return A_new
end

function swap_cols_onehot(A, k, e_p, G)
    """Swaps columns k and p in the second dimension of A using one-hot e_p"""
    e_k = ((1:G) .== k) |> Lux.f32

    col_k = A[:, k:k, :, :]
    col_p = sum(A .* reshape(e_p, 1, G); dims = 2)

    A_new = A .+
        (col_p .- col_k) .* reshape(e_k, 1, G) .+
        (col_k .- col_p) .* reshape(e_p, 1, G)

    return A_new
end

function swap_b_rows_onehot(b, k, e_p, G)
    """Swaps corresponding rows of b using one-hot e_p"""
    e_k = ((1:G) .== k) |> Lux.f32

    b_k = b[k:k, :, :, :]
    b_p = sum(b .* e_p; dims = 1)

    b_new = b .+
        (b_p .- b_k) .* e_k .+
        (b_k .- b_p) .* e_p

    return b_new
end

function swap_P_rows_onehot(P, k, e_p, G)
    """Swaps rows k and p in the first dimension of P using one-hot e_p"""
    e_k = ((1:G) .== k) |> Lux.f32

    row_k = sum(P .* e_k; dims = 1)
    row_p = sum(P .* e_p; dims = 1)

    P_new = P .+
        (row_p .- row_k) .* e_k .+
        (row_k .- row_p) .* e_p

    return P_new
end

function unpermute_coef(P, coef_perm, G, O, J)
    P_T = PermutedDimsArray(P, (2, 1, 3))
    coef_out = sum(
        P_T .* reshape(coef_perm, 1, G, O, J);
        dims = 2
    )
    return coef_out
end

function forward_elimination(
        A,
        b,
        basis;
        ε = 1.0f-4,
    )
    G = basis.G
    lower_mask_all = (basis.lower_mask .* 1.0f0) |> Lux.f32
    upper_mask_all = (basis.upper_mask .* 1.0f0) |> Lux.f32

    state = (A, b, eyeG(G))
    for k in 1:(G - 1)
        A_acc, b_acc, P_acc = state

        # Pivot: find largest diagonal element in remaining rows/cols
        e_p = pivot_onehot(k, A_acc, lower_mask_all, G; tie_bias = ε)

        # Symmetric swap on A, swap rows of b, and track P
        A_swapped = swap_rows_onehot(A_acc, k, e_p, G)
        A_swapped = swap_cols_onehot(A_swapped, k, e_p, G)
        b_swapped = swap_b_rows_onehot(b_acc, k, e_p, G)
        P_new = swap_P_rows_onehot(P_acc, k, e_p, G)

        # Elimination on swapped A,b
        A_new, b_new = eliminator(
            k,
            A_swapped,
            b_swapped,
            lower_mask_all,
            upper_mask_all,
        )
        state = (A_new, b_new, P_new)
    end

    return state
end

function backward_substitution(
        A,
        b,
        basis,
        P,
    )
    J, O, G = basis.I, basis.O, basis.G
    k_mask_all = (basis.k_mask .* 1.0f0) |> Lux.f32
    upper_mask_all = (basis.lower_mask .* 1.0f0) |> Lux.f32

    coef_perm = zero(b)
    for k in G:-1:1
        coef_perm = backsubber(
            k,
            coef_perm,
            A,
            b,
            k_mask_all,
            upper_mask_all,
            J, O, G
        )
    end

    # Unpermute the solution
    coef = unpermute_coef(P, coef_perm, G, O, J)
    return coef
end

end
