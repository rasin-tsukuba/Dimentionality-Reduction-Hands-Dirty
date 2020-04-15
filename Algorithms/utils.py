import sklearn

def compute_svd(data, n_components,
                n_iters, random_state):
    U, s, V = sklearn.utils.extmath.randomized_svd(
        data,
        n_components,
        n_iters,
        random_state
    )

    U, V = sklearn.utils.extmath.svd_flip(U, V)

    return U, s, V