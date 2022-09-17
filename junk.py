
    # Broadcast, see https://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/
    # Inputs: array A with m dimensions; array B with n dimensions
        # p = max(m, n)
        # if m < p:
            # left-pad A's shape with 1s until it also has p dimensions
        # else if n < p:
            # left-pad B's shape with 1s until is also has p dimensions
        # result_dims = new list with p elements
        # for i in p-1 ... 0:
            # A_dim_i = A.shape[i]
            # B_dim_i = B.shape[i]
            # if A_dim_i != 1 and B_dim_i != 1 and A_dim_i != B_dim_i:
                # raise ValueError("could not broadcast")
            # else:
                # result_dims[i] = max(A_dim_i, B_dim_i)

    # Hence, broadcasting is symmetric in A and B.
    # And, A op B is commutative iff op is commutative.


