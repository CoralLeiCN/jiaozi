# seems similar
@numba.jit(nopython=True)
def add_at_numba1(xt, idx_a, idx_b):
    for i in range(len(idx_a)):
        xt[idx_a[i], idx_b[i]] += 1

@numba.jit(nopython=True)
def add_at_numba2(xt, idx_a, idx_b):
    for i, j in zip(idx_a, idx_b):
       xt[i, j] += 1

a,b = df['animal'].values, df['color'].values
uniq_vals_a, idx_a = custom_np_unqiue_with_inverse(a)
uniq_vals_b, idx_b = custom_np_unqiue_with_inverse(b)
shape_xt = (uniq_vals_a.size, uniq_vals_b.size)
xt = np.zeros(shape_xt, dtype='uint')

%%timeit
add_at_numba1(xt, idx_a, idx_b)

%%timeit
add_at_numba2(xt, idx_a, idx_b)