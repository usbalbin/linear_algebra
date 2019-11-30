/**
 * R, K to 1D index
 */
#define ix(index_r, index_k, cols) ((index_r) * (cols) + (index_k))

/**
 * Compute res = lhs {op} rhs for two matrices
 *
 * lhs: elem_count
 * rhs: elem_count
 * res: elem_count
 */
kernel void {op_name}_mat_{T}(global const {T} *lhs, global const {T} *rhs, global {T} *res) {
    int index_r = get_global_id(1);
    int index_k = get_global_id(0);

    int k = get_global_size(0);
    int index = ix(index_r, index_k, k);

    res[index] = lhs[index] {op} rhs[index];
}

/**
 * Compute lhs = lhs {op} rhs for two matrices
 *
 * lhs: elem_count
 * rhs: elem_count
 */
kernel void {op_name}_assign_mat_{T}(global {T} *lhs, global const {T} *rhs) {
    int index_r = get_global_id(1);
    int index_k = get_global_id(0);

    int k = get_global_size(0);
    int index = ix(index_r, index_k, k);

    lhs[index] {op}= rhs[index];
}
