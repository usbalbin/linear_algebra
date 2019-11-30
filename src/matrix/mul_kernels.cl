/**
 * R, K to 1D index
 */
#define ix(index_r, index_k, cols) ((index_r) * (cols) + (index_k))

/**
 * Multiply matrix by matrix
 *
 * A: RxM
 * B: MxK
 * C: RxK
 */
kernel void mul_mat_{T}(global const {T} *a, global const {T} *b, global {T} *c, uint m) {
    int index_r = get_global_id(1);
    int index_k = get_global_id(0);

    int r = get_global_size(1);
    int k = get_global_size(0);

    {T} sum = 0;
    for(int index_m = 0; index_m < m; index_m++) {
        sum += a[ix(index_r, index_m, m)] * b[ix(index_m, index_k, k)];
    }
    c[ix(index_r, index_k, k)] = sum;
}

/**
 * Multiply matrix by matrix
 *
 * A: RxM
 * B: MxK
 * C: RxK
 *
 * Scratch: M
 */
kernel void mul_mat_scratch_{T}(global const {T} *a, global const {T} *b, global {T} *c, uint m, local {T} *scratch) {
    int index_r = get_global_id(1);
    int index_k = get_global_id(0);

    int r = get_global_size(1);
    int k = get_global_size(0);

    //int lid_r = get_local_id(0);
    int lid_k = get_local_id(0);//(1);
    int lz_k = get_local_size(0);

    {T} sum = 0;
    for(int m_base = 0; m_base < m; m_base += lz_k) {
        scratch[lid_k] = a[ix(index_r, m_base + lid_k, m)];
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = 0; i < lz_k && m_base + i < m; i++) {
            int index_m = m_base + i;
            sum += scratch[i] * b[ix(index_m, index_k, k)];
            //sum += a[ix(index_r, m_base + i, m)]/*scratch[i]*/ * b[ix(m_base + i, index_k, k)];
        }
    }
    c[ix(index_r, index_k, k)] = sum;
}

/**
 * Multiply matrix by transposed matrix
 *
 * A: RxM
 * B: KxM
 * C: RxK
 */
kernel void mul_trans_mat_{T}(global const {T} *a, global const {T} *b, global {T} *c, uint m) {
    int index_r = get_global_id(1);
    int index_k = get_global_id(0);

    int r = get_global_size(1);
    int k = get_global_size(0);

    {T} sum = 0;
    for(int index_m = 0; index_m < m; index_m++) {
        sum += a[ix(index_r, index_m, m)] * b[ix(index_k, index_m, m)];
    }
    c[ix(index_r, index_k, k)] = sum;
}
