
//---------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------- Utils -----------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------




//---------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------- Vector ----------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------

#define i get_global_id(0)

kernel void {T}_add_vec_vec(global {T}* C, global {T}* A, global {T}* B) {
    C[i] = A[i] + B[i];
}

kernel void {T}_sub_vec_vec(global {T}* C, global {T}* A, global {T}* B) {
    C[i] = A[i] - B[i];
}

kernel void {T}_mul_vec_vec(global {T}* C, global {T}* A, global {T}* B) {
    C[i] = A[i] * B[i];
}

kernel void {T}_div_vec_vec(global {T}* C, global {T}* A, global {T}* B) {
    C[i] = A[i] / B[i];
}

//

kernel void {T}_add_assign_vec_vec(global {T}* C, global {T}* B) {
    C[i] += B[i];
}

kernel void {T}_sub_assign_vec_vec(global {T}* C, global {T}* B) {
    C[i] -= B[i];
}

kernel void {T}_mul_assign_vec_vec(global {T}* C, global {T}* B) {
    C[i] *= B[i];
}

kernel void {T}_div_assign_vec_vec(global {T}* C, global {T}* B) {
    C[i] /= B[i];
}

//

kernel void {T}_mul_vec_scl(global {T}* C, global {T}* A, {T} B) {
    C[i] = A[i] * B;
}

kernel void {T}_div_vec_scl(global {T}* C, global {T}* A, {T} B) {
    C[i] = A[i] / B;
}

//

kernel void {T}_mul_assign_vec_scl(global {T}* C, {T} B) {
    C[i] *= B;
}

kernel void {T}_div_assign_vec_scl(global {T}* C, {T} B) {
    C[i] /= B;
}

//

kernel void {T}_eq_vec(global uchar* C, global {T}* A, global {T}* B) {
    if (A[i] != B[i])
        *C = false;
}

//---------------------------------------------------------------------------------------------------------------------
//------------------------------------------------- Matrix vec --------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------

inline {T} {T}_dot_product(global {T}* a, int stride_a, global {T}* b, int stride_b, int count) {
    {T} res = 0;
    for(int j = 0; j < count; j++) {
        res += a[j * stride_a] * b[j * stride_b];
    }
    return res;
}


#define col get_global_id(0)
kernel void {T}_mul_vec_mat(global {T}* C, global {T}* A, global {T}* B, int B_col_count, int A_len) {
    C[col] = {T}_dot_product(A, 1, &B[col], B_col_count, A_len);
}
#undef col


kernel void {T}_mul_vec_transpose_mat(global {T}* C, global {T}* A, global {T}* B, int B_col_count, int A_len) {
    C[i] = {T}_dot_product(A, 1, &B[i * B_col_count], 1, A_len);
}


//---------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------- Matrix ----------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------

//TODO make 1D or 2D kernel out of mul_mat_mat without for-loop

kernel void {T}_mul_mat_mat(global {T}* C, global {T}* A, global {T}* B, int C_col_count, int A_col_count) {
    //C.row_count = A.row_count
    //C.col_count = B.col_count
    //A.col_count = B.row_count
    int row = i / C_col_count;
    int col = i % C_col_count;

    C[i] = {T}_dot_product(&A[A_col_count * row], 1, &B[col], C_col_count, A_col_count);
}


//Iterates over one row per work item
#define row get_global_id(0)
kernel void {T}_mul_mat_mat_row(global {T}* C, global {T}* A, global {T}* B, int C_col_count, int A_col_count) {
    //C.row_count = A.row_count
    //C.col_count = B.col_count
    //A.col_count = B.row_count
    for (int col = 0; col < C_col_count; col++) {
        C[row * C_col_count + col] = {T}_dot_product(&A[A_col_count * row], 1, &B[col], C_col_count, A_col_count);
    }
}
#undef row

//Iterates over one col per work item
#define col get_global_id(0)
kernel void {T}_mul_mat_mat_col(global {T}* C, global {T}* A, global {T}* B, int C_row_count, int C_col_count, int A_col_count) {
    //C.row_count = A.row_count
    //C.col_count = B.col_count
    //A.col_count = B.row_count
    for (int row = 0; row < C_row_count; row++) {
        C[row * C_col_count + col] = {T}_dot_product(&A[A_col_count * row], 1, &B[col], C_col_count, A_col_count);
    }
}
#undef col