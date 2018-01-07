#define i get_global_id(0)

kernel void add_vec_vec(global {T}* C, global {T}* A, global {T}* B) {
    C[i] = A[i] + B[i];
}

kernel void sub_vec_vec(global {T}* C, global {T}* A, global {T}* B) {
    C[i] = A[i] - B[i];
}

kernel void mul_vec_vec(global {T}* C, global {T}* A, global {T}* B) {
    C[i] = A[i] * B[i];
}

kernel void div_vec_vec(global {T}* C, global {T}* A, global {T}* B) {
    C[i] = A[i] / B[i];
}

//

kernel void add_assign_vec_vec(global {T}* C, global {T}* B) {
    C[i] += B[i];
}

kernel void sub_assign_vec_vec(global {T}* C, global {T}* B) {
    C[i] -= B[i];
}

kernel void mul_assign_vec_vec(global {T}* C, global {T}* B) {
    C[i] *= B[i];
}

kernel void div_assign_vec_vec(global {T}* C, global {T}* B) {
    C[i] /= B[i];
}

//

kernel void mul_vec_scl(global {T}* C, global {T}* A, {T} B) {
    C[i] = A[i] * B;
}

kernel void div_vec_scl(global {T}* C, global {T}* A, {T} B) {
    C[i] = A[i] / B;
}

//

kernel void mul_assign_vec_scl(global {T}* C, {T} B) {
    C[i] *= B;
}

kernel void div_assign_vec_scl(global {T}* C, {T} B) {
    C[i] /= B;
}

//

kernel void eq_vec(global uchar* C, global {T}* A, global {T}* B) {
    if (A[i] != B[i])
        *C = false;
}