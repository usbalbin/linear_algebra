
/// The content of this file will only be used when running tests for this
/// this crate. Users of this library should create ther own file with the
/// same name in their src/ folder where thay may define extra OpenCL
/// functions read doc for create_kernel() for more info

#if defined(IS_FLOAT) || defined(IS_DOUBLE)

{T} {T}_test_func({T} x) {
    return exp(x);
}

#endif



kernel void {T}_test_kernel(global {T}* a, global {T}* b) {
    a[i] = b[i] * ({T})(2);
}