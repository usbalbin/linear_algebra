#![feature(iterator_step_by)]

extern crate ocl;

use ocl::ProQue;
use ocl::Buffer;

pub mod vector;
pub mod traits;

use traits::Parameter;
pub mod matrix;

struct OclData {
    queue: ocl::ProQue,

    add_vec_vec: ocl::Kernel,
    sub_vec_vec: ocl::Kernel,
    mul_vec_vec: ocl::Kernel,
    div_vec_vec: ocl::Kernel,

    add_assign_vec_vec: ocl::Kernel,
    sub_assign_vec_vec: ocl::Kernel,
    mul_assign_vec_vec: ocl::Kernel,
    div_assign_vec_vec: ocl::Kernel,

    mul_vec_scl: ocl::Kernel,
    div_vec_scl: ocl::Kernel,

    mul_assign_vec_scl: ocl::Kernel,
    div_assign_vec_scl: ocl::Kernel,

    eq_vec: ocl::Kernel,

    //Matrix vec

    mul_vec_mat: ocl::Kernel,
    mul_vec_transpose_mat: ocl::Kernel,

    //Matrix

    mul_mat_mat: ocl::Kernel,

}


fn cl_data<T: Parameter>() -> &'static mut Option<OclData> {
    static mut OCL_DATA: Option<OclData> = None;
    static INIT_ONCE: std::sync::Once = std::sync::ONCE_INIT;

    unsafe { //TODO: remove me to prevent bugs when multithreading
        INIT_ONCE.call_once(|| {
            init_cl::<T>(&mut OCL_DATA);
        });
        &mut OCL_DATA
    }
}

/// Create Kernel object from kernel source in extra_kernels.cl
///
/// Note:
/// In kernel source, use {T} as type when referring to corresponding rust equivalent T.
/// Variable name 'i' is defined as get_global_id(0)
pub fn create_kernel<T: Parameter>(kernel_name: &str) -> ocl::Kernel {
    let queue = &mut cl_data::<T>().as_mut().unwrap().queue;
    match queue.create_kernel(kernel_name) {
        Ok(kernel) => kernel,
        Err(error) => panic!("Failed to create kernel. Forgot to add kernel source to kernel.cl?: {}", error)
    }
}

fn load_extra_src() -> Result<String, std::io::Error> {
    use std::fs::File;
    use std::io::prelude::*;
    let mut file = File::open("src/extra_kernels.cl")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

unsafe fn init_cl<T: Parameter>(ocl_data: &mut Option<OclData>) {
    let extra_src = load_extra_src().unwrap_or_default();
    let src = if T::type_to_str() == "double" {
        "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    } else {
        ""
    }.to_owned();
    let src = src + include_str!("kernels.cl")  + "\n" + &extra_src;
    let src = src.replace("{T}", T::type_to_str());

    let queue = match ProQue::builder()
        .device(ocl::flags::DEVICE_TYPE_GPU)
        .src(src)
        .build()
        {
            Err(e) => panic!("Failed to compile kernels with error: {}", e),
            Ok(q) => q,
        };


    let add_vec_vec = queue.create_kernel("add_vec_vec").unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);
    let sub_vec_vec = queue.create_kernel("sub_vec_vec").unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);
    let mul_vec_vec = queue.create_kernel("mul_vec_vec").unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);
    let div_vec_vec = queue.create_kernel("div_vec_vec").unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);

    let add_assign_vec_vec = queue.create_kernel("add_assign_vec_vec").unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);
    let sub_assign_vec_vec = queue.create_kernel("sub_assign_vec_vec").unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);
    let mul_assign_vec_vec = queue.create_kernel("mul_assign_vec_vec").unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);
    let div_assign_vec_vec = queue.create_kernel("div_assign_vec_vec").unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);

    let mul_vec_scl = queue.create_kernel("mul_vec_scl").unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_scl_named::<T>("B", None);
    let div_vec_scl = queue.create_kernel("div_vec_scl").unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_scl_named::<T>("B", None);

    let mul_assign_vec_scl = queue.create_kernel("mul_assign_vec_scl").unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_scl_named::<T>("B", None);
    let div_assign_vec_scl = queue.create_kernel("div_assign_vec_scl").unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_scl_named::<T>("B", None);

    let eq_vec = queue.create_kernel("eq_vec").unwrap()
        .arg_buf_named::<u8, Buffer<u8>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);

    //Matrix vec

    let mul_vec_mat = queue.create_kernel("mul_vec_mat").unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None)
        .arg_scl_named::<i32>("B_col_count", None)
        .arg_scl_named::<i32>("A_len", None);

    let mul_vec_transpose_mat = queue.create_kernel("mul_vec_transpose_mat").unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None)
        .arg_scl_named::<i32>("B_col_count", None)
        .arg_scl_named::<i32>("A_len", None);


    //Matrix

    let mul_mat_mat = queue.create_kernel("mul_mat_mat").unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None)
        .arg_scl_named::<i32>("C_col_count", None)
        .arg_scl_named::<i32>("A_col_count", None);

    *ocl_data = Some(OclData{
        queue,

        add_vec_vec,
        sub_vec_vec,
        mul_vec_vec,
        div_vec_vec,

        add_assign_vec_vec,
        sub_assign_vec_vec,
        mul_assign_vec_vec,
        div_assign_vec_vec,

        mul_vec_scl,
        div_vec_scl,

        mul_assign_vec_scl,
        div_assign_vec_scl,

        eq_vec,

        //Matrix vec

        mul_vec_mat,
        mul_vec_transpose_mat,

        //Matrix

        mul_mat_mat,
    });
}


















#[test]
fn vec_eq() {
    use vector::*;

    let a = Vector::from_vec(vec![1, 2, 3, 4]);
    let b = Vector::from_vec(vec![1, 2, 3, 5]);
    let c = Vector::from_vec(vec![1, 2, 3]);
    let d = Vector::from_vec(vec![1, 2, 3, 4]);

    assert_eq!(a, a);
    assert_ne!(a, c);
    assert_ne!(a, b);
    assert_eq!(a, d);
}

#[test]
fn vec_mat_mul() {
    use vector::*;
    use matrix::*;

    let a = Vector::from_vec(vec![1, 2, 3, 4]);
    let b = Vector::from_vec(vec![5, 4, 9]);
    let m = Matrix::from_vec(vec![
        1, 0, 0,
        0, 2, 0,
        0, 0, 3,
        1, 0, 0
    ], 4, 3);

    let p = &a * &m;
    assert_eq!(p, b);
}

#[test]
fn mat_mat_mul() {
    use matrix::*;

    let a = Matrix::from_vec(vec![
        1, 2, 3,
        4, 5, 6,
    ], 2, 3);

    let b = Matrix::from_vec(vec![
        7, 8,
        9, 10,
        11, 12
    ], 3, 2);

    let c = Matrix::from_vec(vec![
        58, 64,
        139, 154
    ], 2, 2);

    assert_eq!(c, &a * &b);
}

#[test]
pub fn vec_transpose_mat_mul() {
    use vector::*;
    use matrix::*;

    let a = Vector::from_vec(vec![1, 2, 3, 4]);
    let b = Vector::from_vec(vec![5, 4, 9]);
    let m = Matrix::from_vec(vec![
        1, 0, 0, 1,
        0, 2, 0, 0,
        0, 0, 3, 0
    ], 3, 4);

    let p = mul_transpose_mat(&a, &m);
    assert_eq!(p, b);
}


#[test]
pub fn vec_add() {
    use vector::*;

    let a = Vector::from_vec(vec![1, 2, 3, 4]);
    let r = Vector::from_vec(vec![2, 4, 6, 8]);

    let s = &a + &a;

    assert_eq!(r, s);
}

#[test]
pub fn vec_mul_scl() {
    use vector::*;

    let a = Vector::from_vec(vec![1, 2, 3, 4]);
    let r = Vector::from_vec(vec![2, 4, 6, 8]);


    let p = &a * 2;

    assert_eq!(r, p);
}