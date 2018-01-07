#![feature(iterator_step_by)]

extern crate ocl;

use ocl::ProQue;
use ocl::Buffer;

pub mod vector;
pub mod traits;

use traits::Parameter;
//pub mod matrix;

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
}


fn cl_data<T: Parameter>() -> &'static mut Option<OclData> {
    static mut OCL_DATA: Option<OclData> = None;
    static INIT_ONCE: std::sync::Once = std::sync::ONCE_INIT;

    unsafe { //TODO: remove me to prevent bugs when multithreading
        INIT_ONCE.call_once(|| {
            init_helper::<T>(&mut OCL_DATA);
        });
        &mut OCL_DATA
    }
}

unsafe fn init_helper<T: Parameter>(ocl_data: &mut Option<OclData>) {
    let src = if T::type_to_str() == "double" {
        "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    } else {
        ""
    }.to_owned() +
        &include_str!("kernels.cl").replace("{T}", T::type_to_str());

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
    });
}



















/*
#[test]
pub fn vec_mat_mul() {
    use vector::*;
    use matrix::*;

    let a = Vector::from_vec(vec![1, 2, 3, 4]);
    let b = Vector::from_vec(vec![5, 4, 9, 4]);
    let m = Matrix::from_vec(vec![
        1, 0, 0,
        0, 2, 0,
        0, 0, 3,
        1, 0, 0
    ], 4, 3);

    let p = &a * &m;
    println!("{}", p);
    for (p, b) in p.iter().zip(b.iter()) {
        assert_eq!(p, b);
    }
}
*/

#[test]
pub fn vec_add_and_scalar_mul() {
    use vector::*;

    let a = Vector::from_vec(vec![1, 2, 3, 4]);
    let r = Vector::from_vec(vec![2, 4, 6, 8]);

    let s = &a + &a;
    let p = &a * 2;

    assert_eq!(r, s);
    assert_eq!(s, p);
}