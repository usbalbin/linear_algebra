
#[macro_use]mod kernel_helpers;
mod param;
use param::Param;

use std::ops::{Add, Sub, Mul};
use std::fmt;

const WG_SIZE: usize = 64;

lazy_static!{
    pub static ref PRO_QUE: ocl::ProQue = ocl::ProQue::builder()
        .src(kernel_helpers::generate_kernel_source())
        .build()
        .map(|q| {
            let device = q.device();
            println!("{:?}", device);
            println!("{}", device.vendor().unwrap());
            println!("{}",  device.name().unwrap());
            q
        })
        .expect("Failed to build kernel src");
}

pub struct Matrix<T: Param, const R: usize, const K: usize> {
    data: ocl::Buffer<T>
}

impl<T: Param, const R: usize, const K: usize> Matrix<T, {R}, {K}> {
    pub fn new(data: &[T]) -> Self {
        assert_eq!(data.len(), R * K);

        let data = ocl::Buffer::builder()
            .queue(PRO_QUE.queue().clone())
            .len(data.len())
            .copy_host_slice(data)
            .build()
            .expect("Failed to setup buffer a");
        Matrix { data }
    }

    pub fn scrambled_vec(min: T, max: T) -> Self
        where T: ocl::OclScl + rand::distributions::range::SampleRange
    {
        Matrix::new(&ocl_extras::scrambled_vec((min, max), R * K))
    }

    ref_binop_func!{
        pub fn ref_elem_mul(lhs: &Self, rhs: &Self) -> res: Self {
            elem_mul_mat()
        }
    }

    owned_binop_func!{
        pub fn elem_mul(lhs: Self, rhs: &Self) -> res: Self {
            elem_mul_assign_mat()
        }
    }

    ref_binop_func!{
        pub fn ref_elem_div(lhs: &Self, rhs: &Self) -> res: Self {
            elem_div_mat()
        }
    }

    owned_binop_func!{
        pub fn elem_div(lhs: Self, rhs: &Self) -> res: Self {
            elem_div_assign_mat()
        }
    }
}

impl<T: Param + fmt::Display + Default, const R: usize, const K: usize> fmt::Display for Matrix<T, {R}, {K}> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut data = vec![Default::default(); R * K];
        self.data.read(&mut data[..]).enq().expect("Failed to read buffer");

        for index_r in 0..R {
            for index_k in 0..(K-1) {
                write!(f, "{}, ", data[index_r * K + index_k])?;
            }
            writeln!(f, "{}", data[index_r * K + (K - 1)])?;
        }
        Ok(())
    }
}


impl<T, const R: usize, const M: usize, const K: usize> Mul<&Matrix<T, {M}, {K}>> for &Matrix<T, {R}, {M}>
    where T: Param + Mul<T, Output=T> + Copy
{
    type Output = Matrix<T, {R}, {K}>;

    ref_binop_func!{
        fn mul(lhs: Self, rhs: &Matrix<T, {M}, {K}>) -> res: Self::Output {
            mul_mat(m = M as u32)
        }
    }
}

impl_ref_elem_wise_mat_op!(impl Add, fn add, add_mat());
impl_owned_elem_wise_mat_op!(impl Add, fn add, add_assign_mat());
impl_ref_elem_wise_mat_op!(impl Sub, fn sub, sub_mat());
impl_owned_elem_wise_mat_op!(impl Sub, fn sub, sub_assign_mat());


generate_kernel_getters! {
    fn mul_mat(
        lhs = None::<&ocl::Buffer<T>>,
        rhs = None::<&ocl::Buffer<T>>,
        res = None::<&ocl::Buffer<T>>,
        m = 0
    );

    fn add_mat(
        lhs = None::<&ocl::Buffer<T>>,
        rhs = None::<&ocl::Buffer<T>>,
        res = None::<&ocl::Buffer<T>>
    );
    fn add_assign_mat(
        lhs = None::<&ocl::Buffer<T>>,
        rhs = None::<&ocl::Buffer<T>>
    );

    fn sub_mat(
        lhs = None::<&ocl::Buffer<T>>,
        rhs = None::<&ocl::Buffer<T>>,
        res = None::<&ocl::Buffer<T>>
    );
    fn sub_assign_mat(
        lhs = None::<&ocl::Buffer<T>>,
        rhs = None::<&ocl::Buffer<T>>
    );

    fn elem_mul_mat(
        lhs = None::<&ocl::Buffer<T>>,
        rhs = None::<&ocl::Buffer<T>>,
        res = None::<&ocl::Buffer<T>>
    );

    fn elem_mul_assign_mat(
        lhs = None::<&ocl::Buffer<T>>,
        rhs = None::<&ocl::Buffer<T>>
    );

    fn elem_div_mat(
        lhs = None::<&ocl::Buffer<T>>,
        rhs = None::<&ocl::Buffer<T>>,
        res = None::<&ocl::Buffer<T>>
    );

    fn elem_div_assign_mat(
        lhs = None::<&ocl::Buffer<T>>,
        rhs = None::<&ocl::Buffer<T>>
    );
}
