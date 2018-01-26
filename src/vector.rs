
extern crate ocl;

use ocl::{Buffer, MemFlags, Kernel};

use matrix::Matrix;

use ::std::ops::{Add, Mul};

use traits::Parameter;
use util::*;

use KernelsGuard;

use cl_data;
use get_kernels;
use GROUP_COUNT;
use GLOBAL_WORK_SIZE;
use LOCAL_WORK_SIZE;

pub struct Vector<T>
    where T: Parameter
{
    pub(crate) data: Buffer<T>
}

impl<T: Parameter> Vector<T> {
    pub fn new(default_val: T, size: usize) -> Vector<T>
        where T: Clone
    {
        Vector::from_vec(vec![default_val; size])
    }

    pub unsafe fn uninitialized_lock_free(size: usize, queue: ocl::Queue) -> Vector<T> {
        let buff = Buffer::builder()
            .queue(queue)
            .flags(MemFlags::new().read_write())
            .dims(size)
            .build().unwrap();

        Vector {
            data: buff
        }
    }

    pub unsafe fn uninitialized(size: usize) -> Vector<T> {
        let queue = get_kernels::<T>(T::type_to_str()).queue.clone();
        Vector::<T>::uninitialized_lock_free(size, queue)
    }

    pub fn from_vec(v: Vec<T>) -> Vector<T> {
        let queue = cl_data::<T>().queue.queue().clone();
        Vector::<T>::from_vec_lock_free(v, queue)
    }

    pub(crate) fn from_vec_lock_free(v: Vec<T>, queue: ocl::Queue) -> Vector<T> {
        let buff = Buffer::builder()
            .queue(queue)
            .flags(MemFlags::new().read_write().copy_host_ptr())
            .dims(v.len())
            .host_data(&v)
            .build().unwrap();

        Vector {
            data: buff
        }
    }

    pub fn to_vec(&self) -> Vec<T> {
        let mut res = Vec::with_capacity(self.len());
        unsafe {
            res.set_len(self.len())     // TODO: remove this ugliness
        }
        self.data.read(&mut res).enq().unwrap();
        res
    }

    pub fn generate(kernel: &mut Kernel, size: usize) -> Vector<T> {
        let mut res = unsafe { Vector::uninitialized(size) };

        kernel.set_arg_buf_named("C", Some(&mut res.data)).unwrap();
        unsafe { kernel.cmd().gws(res.len()).enq().unwrap(); }
        res
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn from_for_each2_lock_free(a: &Vector<T>, b: &Vector<T>, kernel: &mut Kernel, queue: ocl::Queue) -> Vector<T> {
        assert_eq!(a.len(), b.len());

        let mut res = unsafe{ Vector::uninitialized_lock_free(a.len(), queue) };

        kernel.set_arg_buf_named("C", Some(&mut res.data)).unwrap();
        kernel.set_arg_buf_named("A", Some(&a.data)).unwrap();
        kernel.set_arg_buf_named("B", Some(&b.data)).unwrap();

        unsafe { kernel.cmd().gws(res.len()).enq().unwrap(); }

        res
    }

    pub fn for_each_mut(&mut self, other: &Vector<T>, kernel: &mut Kernel){
        assert_eq!(self.len(), other.len());

        kernel.set_arg_buf_named("C", Some(&mut self.data)).unwrap();
        kernel.set_arg_buf_named("B", Some(&other.data)).unwrap();

        unsafe { kernel.cmd().gws(self.len()).enq().unwrap(); }
    }

    /// Applies p for every element in vector
    pub fn map_mut<F: FnMut(&mut T)>(&mut self, kernel: &mut Kernel) {
        kernel.set_arg_buf_named("C", Some(&mut self.data)).unwrap();

        unsafe { kernel.cmd().gws(self.len()).enq().unwrap(); }
    }

    /// Returns copy of self with p applied to each element
    pub fn map_lock_free<F: FnMut(&T)->T >(&self, kernel: &mut Kernel, queue: ocl::Queue) -> Vector<T> {
        let mut res = unsafe{ Vector::uninitialized_lock_free(self.len(), queue) };

        kernel.set_arg_buf_named("C", Some(&mut res.data)).unwrap();
        kernel.set_arg_buf_named("B", Some(&self.data)).unwrap();

        unsafe { kernel.cmd().gws(res.len()).enq().unwrap(); }
        res
    }

    pub unsafe fn get_buffer(&self) -> &ocl::Buffer<T> {
        &self.data
    }

    pub unsafe fn get_buffer_mut(&mut self) -> &mut ocl::Buffer<T> {
        &mut self.data
    }

    pub(crate) unsafe fn read_file_only_data(file: &mut ::std::fs::File, elem_count: u64) -> Result<Vector<T>, ::std::io::Error> {
        use std::io::Read;

        let mut data = Vec::with_capacity(elem_count as usize);
        let mut raw = Vec::with_capacity(::std::mem::size_of::<T>());

        let elem_size = ::std::mem::size_of::<T>();


        for _ in 0..(elem_count) {
            for _ in 0..elem_size { //TODO: Check if this needs to be reversed
                let mut byte = [0u8];
                file.read_exact(&mut byte)?;
                raw.push(byte);
            }
            let elem = *(raw[..].as_ptr() as *const T);
            data.push(elem);
            raw.clear();
        }
        Ok(Vector::from_vec(data))
    }

    pub(crate) fn write_file_only_data(&self, file: &mut ::std::fs::File) -> Result<(), ::std::io::Error> {
        use std::slice;
        use std::io::Write;

        let data = self.to_vec();
        unsafe {
            let data: &[u8] = slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * ::std::mem::size_of::<T>(),
            );
            file.write_all(data)
        }
    }

    /// Append Vector data to file.
    /// NOTE! Vector will be encoded in the current systems endianness
    ///
    /// Two u64 will be placed first, representing the number of bytes per element and the
    /// number of elements respectively.
    pub fn write_to_file(&self, file: &mut ::std::fs::File) -> ::std::result::Result<(), ::std::io::Error> {
        write_u64(file, ::std::mem::size_of::<T>() as u64)?;     //Store element size in bytes
        write_u64(file, self.len() as u64)?;                     //Store element count

        self.write_file_only_data(file)                             //Store data
    }

    /// Read data to from file.
    /// NOTE! Buffer will be interpreted in the current systems endianness
    pub unsafe fn read_from_file(file: &mut ::std::fs::File) -> Result<Vector<T>, ::std::io::Error> {
        let elem_size = read_u64(file)?;


        if (elem_size as usize) != ::std::mem::size_of::<T>() {
            panic!(
                "Elem size from buffer does not seem to match, what was expected!\
                 Missmatch in endiannes?"
            );
        }

        let elem_count = read_u64(file)?;

        Self::read_file_only_data(file, elem_count)
    }

    /// Save Vector to specified path.
    /// NOTE! The file will be encoded in the current systems endianness
    pub fn save(&self, path: &str) -> Result<(), ::std::io::Error> {
        use std::fs::File;

        let mut file = File::create(path)?;

        self.write_to_file(&mut file)
    }

    /// Open Vector from specified path.
    /// NOTE! The file will be interpreted in the current systems endianness
    pub unsafe fn open(path: &str) -> Result<Vector<T>, ::std::io::Error> {
        use std::fs::File;

        let mut file = File::open(path).expect("Failed to open file");

        Ok(Vector::<T>::read_from_file(&mut file).expect("Failed to read all data from file"))
    }
}

impl<T: Parameter + ::std::iter::Sum<T>> Vector<T> {
    pub fn sum(&self) -> T {
        let mut kernels = get_kernels::<T>(T::type_to_str());
        let queue = kernels.queue.clone();
        let kernel = &mut kernels.sum_vec;

        unsafe {
            let mut tmp = Vector::uninitialized_lock_free(GROUP_COUNT, queue);

            kernel.set_arg_buf_named("data", Some(&self.data)).unwrap();
            kernel.set_arg_buf_named("results", Some(&mut tmp.data)).unwrap();
            kernel.set_arg_scl_named("count", self.len() as i32).unwrap();

            kernel.cmd().gws(GLOBAL_WORK_SIZE).lws(LOCAL_WORK_SIZE).enq().unwrap();
            tmp.to_vec().into_iter().sum()
        }
    }
}


pub fn dot<T: Parameter + Mul + ::std::iter::Sum<T>>(a: &Vector<T>, b: &Vector<T>) -> T {
    assert_eq!(a.len(), b.len());

    let mut kernels = get_kernels::<T>(T::type_to_str());
    let queue = kernels.queue.clone();
    let kernel = &mut kernels.dot_vec_vec;

    unsafe {
        let mut tmp = Vector::uninitialized_lock_free(GROUP_COUNT, queue);

        kernel.set_arg_buf_named("a", Some(&a.data)).unwrap();
        kernel.set_arg_buf_named("b", Some(&b.data)).unwrap();
        kernel.set_arg_buf_named("results", Some(&mut tmp.data)).unwrap();
        kernel.set_arg_scl_named("count", a.len() as i32).unwrap();

        kernel.cmd().gws(GLOBAL_WORK_SIZE).lws(LOCAL_WORK_SIZE).enq().unwrap();
        tmp.to_vec().into_iter().sum()
    }
}


impl<'a, 'b, T> ::std::ops::Add<&'b Vector<T>> for &'a Vector<T>
    where T: Copy + ::std::ops::Add<T, Output=T> + Parameter
{
    type Output = Vector<T>;
    fn add(self, other: &'b Vector<T>) -> Vector<T> {
        let mut kernels = get_kernels::<T>(T::type_to_str());
        let queue = kernels.queue.clone();
        Vector::from_for_each2_lock_free(
        self, other,
        &mut kernels.add_vec_vec,
            queue
        )
    }
}

impl<'a, T> ::std::ops::AddAssign<&'a Vector<T>> for Vector<T>
    where T: Copy + ::std::ops::AddAssign<T> + Parameter
{
    fn add_assign(&mut self, other: &'a Vector<T>) {
        let mut kernels = get_kernels::<T>(T::type_to_str());
        Vector::for_each_mut(
        self, other,
        &mut kernels.add_assign_vec_vec
        );
    }
}

impl<'a, T> ::std::ops::Add<&'a Vector<T>> for Vector<T>
    where T: Copy + ::std::ops::AddAssign<T> + Parameter
{
    type Output = Vector<T>;
    fn add(mut self, other: &'a Vector<T>) -> Vector<T> {
        self += other;
        self
    }
}

impl<'a, 'b, T> ::std::ops::Sub<&'b Vector<T>> for &'a Vector<T>
    where T: Copy + ::std::ops::Sub<T, Output=T> + Parameter
{
    type Output = Vector<T>;
    fn sub(self, other: &'b Vector<T>) -> Vector<T> {
        let mut kernels = get_kernels::<T>(T::type_to_str());
        let queue = kernels.queue.clone();
        Vector::from_for_each2_lock_free(
            self, other,
            &mut kernels.sub_vec_vec,
            queue
        )
    }
}

impl<'a, T> ::std::ops::SubAssign<&'a Vector<T>> for Vector<T>
    where T: Copy + ::std::ops::SubAssign<T> + Parameter
{
    fn sub_assign(&mut self, other: &'a Vector<T>) {
        let mut kernels = get_kernels::<T>(T::type_to_str());
        Vector::for_each_mut(
            self, other,
            &mut kernels.sub_assign_vec_vec);
    }
}

//Mul by scalar
impl<'a, 'b, T> ::std::ops::Mul<T> for &'a Vector<T>
    where T: Copy + ::std::ops::Mul<T, Output=T> + Parameter
{
    type Output = Vector<T>;
    fn mul(self, scalar: T) -> Vector<T> {
        let mut kernels = get_kernels::<T>(T::type_to_str());

        let queue = kernels.queue.clone();
        let kernel = &mut kernels.mul_vec_scl;

        let mut res = unsafe{ Vector::uninitialized_lock_free(
            self.len(),
            queue
        )};

        kernel.set_arg_buf_named("C", Some(&mut res.data)).unwrap();
        kernel.set_arg_buf_named("A", Some(&self.data)).unwrap();
        kernel.set_arg_scl_named("B", scalar).unwrap();

        unsafe { kernel.cmd().gws(res.len()).enq().unwrap(); }

        res
    }
}

impl<T> ::std::ops::MulAssign<T> for Vector<T>
    where T: Copy + ::std::ops::MulAssign<T> + Parameter
{
    fn mul_assign(&mut self, scalar: T) {
        let mut kernels = get_kernels::<T>(T::type_to_str());
        let kernel = &mut kernels.mul_assign_vec_scl;

        kernel.set_arg_buf_named("C", Some(&mut self.data)).unwrap();
        kernel.set_arg_scl_named("B", scalar).unwrap();

        unsafe { kernel.cmd().gws(self.len()).enq().unwrap(); }
    }
}

impl<T> ::std::ops::Mul<T> for Vector<T>
    where T: Copy + ::std::ops::MulAssign<T> + Parameter
{
    type Output = Vector<T>;
    fn mul(mut self, scalar: T) -> Vector<T> {
        self *= scalar;
        self
    }
}

impl<'a, 'b, T> ::std::ops::Mul<&'b Vector<T>> for &'a Vector<T>
    where T: Copy + ::std::ops::Mul<T, Output=T> + Parameter
{
    type Output = Vector<T>;
    fn mul(self, other: &'b Vector<T>) -> Vector<T> {
        let mut kernels = get_kernels::<T>(T::type_to_str());
        let queue = kernels.queue.clone();
        Vector::from_for_each2_lock_free(
            self,
            other,
            &mut kernels.mul_vec_vec,
            queue
        )
    }
}

impl<'a, T> ::std::ops::MulAssign<&'a Vector<T>> for Vector<T>
    where T: Copy + ::std::ops::MulAssign<T> + Parameter
{
    fn mul_assign(&mut self, other: &'a Vector<T>) {
        let mut kernels = get_kernels::<T>(T::type_to_str());
        Vector::for_each_mut(self, other, &mut kernels.mul_assign_vec_vec);
    }
}

//Vector * Matrix
impl<'a, 'b, T> ::std::ops::Mul<&'b Matrix<T>> for &'a Vector<T>
    where T: Copy + Mul<T, Output=T> + Add + Parameter
{
    type Output = Vector<T>;
    fn mul(self, other_m: &'b Matrix<T>) -> Vector<T> {//TODO: check me
        assert_eq!(self.len(), other_m.get_row_count());

        let mut kernels = get_kernels::<T>(T::type_to_str());
        let mut res = unsafe{ Vector::uninitialized_lock_free(
            other_m.get_col_count(),
            kernels.queue.clone())
        };

        let kernel = &mut kernels.mul_vec_mat;

        kernel.set_arg_buf_named("C", Some(&mut res.data)).unwrap();
        kernel.set_arg_buf_named("A", Some(&self.data)).unwrap();
        kernel.set_arg_buf_named("B", Some(&other_m.data.data)).unwrap();

        kernel.set_arg_scl_named::<i32>("B_col_count", other_m.get_col_count() as i32).unwrap();
        kernel.set_arg_scl_named::<i32>("A_len", self.len() as i32).unwrap();

        unsafe { kernel.cmd().gws(res.len()).enq().unwrap(); }

        res
    }
}

/// Compute vector * transpose(other_m), where transpose(other_m) is the transpose of matrix other_m
pub fn mul_transpose_mat<T>(vector: &Vector<T>, other_m: &Matrix<T>) -> Vector<T>
    where T: Copy + Mul<T, Output=T> + Add + Parameter
{
    assert_eq!(vector.len(), other_m.get_col_count());

    let mut kernels = get_kernels::<T>(T::type_to_str());
    let mut res = unsafe { Vector::uninitialized_lock_free(
        other_m.get_row_count(),
        kernels.queue.clone()
    )};


    let kernel = &mut kernels.mul_vec_transpose_mat;

    kernel.set_arg_buf_named("C", Some(&mut res.data)).unwrap();
    kernel.set_arg_buf_named("A", Some(&vector.data)).unwrap();
    kernel.set_arg_buf_named("B", Some(&other_m.data.data)).unwrap();

    kernel.set_arg_scl_named::<i32>("B_col_count", other_m.get_col_count() as i32).unwrap();
    kernel.set_arg_scl_named::<i32>("A_len", vector.len() as i32).unwrap();

    unsafe { kernel.cmd().gws(res.len()).enq().unwrap(); }
    res
}

impl<'a, 'b, T> ::std::ops::Div<&'b Vector<T>> for &'a Vector<T>
    where T: Copy + ::std::ops::Div<T, Output=T> + Parameter
{
    type Output = Vector<T>;
    fn div(self, other: &'b Vector<T>) -> Vector<T> {
        let mut kernels = get_kernels::<T>(T::type_to_str());
        let queue = kernels.queue.clone();

        Vector::from_for_each2_lock_free(
            self,
            other,
            &mut kernels.div_vec_vec,
            queue
        )
    }
}

impl<'a, T> ::std::ops::DivAssign<&'a Vector<T>> for Vector<T>
    where T: Copy + ::std::ops::DivAssign<T> + Parameter
{
    fn div_assign(&mut self, other: &'a Vector<T>) {
        let mut kernels = get_kernels::<T>(T::type_to_str());
        Vector::for_each_mut(self, other, &mut kernels.div_assign_vec_vec);
    }
}

//Div by scalar
impl<'a, 'b, T> ::std::ops::Div<T> for &'a Vector<T>
    where T: Copy + ::std::ops::Div<T, Output=T> + Parameter
{
    type Output = Vector<T>;
    fn div(self, scalar: T) -> Vector<T> {

        let mut kernels = get_kernels::<T>(T::type_to_str());
        let queue = kernels.queue.clone();
        let kernel = &mut kernels.div_vec_scl;

        let mut res = unsafe{ Vector::uninitialized_lock_free(
            self.len(),
            queue
        )};

        kernel.set_arg_buf_named("C", Some(&mut res.data)).unwrap();
        kernel.set_arg_buf_named("A", Some(&self.data)).unwrap();
        kernel.set_arg_scl_named("B", scalar).unwrap();

        unsafe { kernel.cmd().gws(res.len()).enq().unwrap(); }

        res
    }
}

impl<T> ::std::ops::DivAssign<T> for Vector<T>
    where T: Copy + ::std::ops::DivAssign<T> + Parameter
{
    fn div_assign(&mut self, scalar: T) {
        let mut kernels = get_kernels::<T>(T::type_to_str());
        let kernel = &mut kernels.div_assign_vec_scl;

        kernel.set_arg_buf_named("C", Some(&mut self.data)).unwrap();
        kernel.set_arg_scl_named("B", scalar).unwrap();

        unsafe { kernel.cmd().gws(self.len()).enq().unwrap(); }
    }
}

impl<'a, 'b, T> ::std::cmp::PartialEq for Vector<T>
    where T: Copy + ::std::cmp::PartialEq + Parameter
{
    fn eq(&self, other: &Vector<T>) -> bool {
        if self.len() != other.len() {
            return false;
        }

        let cl_data = cl_data::<T>();
        let queue = cl_data.queue.queue().clone();
        let mut kernels = KernelsGuard(cl_data, T::type_to_str());
        let kernel = &mut kernels.eq_vec;

        let mut is_equal = Vector::from_vec_lock_free(
            vec![1u8],
            queue,
        );

        kernel.set_arg_buf_named("C", Some(&mut is_equal.data)).unwrap();
        kernel.set_arg_buf_named("A", Some(&self.data)).unwrap();
        kernel.set_arg_buf_named("B", Some(&other.data)).unwrap();

        unsafe { kernel.cmd().gws(self.len()).enq().unwrap(); }

        is_equal.to_vec()[0] != 0
    }
}


impl<T: Parameter + ::std::fmt::Display + Clone> ::std::fmt::Display for Vector<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result{
        let v = self.to_vec();
        write!(f, "{{ {}", v[0])?;
        for i in 1..self.len() {
            write!(f, ", {}", v[i])?;
        }
        write!(f, " }}")
    }
}

impl<T: Parameter + ::std::fmt::Debug + Clone> ::std::fmt::Debug for Vector<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result{
        let v = self.to_vec();
        write!(f, "{{ {:?}", v[0])?;
        for i in 1..self.len() {
            write!(f, ", {:?}", v[i])?;
        }
        write!(f, " }}")
    }
}