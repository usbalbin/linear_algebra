
extern crate ocl;

use ocl::{Buffer, MemFlags, Kernel};

//use matrix::Matrix;

use ::std::ops::{Add, Mul};

use traits::Parameter;

use cl_data;
/*
pub fn dot_it_it<'a, T, ItA, ItB>(a: ItA, b: ItB) -> T
    where
        T: 'a +
            Copy +
            ::std::ops::Mul +
            ::std::iter::Sum<<T as ::std::ops::Mul>::Output> +
            Parameter,
        ItA: Iterator<Item = &'a T>,
        ItB: Iterator<Item = &'a T>
{
    //assert_eq!(a.count(), b.count());
    a.zip(b).map(|(x, y)| *x * *y).sum()
}

pub fn dot<'a, T, IT>(vector: &Vector<T>, it: IT) -> T
    where
        T: 'a +
            Parameter +
            Copy +
            Add +
            Mul +
            ::std::iter::Sum<<T as ::std::ops::Mul>::Output>,
        IT: Iterator<Item = &'a T>
{
    //assert_eq!(vector.len(), it.count());
    vector.iter().zip(it).map(|(v, r)| *v * *r).sum()
}
*/
pub struct Vector<T>
    where T: Parameter
{
    data: Buffer<T>
}

impl<T: Parameter> Vector<T> {
    pub fn new(default_val: T, size: usize) -> Vector<T>
        where T: Clone
    {
        Vector::from_vec(vec![default_val; size])
    }

    pub unsafe fn uninitialized(size: usize) -> Vector<T> {
        let buff = Buffer::builder()
            .queue(cl_data::<T>().as_mut().unwrap().queue.queue().clone())
            .flags(MemFlags::new().read_write())
            .dims(size)
            .build().unwrap();

        Vector {
            data: buff
        }
    }

    pub fn from_vec(v: Vec<T>) -> Vector<T> {
        let buff = Buffer::builder()
            .queue(cl_data::<T>().as_mut().unwrap().queue.queue().clone())
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

    pub fn from_for_each2(a: &Vector<T>, b: &Vector<T>, kernel: &mut Kernel) -> Vector<T> {
        assert_eq!(a.len(), b.len());

        let mut res = unsafe{ Vector::uninitialized(a.len()) };

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

    /*pub fn iter(&self) -> ::std::slice::Iter<T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> ::std::slice::IterMut<T> {
        self.data.iter_mut()
    }

    pub fn chunks(&self, size: usize) -> ::std::slice::Chunks<T> {
        self.data.chunks(size)
    }

    pub fn chunks_mut(&mut self, size: usize) -> ::std::slice::ChunksMut<T> {
        self.data.chunks_mut(size)
    }*/

    /// Applies p for every element in vector
    pub fn map_mut<F: FnMut(&mut T)>(&mut self, kernel: &mut Kernel) {
        kernel.set_arg_buf_named("C", Some(&mut self.data)).unwrap();

        unsafe { kernel.cmd().gws(self.len()).enq().unwrap(); }
    }

    /// Returns copy of self with p applied to each element
    pub fn map<F: FnMut(&T)->T >(&self, kernel: &mut Kernel) -> Vector<T> {
        let mut res = unsafe{ Vector::uninitialized(self.len()) };

        kernel.set_arg_buf_named("C", Some(&mut res.data)).unwrap();
        kernel.set_arg_buf_named("B", Some(&self.data)).unwrap();

        unsafe { kernel.cmd().gws(res.len()).enq().unwrap(); }
        res
    }
}

impl<'a, 'b, T> ::std::ops::Add<&'b Vector<T>> for &'a Vector<T>
    where T: Copy + ::std::ops::Add<T, Output=T> + Parameter
{
    type Output = Vector<T>;
    fn add(self, other: &'b Vector<T>) -> Vector<T> {
        Vector::from_for_each2(self, other, &mut cl_data::<T>().as_mut().unwrap().add_vec_vec)
    }
}

impl<'a, T> ::std::ops::AddAssign<&'a Vector<T>> for Vector<T>
    where T: Copy + ::std::ops::AddAssign<T> + Parameter
{
    fn add_assign(&mut self, other: &'a Vector<T>) {
        Vector::for_each_mut(self, other, &mut cl_data::<T>().as_mut().unwrap().add_assign_vec_vec);
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
        Vector::from_for_each2(self, other, &mut cl_data::<T>().as_mut().unwrap().sub_vec_vec)
    }
}

impl<'a, T> ::std::ops::SubAssign<&'a Vector<T>> for Vector<T>
    where T: Copy + ::std::ops::SubAssign<T> + Parameter
{
    fn sub_assign(&mut self, other: &'a Vector<T>) {
        Vector::for_each_mut(self, other, &mut cl_data::<T>().as_mut().unwrap().sub_assign_vec_vec);
    }
}

//Mul by scalar
impl<'a, 'b, T> ::std::ops::Mul<T> for &'a Vector<T>
    where T: Copy + ::std::ops::Mul<T, Output=T> + Parameter
{
    type Output = Vector<T>;
    fn mul(self, scalar: T) -> Vector<T> {
        let kernel = &mut cl_data::<T>().as_mut().unwrap().mul_vec_scl;

        let mut res = unsafe{ Vector::uninitialized(self.len()) };

        kernel.set_arg_buf_named("C", Some(&mut res.data)).unwrap();
        kernel.set_arg_buf_named("A", Some(&self.data)).unwrap();
        kernel.set_arg_scl_named("B", scalar).unwrap();

        unsafe { kernel.cmd().gws(res.len()).enq().unwrap(); }

        res
    }
}

impl<T> ::std::ops::MulAssign<T> for Vector<T>
    where T: Copy + ::std::ops::Mul<T, Output=T> + Parameter
{
    fn mul_assign(&mut self, scalar: T) {
        let kernel = &mut cl_data::<T>().as_mut().unwrap().mul_assign_vec_scl;

        kernel.set_arg_buf_named("C", Some(&mut self.data)).unwrap();
        kernel.set_arg_scl_named("B", scalar).unwrap();

        unsafe { kernel.cmd().gws(self.len()).enq().unwrap(); }
    }
}

impl<T> ::std::ops::Mul<T> for Vector<T>
    where T: Copy + ::std::ops::Mul<T, Output=T> + Parameter
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
        Vector::from_for_each2(self, other, &mut cl_data::<T>().as_mut().unwrap().mul_vec_vec)
    }
}

/*impl<'a, 'b, T> ::std::ops::Mul<&'b Matrix<T>> for &'a Vector<T>
    where T: Copy + Mul<T, Output=T> + Add + ::std::iter::Sum + Parameter
{
    type Output = Vector<T>;
    fn mul(self, other_m: &'b Matrix<T>) -> Vector<T> {//TODO: check me
        assert_eq!(self.len(), other_m.get_row_count());
        let result =
            (0..other_m.get_col_count())//For each column
                .map(|i| dot(self, other_m.get_col(i)));
        Vector {
            data: result.collect()
        }
    }
}

/// Compute vector * other_m^-1, where other_m^-1 is the transpose of matrix other_m
pub fn mul_transpose_mat<T>(vector: &Vector<T>, other_m: &Matrix<T>) -> Vector<T>
    where T: Copy + Mul<T, Output=T> + Add + ::std::iter::Sum + Parameter
{
    assert_eq!(vector.len(), other_m.get_col_count());
    let result =
        (0..other_m.get_row_count())//For each row
            .map(|i| dot(vector, other_m.get_row(i)));
    Vector{
        data: result.collect()
    }
}*/

impl<'a, 'b, T> ::std::ops::Div<&'b Vector<T>> for &'a Vector<T>
    where T: Copy + ::std::ops::Div<T, Output=T> + Parameter
{
    type Output = Vector<T>;
    fn div(self, other: &'b Vector<T>) -> Vector<T> {
        Vector::from_for_each2(self, other, &mut cl_data::<T>().as_mut().unwrap().div_vec_vec)
    }
}

//Div by scalar
impl<'a, 'b, T> ::std::ops::Div<T> for &'a Vector<T>
    where T: Copy + ::std::ops::Div<T, Output=T> + Parameter
{
    type Output = Vector<T>;
    fn div(self, scalar: T) -> Vector<T> {
        let kernel = &mut cl_data::<T>().as_mut().unwrap().div_vec_scl;

        let mut res = unsafe{ Vector::uninitialized(self.len()) };

        kernel.set_arg_buf_named("C", Some(&mut res.data)).unwrap();
        kernel.set_arg_buf_named("A", Some(&self.data)).unwrap();
        kernel.set_arg_scl_named("B", scalar).unwrap();

        unsafe { kernel.cmd().gws(res.len()).enq().unwrap(); }

        res
    }
}

impl<'a, 'b, T> ::std::cmp::PartialEq for Vector<T>
    where T: Copy + ::std::cmp::Eq + Parameter
{
    fn eq(&self, other: &Vector<T>) -> bool {
        let kernel = &mut cl_data::<T>().as_mut().unwrap().eq_vec;

        let mut is_equal = Vector::from_vec(vec![1u8]);

        kernel.set_arg_buf_named("C", Some(&mut is_equal.data)).unwrap();
        kernel.set_arg_buf_named("A", Some(&self.data)).unwrap();
        kernel.set_arg_buf_named("B", Some(&other.data)).unwrap();

        unsafe { kernel.cmd().gws(self.len()).enq().unwrap(); }

        is_equal.to_vec()[0] != 0
    }
}

/*impl<T: Parameter> ::std::ops::IndexMut<usize> for Vector<T> {
    fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut T {
        &mut self.data[index]
    }
}

impl<T: Parameter> ::std::ops::Index<usize> for Vector<T> {
    type Output = T;
    fn index<'a>(&'a self, index: usize) -> &'a T {
        &self.data[index]
    }
}*/

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