
extern crate ocl;

use cl_data;
use traits::Parameter;
use vector::*;


pub struct Matrix<T: Parameter> {
    pub(crate) data: Vector<T>,
    row_count: usize,
    col_count: usize
}

impl<T: Parameter> Matrix<T> {
    pub fn new(default_value: T, row_count: usize, col_count: usize) -> Matrix<T>
        where T: Copy
    {
        Matrix {
            data: Vector::new(default_value, row_count * col_count),
            row_count,
            col_count
        }
    }

    pub unsafe fn uninitialized(row_count: usize, col_count: usize) -> Matrix<T> {
        Matrix {
            data: Vector::uninitialized(row_count * col_count),
            row_count,
            col_count
        }
    }

    pub fn from_vec(v: Vec<T>, row_count: usize, col_count: usize) -> Matrix<T> {
        assert_eq!(v.len(), row_count * col_count);
        Matrix {
            data: Vector::from_vec(v),
            row_count,
            col_count
        }
    }

    pub fn generate<F: FnMut(usize)-> T>(kernel: &mut ocl::Kernel, row_count: usize, col_count: usize) -> Matrix<T> {
        Matrix {
            data: Vector::generate(kernel, row_count * col_count),
            row_count,
            col_count
        }
    }

    pub fn len(&self) -> (usize, usize) {
        (self.row_count, self.col_count)
    }

    pub fn get_row_count(&self) -> usize {
        self.row_count
    }

    pub fn get_col_count(&self) -> usize {
        self.col_count
    }

    pub unsafe fn get_buffer(&self) -> &ocl::Buffer<T> {
        self.data.get_buffer()
    }

    pub unsafe fn get_buffer_mut(&mut self) -> &mut ocl::Buffer<T> {
        self.data.get_buffer_mut()
    }
}


impl<'a, 'b, T: Parameter + ::std::ops::Add<T, Output=T>> ::std::ops::Add<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: &'b Matrix<T>) -> Matrix<T> {
        assert_eq!(self.row_count, other.row_count);
        assert_eq!(self.col_count, other.col_count);

        Matrix {
            data: &self.data + &other.data,
            row_count: self.row_count,
            col_count: self.col_count
        }
    }
}

impl<'a, T: Parameter + ::std::ops::AddAssign<T>> ::std::ops::AddAssign<&'a Matrix<T>> for Matrix<T> {
    fn add_assign(&mut self, other: &'a Matrix<T>) {
        assert_eq!(self.row_count, other.row_count);
        assert_eq!(self.col_count, other.col_count);
        self.data += &other.data;
    }
}

impl<'a, T: Parameter + ::std::ops::AddAssign<T>> ::std::ops::Add<&'a Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn add(mut self, other: &'a Matrix<T>) -> Matrix<T> {
        self += other;
        self
    }
}

impl<'a, 'b, T: Parameter + ::std::ops::Sub<T, Output=T>> ::std::ops::Sub<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: &'b Matrix<T>) -> Matrix<T> {
        assert_eq!(self.row_count, other.row_count);
        assert_eq!(self.col_count, other.col_count);

        Matrix {
            data: &self.data - &other.data,
            row_count: self.row_count,
            col_count: self.col_count
        }
    }
}

impl<'a, T: Parameter + ::std::ops::SubAssign<T>> ::std::ops::SubAssign<&'a Matrix<T>> for Matrix<T> {
    fn sub_assign(&mut self, other: &'a Matrix<T>) {
        assert_eq!(self.row_count, other.row_count);
        assert_eq!(self.col_count, other.col_count);
        self.data -= &other.data;
    }
}

impl<'a, T: Parameter + ::std::ops::SubAssign<T>> ::std::ops::Sub<&'a Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(mut self, other: &'a Matrix<T>) -> Matrix<T> {
        self -= other;
        self
    }
}

//Mul with scalar
impl<'a, 'b, T: Parameter + ::std::ops::Mul<T, Output=T>> ::std::ops::Mul<T> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, scalar: T) -> Matrix<T> {
        Matrix {
            data: &self.data * scalar,
            row_count: self.row_count,
            col_count: self.col_count
        }
    }
}

impl<'a, 'b, T: Parameter + ::std::ops::MulAssign<T>> ::std::ops::MulAssign<T> for Matrix<T> {
    fn mul_assign(&mut self, scalar: T) {
        self.data *= scalar;
    }
}

impl<'a, 'b, T: Parameter + ::std::ops::MulAssign<T>> ::std::ops::Mul<T> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(mut self, scalar: T) -> Matrix<T> {
        self *= scalar;
        self
    }
}


impl<'a, 'b, T> ::std::ops::Mul<&'b Matrix<T>> for &'a Matrix<T>
    where T:
    Parameter +
        ::std::ops::Mul<T, Output=T> +
        ::std::ops::Add<T, Output=T>
{
    type Output = Matrix<T>;
    fn mul(self, other: &'b Matrix<T>) -> Matrix<T> {
        assert_eq!(self.col_count, other.row_count);
        mul_helper(&self.data.data, &other.data.data, self.row_count, self.col_count, other.col_count)
    }
}

/// Multiplies two vectors as if they where one one-column-matrix and one one-row-matrix respectively
/// resulting in a matrix with row.len() columns and col.len() rows
pub fn mul_column_row<T: Parameter + ::std::ops::Mul<T, Output=T>>(column: &Vector<T>, row: &Vector<T>) -> Matrix<T> {
    mul_helper(&column.data, &row.data, column.len(), 1, row.len())
}

fn mul_helper<T: Parameter>(a: &ocl::Buffer<T>, b: &ocl::Buffer<T>, a_row_count: usize, a_col_count: usize, b_col_count: usize) -> Matrix<T> {

    let mut res = unsafe{ Matrix::uninitialized(a_row_count, b_col_count) };

    let kernel = &mut cl_data::<T>().as_mut().unwrap().mul_mat_mat;

    kernel.set_arg_buf_named("C", Some(&mut res.data.data)).unwrap();
    kernel.set_arg_buf_named("A", Some(a)).unwrap();
    kernel.set_arg_buf_named("B", Some(b)).unwrap();

    kernel.set_arg_scl_named::<i32>("C_col_count", res.col_count as i32).unwrap();
    kernel.set_arg_scl_named::<i32>("A_col_count", a_col_count as i32).unwrap();

    unsafe { kernel.cmd().gws(res.get_col_count() * res.get_row_count()).enq().unwrap(); }

    res
}

//Div by scalar
impl<'a, 'b, T: Parameter + ::std::ops::Div<T, Output=T>> ::std::ops::Div<T> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn div(self, scalar: T) -> Matrix<T> {
        Matrix {
            data: &self.data / scalar,
            row_count: self.row_count,
            col_count: self.col_count
        }
    }
}

impl<'a, 'b, T> ::std::cmp::PartialEq for Matrix<T>
    where T: Copy + ::std::cmp::Eq + Parameter
{
    fn eq(&self, other: &Matrix<T>) -> bool {
        self.row_count == other.row_count &&
            self.col_count == other.col_count &&
            self.data == other.data
    }
}

impl<T: Parameter + ::std::fmt::Debug> ::std::fmt::Debug for Matrix<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        let v = self.data.to_vec();
        writeln!(f, "Mat{}x{} {{", self.row_count, self.col_count)?;
        for row in 0..self.row_count {
            write!(f, "\t {:?}", v[row * self.col_count + 0])?;
            for col in 1..self.col_count {
                write!(f, ", {:?}",v[row * self.col_count + col])?;
            }
            writeln!(f)?;
        }
        writeln!(f, "}}")
    }
}

impl<T: Parameter + ::std::fmt::Display> ::std::fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        let v = self.data.to_vec();
        writeln!(f, "Mat{}x{} {{", self.row_count, self.col_count)?;
        for row in 0..self.row_count {
            write!(f, "\t {}", v[row * self.col_count + 0])?;
            for col in 1..self.col_count {
                write!(f, ", {}",v[row * self.col_count + col])?;
            }
            writeln!(f)?;
        }
        writeln!(f, "}}")
    }
}