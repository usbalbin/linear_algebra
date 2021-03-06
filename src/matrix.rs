
extern crate ocl;

use get_kernels;
use traits::Parameter;
use util::*;
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
        let queue = get_kernels::<T>(T::type_to_str()).queue.clone();
        Matrix::<T>::uninitialized_lock_free(row_count, col_count, queue)
    }

    unsafe fn uninitialized_lock_free(row_count: usize, col_count: usize, queue: ocl::Queue) -> Matrix<T> {
        Matrix {
            data: Vector::uninitialized_lock_free(row_count * col_count, queue),
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

    /// Append Vector data to file.
    /// NOTE! Vector will be encoded in the current systems endianness
    ///
    /// Two u64 will be placed first, representing the number of bytes per element and the
    /// number of elements respectively.
    pub fn write_to_file(&self, file: &mut ::std::fs::File) -> ::std::result::Result<(), ::std::io::Error> {

        write_u64(file, ::std::mem::size_of::<T>() as  u64)?;         //Store element size in bytes

        write_u64(file, self.row_count as u64)?;                      //Store row count
        write_u64(file, self.col_count as u64)?;                      //Store column count

        self.data.write_file_only_data(file)                             //Store data
    }

    /// Read data to from file.
    /// NOTE! Buffer will be interpreted in the current systems endianness
    pub unsafe fn read_from_file(file: &mut ::std::fs::File) -> Result<Matrix<T>, ::std::io::Error> {
        let elem_size = read_u64(file)?;


        assert_eq!(elem_size as usize, ::std::mem::size_of::<T>(),
                   "Elem size from buffer does not seem to match, what was expected!\
                 Missmatch in endiannes?"
        );

        let row_count = read_u64(file)? as usize;
        let col_count = read_u64(file)? as usize;

        Ok(Matrix {
            data: Vector::<T>::read_file_only_data(file, (row_count * col_count) as u64)?,
            row_count,
            col_count
        })
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
    pub unsafe fn open(path: &str) -> Result<Matrix<T>, ::std::io::Error> {
        use std::fs::File;

        let mut file = File::open(path).expect("Failed to open file");

        Ok(Self::read_from_file(&mut file).expect("Failed to read all data from file"))
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
    let mut kernels = get_kernels::<T>(T::type_to_str());
    let queue = kernels.queue.clone();
    let kernel = &mut kernels.mul_mat_mat;

    let mut res = unsafe{ Matrix::uninitialized_lock_free(
        a_row_count,
        b_col_count,
        queue
    )};

    kernel.set_arg_buf_named("C", Some(&mut res.data.data)).unwrap();
    kernel.set_arg_buf_named("A", Some(a)).unwrap();
    kernel.set_arg_buf_named("B", Some(b)).unwrap();

    kernel.set_arg_scl_named::<i32>("C_col_count", res.col_count as i32).unwrap();
    kernel.set_arg_scl_named::<i32>("A_col_count", a_col_count as i32).unwrap();

    unsafe {
        let mut event = ocl::Event::empty();
        kernel.cmd().enew(&mut event).gws(res.get_col_count() * res.get_row_count()).enq().unwrap();
        event.wait_for().unwrap();
    }

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
    where T: Copy + ::std::cmp::PartialEq + Parameter
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