
use vector::*;

pub struct Matrix<T> {
    data: Vector<T>,
    row_count: usize,
    col_count: usize
}

impl<T: Copy> Matrix<T> {
    pub fn new(default_value: T, row_count: usize, col_count: usize) -> Matrix<T> {
        Matrix {
            data: Vector::new(default_value, row_count * col_count),
            row_count,
            col_count
        }
    }

    pub unsafe fn uninitialized(row_count: usize, col_count: usize) -> Matrix<T> {
        Matrix::new(::std::mem::uninitialized(), row_count, col_count)
    }

    pub fn from_vec(v: Vec<T>, row_count: usize, col_count: usize) -> Matrix<T> {
        assert_eq!(v.len(), row_count * col_count);
        Matrix {
            data: Vector::from_vec(v),
            row_count,
            col_count
        }
    }
}


impl<'a, 'b, T: Copy + ::std::ops::Add<T, Output=T>> ::std::ops::Add<&'b Matrix<T>> for &'a Matrix<T> {
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

impl<'a, 'b, T: Copy + ::std::ops::Sub<T, Output=T>> ::std::ops::Sub<&'b Matrix<T>> for &'a Matrix<T> {
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

//Mul with scalar
impl<'a, 'b, T: Copy + ::std::ops::Mul<T, Output=T>> ::std::ops::Mul<T> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, scalar: T) -> Matrix<T> {
        Matrix {
            data: &self.data * scalar,
            row_count: self.row_count,
            col_count: self.col_count
        }
    }
}


impl<'a, 'b, T: Copy + ::std::ops::Mul<T, Output=T> + ::std::ops::Add<T, Output=T>> ::std::ops::Mul<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, other: &'b Matrix<T>) -> Matrix<T> {
        assert_eq!(self.col_count, other.row_count);

        let mut res = unsafe{ Matrix::uninitialized(self.row_count, other.col_count) };

        for row in 0..res.row_count {
            for col in 0..res.col_count {
                res.data[row * res.col_count + col + 0] =
                    self.data[row * self.col_count + col + 0] * other.data[0 * other.col_count + col];
                for i in 1..self.col_count {
                    res.data[row * res.col_count + col] =
                        res.data[row * res.col_count + col] +
                            (self.data[row * self.col_count + i] * other.data[i * other.col_count + col]);
                }
            }
        }
        res
    }
}

//Div by scalar
impl<'a, 'b, T: Copy + ::std::ops::Div<T, Output=T>> ::std::ops::Div<T> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn div(self, scalar: T) -> Matrix<T> {
        Matrix {
            data: &self.data / scalar,
            row_count: self.row_count,
            col_count: self.col_count
        }
    }
}

impl<T: ::std::fmt::Display> ::std::fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        writeln!(f, "Mat{}x{} {{", self.row_count, self.col_count)?;
        for row in 0..self.row_count {
            write!(f, "\t {}", self.data[row * self.col_count + 0])?;
            for col in 1..self.col_count {
                write!(f, ", {}", self.data[row * self.col_count + col])?;
            }
            writeln!(f)?;
        }
        writeln!(f, "}}")
    }
}