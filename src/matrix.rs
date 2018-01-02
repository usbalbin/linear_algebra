

use vector::*;

//Iterate through every value in given row
pub struct Row<'a, T: 'a> {
    it: ::std::iter::Take<::std::iter::Skip<::std::slice::Iter<'a, T>>>
}

impl<'a, T: Copy> Iterator for Row<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        self.it.next()
    }
}

//Iterate through every value in given column
pub struct Column<'a, T: 'a> {
    it: ::std::iter::Take<::std::iter::StepBy<::std::iter::Skip<::std::slice::Iter<'a, T>>>>
}

impl<'a, T> Iterator for Column<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        self.it.next()
    }
}

pub struct Matrix<T> {
    data: Vector<T>,
    row_count: usize,
    col_count: usize
}

impl<T> Matrix<T> {
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

    pub fn generate<F: FnMut(usize)-> T>(f: F, row_count: usize, col_count: usize) -> Matrix<T> {
        Matrix {
            data: Vector::generate(f, row_count * col_count),
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

    pub fn get_row(&self, row: usize) -> Row<T> {
        assert!(row < self.get_row_count());
        Row {
            it: self.data.iter().skip(row * self.col_count).take(self.col_count)
        }
    }

    pub fn get_col(&self, column: usize) -> Column<T> {
        assert!(column < self.get_col_count());
        Column {
            it: self.data.iter().skip(column).step_by(self.col_count).take(self.row_count)
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

impl<'a, T: Copy + ::std::ops::AddAssign<T>> ::std::ops::AddAssign<&'a Matrix<T>> for Matrix<T> {
    fn add_assign(&mut self, other: &'a Matrix<T>) {
        assert_eq!(self.row_count, other.row_count);
        assert_eq!(self.col_count, other.col_count);
        self.data += &other.data;
    }
}

impl<'a, T: Copy + ::std::ops::AddAssign<T>> ::std::ops::Add<&'a Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn add(mut self, other: &'a Matrix<T>) -> Matrix<T> {
        self += other;
        self
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

impl<'a, T: Copy + ::std::ops::SubAssign<T>> ::std::ops::SubAssign<&'a Matrix<T>> for Matrix<T> {
    fn sub_assign(&mut self, other: &'a Matrix<T>) {
        assert_eq!(self.row_count, other.row_count);
        assert_eq!(self.col_count, other.col_count);
        self.data -= &other.data;
    }
}

impl<'a, T: Copy + ::std::ops::SubAssign<T>> ::std::ops::Sub<&'a Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(mut self, other: &'a Matrix<T>) -> Matrix<T> {
        self -= other;
        self
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

impl<'a, 'b, T: Copy + ::std::ops::Mul<T, Output=T>> ::std::ops::MulAssign<T> for Matrix<T> {
    fn mul_assign(&mut self, scalar: T) {
        self.data *= scalar;
    }
}

impl<'a, 'b, T: Copy + ::std::ops::Mul<T, Output=T>> ::std::ops::Mul<T> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(mut self, scalar: T) -> Matrix<T> {
        self *= scalar;
        self
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

/// Multiplies two vectors as if they where one one-column-matrix and one one-row-matrix respectively
/// resulting in a matrix with row.len() columns and col.len() rows
pub fn mul_column_row<T: ::std::ops::Mul<T, Output=T> + Copy>(column: &Vector<T>, row: &Vector<T>) -> Matrix<T> {
    let col_count = row.len();
    let row_count = column.len();
    let mut data = Vec::with_capacity(col_count * row_count);
    unsafe {
        data.set_len(col_count * row_count);
    }

    for row_index in 0..row_count {
        for col_index in 0..col_count {
            data[row_index * col_count + col_index] = column[row_index] * row[col_index];
        }
    }

    Matrix{
        data: Vector::from_vec(
            data
        ),
        row_count,
        col_count
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