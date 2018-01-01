
use matrix::Matrix;

use ::std::ops::{Add, Mul};

pub fn dot<'a, T, IT>(vector: &Vector<T>, it: IT) -> T
    where
        T: 'a +
        Copy +
        Add +
        Mul +
        ::std::iter::Sum<<T as ::std::ops::Mul>::Output>,
        IT: Iterator<Item = &'a T>
{
    //assert_eq!(vector.len(), it.count());
    vector.iter().zip(it).map(|(v, r)| *v * *r).sum()
}

pub struct Vector<T> {
    data: Vec<T>
}

impl<T> Vector<T> {
    pub fn new(default_val: T, size: usize) -> Vector<T>
        where T: Clone
    {
        Vector {
            data: vec![default_val; size]
        }
    }

    pub unsafe fn uninitialized(size: usize) -> Vector<T> {
        let mut data = Vec::with_capacity(size);
        data.set_len(size);
        Vector {
            data
        }
    }

    pub fn from_vec(v: Vec<T>) -> Vector<T> {
        Vector {
            data: v
        }
    }

    pub fn generate<F: FnMut(usize)-> T>(f: F, size: usize) -> Vector<T> {
        let data = (0..size).map(f).collect();

        Vector {
            data
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn from_for_each2<F: Fn(&T, &T) -> T>(a: &Vector<T>, b: &Vector<T>, p: F) -> Vector<T> {
        assert_eq!(a.len(), b.len());

        let mut res = unsafe{ Vector::uninitialized(a.len()) };
        for (r, (s, o)) in res.data.iter_mut().zip(
            a.data.iter().zip(b.data.iter())
        ) {
            *r = p(s,o);
        }
        res
    }

    pub fn for_each_mut<F: Fn(&mut T, &T)>(&mut self, other: &Vector<T>, p: F){
        assert_eq!(self.len(), other.len());

        for (s, o) in self.data.iter_mut().zip(other.data.iter()) {
            p(s, o);
        }
    }

    pub fn iter(&self) -> ::std::slice::Iter<T> {
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
    }

    /// Applies p for every element in vector
    pub fn map_mut<F: FnMut(&mut T)>(&mut self, mut p: F) {
        for elem in self.iter_mut() {
            p(elem)
        }
    }

    /// Returns copy of self with p applied to each element
    pub fn map<F: FnMut(&T)->T >(&self, p: F) -> Vector<T> {
        Vector {
            data: self.data.iter().map(p).collect()
        }
    }
}

impl<'a, 'b, T: Copy + ::std::ops::Add<T, Output=T>> ::std::ops::Add<&'b Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;
    fn add(self, other: &'b Vector<T>) -> Vector<T> {
        Vector::from_for_each2(self, other, |a: &T, b: &T|{
            *a + *b
        })
    }
}

impl<'a, T: Copy + ::std::ops::AddAssign<T>> ::std::ops::AddAssign<&'a Vector<T>> for Vector<T> {
    fn add_assign(&mut self, other: &'a Vector<T>) {
        Vector::for_each_mut(self, other, |a, b|{
            *a += *b
        });
    }
}

impl<'a, 'b, T: Copy + ::std::ops::Sub<T, Output=T>> ::std::ops::Sub<&'b Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;
    fn sub(self, other: &'b Vector<T>) -> Vector<T> {
        Vector::from_for_each2(self, other, |a: &T, b: &T|{
            *a - *b
        })
    }
}

impl<'a, T: Copy + ::std::ops::SubAssign<T>> ::std::ops::SubAssign<&'a Vector<T>> for Vector<T> {
    fn sub_assign(&mut self, other: &'a Vector<T>) {
        Vector::for_each_mut(self, other, |a, b|{
            *a -= *b
        });
    }
}

//Mul by scalar
impl<'a, 'b, T: Copy + ::std::ops::Mul<T, Output=T>> ::std::ops::Mul<T> for &'a Vector<T> {
    type Output = Vector<T>;
    fn mul(self, scalar: T) -> Vector<T> {
        let mut res = unsafe{ Vector::uninitialized(self.len()) };
        for (r, s) in res.data.iter_mut().zip(self.data.iter()) {
            *r = *s * scalar;
        }
        res
    }
}

impl<'a, 'b, T: Copy + ::std::ops::Mul<T, Output=T>> ::std::ops::Mul<&'b Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;
    fn mul(self, other: &'b Vector<T>) -> Vector<T> {
        Vector::from_for_each2(self, other, |a: &T, b: &T|{
            *a * *b
        })
    }
}

impl<'a, 'b, T> ::std::ops::Mul<&'b Matrix<T>> for &'a Vector<T>
    where T: Copy + Mul<T, Output=T> + Add + ::std::iter::Sum
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
pub fn mul_transpose_mat<T: Copy + Mul<T, Output=T> + Add + ::std::iter::Sum>(vector: &Vector<T>, other_m: &Matrix<T>) -> Vector<T> {
    assert_eq!(vector.len(), other_m.get_col_count());
    let result =
        (0..other_m.get_row_count())//For each row
            .map(|i| dot(vector, other_m.get_row(i)));
    Vector{
        data: result.collect()
    }
}

impl<'a, 'b, T: Copy + ::std::ops::Div<T, Output=T>> ::std::ops::Div<&'b Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;
    fn div(self, other: &'b Vector<T>) -> Vector<T> {
        Vector::from_for_each2(self, other, |a: &T, b: &T|{
            *a / *b
        })
    }
}

//Div by scalar
impl<'a, 'b, T: Copy + ::std::ops::Div<T, Output=T>> ::std::ops::Div<T> for &'a Vector<T> {
    type Output = Vector<T>;
    fn div(self, scalar: T) -> Vector<T> {
        let mut res = unsafe{ Vector::uninitialized(self.len()) };
        for (r, s) in res.data.iter_mut().zip(self.data.iter()) {
            *r = *s / scalar;
        }
        res
    }
}

impl<T> ::std::ops::IndexMut<usize> for Vector<T> {
    fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut T {
        &mut self.data[index]
    }
}

impl<T> ::std::ops::Index<usize> for Vector<T> {
    type Output = T;
    fn index<'a>(&'a self, index: usize) -> &'a T {
        &self.data[index]
    }
}

impl<T: ::std::fmt::Display + Clone> ::std::fmt::Display for Vector<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result{
        write!(f, "{{ {}", self.data[0])?;
        for i in 1..self.len() {
            write!(f, ", {}", self.data[i])?;
        }
        write!(f, " }}")
    }
}