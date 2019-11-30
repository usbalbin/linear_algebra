use crate::matrix::Matrix;

pub type Vector<T, const N: usize> = Matrix<T, {N}, 1>;
