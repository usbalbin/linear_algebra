
extern crate linear_algebra;

use linear_algebra::vector::Vector;
use linear_algebra::matrix::Matrix;

fn main() {
    let a = Vector::from_vec(vec![1, 2, 3]);
    let b = Vector::from_vec(vec![2, 3, 4]);
    let m = Matrix::from_vec(vec![
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0, 0
    ], 4, 3);

    assert_eq!(a, &a * &m);
    println!("{}, {}", &a + &b, a.len())
}