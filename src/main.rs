#![feature(const_generics)]

use linalg::Matrix;

fn main() {
    let m: Matrix<f32, 4, 4> = Matrix::new(&[
        2.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 2.0, 0.0,
        0.0, 0.0, 0.0, 2.0,
    ]);

    let n: Matrix<f32, 4, 4> = Matrix::new(&[
        3.0, 0.0, 0.0, 0.0,
        0.0, 4.0, 0.0, 0.0,
        0.0, 0.0, 5.0, 0.0,
        0.0, 0.0, 0.0, 6.0,
    ]);

    println!("{}", m);

    println!("{}", m.ref_elem_mul(&n));

    println!("{}", &m * &n);
}
