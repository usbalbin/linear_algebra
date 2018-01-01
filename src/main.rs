extern crate linear_algebra;

use linear_algebra::*;
use linear_algebra::vector::*;

fn main() {
    let v = vector::Vector::from_vec(vec![0, 1, 2, 3]);
    println!("{}", dot(&v, v.iter()));

    vec_mat_mul();
}