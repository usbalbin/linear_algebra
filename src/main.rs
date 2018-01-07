
extern crate linear_algebra;

use linear_algebra::vector::Vector;

fn main() {
    let a = Vector::from_vec(vec![1, 2, 3, 4]);
    let b = Vector::from_vec(vec![1, 2, 3, 4]);

    println!("{}", &a + &b)
}