#![feature(iterator_step_by)]

pub mod vector;
pub mod matrix;

//#[test]
pub fn vec_mat_mul() {
    use vector::*;
    use matrix::*;

    let a = Vector::from_vec(vec![1, 2, 3, 4]);
    let b = Vector::from_vec(vec![5, 4, 9, 4]);
    let m = Matrix::from_vec(vec![
        1, 0, 0,
        0, 2, 0,
        0, 0, 3,
        1, 0, 0
    ], 4, 3);

    let p = &a * &m;
    println!("{}", p);
    for (p, b) in p.iter().zip(b.iter()) {
        assert_eq!(p, b);
    }
}

//#[test]
pub fn vec_add_and_scalar_mul() {
    use vector::*;

    let a = Vector::from_vec(vec![1, 2, 3, 4]);
    let r = Vector::from_vec(vec![2, 4, 6, 8]);

    let s = &a + &a;
    let p = &a * 2;

    for (r, (s, p)) in r.iter().zip(s.iter().zip(p.iter())) {
        assert_eq!(r, s);
        assert_eq!(s, p);
    }
}