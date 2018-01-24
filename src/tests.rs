

type TestType = u32;

#[test]
fn vec_eq() {
    use vector::*;

    let a: Vector<TestType> = Vector::from_vec(vec![1, 2, 3, 4]);
    let b: Vector<TestType> = Vector::from_vec(vec![1, 2, 3, 5]);
    let c: Vector<TestType> = Vector::from_vec(vec![1, 2, 3]);
    let d: Vector<TestType> = Vector::from_vec(vec![1, 2, 3, 4]);

    assert_eq!(a, a);
    assert_ne!(a, c);
    assert_ne!(a, b);
    assert_eq!(a, d);
}

#[test]
fn vec_mat_mul() {
    use vector::*;
    use matrix::*;

    let a: Vector<TestType> = Vector::from_vec(vec![1, 2, 3, 4]);
    let b: Vector<TestType> = Vector::from_vec(vec![5, 4, 9]);
    let m: Matrix<TestType> = Matrix::from_vec(vec![
        1, 0, 0,
        0, 2, 0,
        0, 0, 3,
        1, 0, 0
    ], 4, 3);

    let p: Vector<TestType> = &a * &m;
    assert_eq!(p, b);
}

#[test]
fn mat_mat_mul() {
    use matrix::*;

    let a: Matrix<TestType> = Matrix::from_vec(vec![
        1, 2, 3,
        4, 5, 6,
    ], 2, 3);

    let b: Matrix<TestType> = Matrix::from_vec(vec![
        7, 8,
        9, 10,
        11, 12
    ], 3, 2);

    let c: Matrix<TestType> = Matrix::from_vec(vec![
        58, 64,
        139, 154
    ], 2, 2);

    assert_eq!(c, &a * &b);
}

#[test]
pub fn vec_transpose_mat_mul() {
    use vector::*;
    use matrix::*;

    let a: Vector<TestType> = Vector::from_vec(vec![1, 2, 3, 4]);
    let b: Vector<TestType> = Vector::from_vec(vec![5, 4, 9]);
    let m: Matrix<TestType> = Matrix::from_vec(vec![
        1, 0, 0, 1,
        0, 2, 0, 0,
        0, 0, 3, 0
    ], 3, 4);

    let p = mul_transpose_mat(&a, &m);
    assert_eq!(p, b);
}

#[test]
pub fn vec_add() {
    use vector::*;

    let a: Vector<TestType> = Vector::from_vec(vec![1, 2, 3, 4]);
    let r: Vector<TestType> = Vector::from_vec(vec![2, 4, 6, 8]);

    let s = &a + &a;

    assert_eq!(r, s);
}

#[test]
pub fn vec_mul_scl() {
    use vector::*;

    let a: Vector<TestType> = Vector::from_vec(vec![1, 2, 3, 4]);
    let r: Vector<TestType> = Vector::from_vec(vec![2, 4, 6, 8]);


    let p: Vector<TestType> = &a * 2;

    assert_eq!(r, p);
}

#[test]
pub fn save_load_vec() {
    use vector::*;
    let path = "vec.tmp";

    let a: Vector<TestType> = Vector::from_vec(vec![0x01_23_45_67, 0x89_AB_CD_EF, 0xFE_DC_BA_98, 0x76_54_32_10]);
    a.save(path).unwrap();

    let b = unsafe{ Vector::<TestType>::open(path).unwrap() };
    ::std::fs::remove_file(path).unwrap();


    assert_eq!(a, b);
}

#[test]
pub fn save_load_mat() {
    use matrix::*;
    let path = "vec.tmp";

    let a: Matrix<TestType> = Matrix::from_vec(vec![
        0x01_23_45_67, 0x89_AB_CD_EF, 0x13_57_9B_DF,
        0xFE_DC_BA_98, 0x76_54_32_10, 0x02_46_8A_CE
    ], 2, 3);
    a.save(path).unwrap();

    let b = unsafe{ Matrix::<TestType>::open(path).unwrap() };
    ::std::fs::remove_file(path).unwrap();

    assert_eq!(a, b);
}