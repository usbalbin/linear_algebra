
pub struct Vector<T> {
    data: Vec<T>
}

impl<T: Clone> Vector<T> {
    pub fn new(default_val: T, size: usize) -> Vector<T> {
        Vector {
            data: vec![default_val; size]
        }
    }

    pub unsafe fn uninitialized(size: usize) -> Vector<T> {
        Vector::new(::std::mem::uninitialized(), size)
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

    fn from_for_each2<F: Fn(&T, &T) -> T>(a: &Vector<T>, b: &Vector<T>, p: F) -> Vector<T> {
        assert_eq!(a.len(), b.len());

        let mut res = unsafe{ Vector::uninitialized(a.len()) };
        for (r, (s, o)) in res.data.iter_mut().zip(
            a.data.iter().zip(b.data.iter())
        ) {
            *r = p(s,o);
        }
        res
    }

    pub fn iter(&self) -> ::std::slice::Iter<T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> ::std::slice::IterMut<T> {
        self.data.iter_mut()
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

impl<'a, 'b, T: Copy + ::std::ops::Sub<T, Output=T>> ::std::ops::Sub<&'b Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;
    fn sub(self, other: &'b Vector<T>) -> Vector<T> {
        Vector::from_for_each2(self, other, |a: &T, b: &T|{
            *a - *b
        })
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
        writeln!(f, "}}")
    }
}