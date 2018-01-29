
extern crate ocl;

use ::std::ops::{
    Add, AddAssign,
    Sub, SubAssign,
    Mul, MulAssign,
    Div, DivAssign,
    Neg
};


pub trait Parameter: ocl::OclPrm {
    fn type_to_str() -> &'static str;
}


pub trait Real: Parameter +
    Add + AddAssign +
    Sub + SubAssign +
    Mul + MulAssign +
    Div + DivAssign +
    Neg
{
    fn sqrt(self) -> Self;
    fn one() -> Self;
    fn zero() -> Self;
}

macro_rules! impl_type_to_str {
    ($( $ty:ident ),+) => {
        $( impl Parameter for $ty {
            fn type_to_str() -> &'static str {
                match stringify!($ty) {
                    "u8" => "uchar",
                    "i8" => "uchar",

                    "u16" => "ushort",
                    "i16" => "short",

                    "u32" => "uint",
                    "i32" => "int",

                    "u64" => "ulong",
                    "i64" => "long",

                    "usize" => if ::std::mem::size_of::<$ty>() == 8 { "ulong"} else { "uint" },
                    "isize" => if ::std::mem::size_of::<$ty>() == 8 { "long"} else { "int" },

                    "f32" => "float",
                    "f64" => "double",
                    _ => panic!("Invalid type")
                }
            }
        } )+
    }
}

macro_rules! impl_sqrt {
    ($( $ty:ident ),+) => {
        $( impl Real for $ty {
            #[allow(unconditional_recursion)]
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            fn one() -> Self {
                1.0 as $ty
            }
            fn zero() -> Self {
                0.0 as $ty
            }
        } )+
    }
}

impl_type_to_str!(u8, i8, u16, i16, u32, i32, u64, i64, usize, isize, f32, f64);

impl_sqrt!(f32, f64);
