
extern crate ocl;

pub trait Parameter: ocl::OclPrm {
    fn type_to_str() -> &'static str;
}

// Implements an unsafe trait for a list of types.
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

impl_type_to_str!(u8, i8, u16, i16, u32, i32, u64, i64, usize, isize, f32, f64);