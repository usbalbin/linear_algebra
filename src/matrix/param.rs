
pub trait Param: ocl::OclPrm {
    fn ocl_type_name() -> &'static str;
}

macro_rules! impl_param {
    ($($t:ident: $s:expr),*) => {
        $(impl Param for $t {
            fn ocl_type_name() -> &'static str {
                $s
            }
        })*
    };
}

impl_param!{
    f32: "float",
    i32: "int",   u32: "uint",
    i16: "short", u16: "ushort",
    i8:  "char",  u8: "uchar"
}
