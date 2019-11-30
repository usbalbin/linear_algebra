macro_rules! default_kernel_args {
    ($builder:expr, $arg:ident = $default:expr) => {{
        $builder.arg_named(stringify!($arg), $default)
    }};
    ($builder:expr, $arg:ident = $default:expr, $($args:ident = $defaults:expr),+) => {{
        default_kernel_args!($builder, $arg = $default);
        default_kernel_args!($builder, $($args = $defaults),+);
    }};
}


macro_rules! generate_kernel_getters {
    (
        fn $kernel:ident($(
            $args:ident = $defaults:expr
        ),+);
    ) => {
        fn $kernel<T: Param>() -> std::sync::MutexGuard<'static, ocl::Kernel> {
            use std::sync::{Mutex, Once};

            static mut KERNEL: Option<Mutex<ocl::Kernel>> = None;
            static INIT: Once = Once::new();

            //This is ok because we only write once
            unsafe {
                INIT.call_once(|| {
                    let mut builder = PRO_QUE
                        .kernel_builder(format!("{}_{}", stringify!($kernel), T::ocl_type_name()));
                    default_kernel_args!(builder, $($args = $defaults),+);
                    KERNEL = Some(Mutex::new(
                        builder
                            .build()
                            .expect("Failed to build kernel")));
                });
                KERNEL.as_ref().unwrap().lock().unwrap()
            }
        }
    };
    (
        fn $kernel:ident($($args:ident = $defaults:expr),+); $(
            fn $kernels:ident($(
                $argss:ident = $defaultss:expr
            ),+);
        )+
    ) => {
        generate_kernel_getters!{
            fn $kernel($(
                $args = $defaults
            ),+);
        }
        generate_kernel_getters!{
            $(
                fn $kernels($(
                    $argss = $defaultss
                ),+);
            )+
        }
    }
}

macro_rules! kernel_set_args {
    ($kernel:expr,) => ();
    ($kernel:expr, $arg:ident = $value:expr) => {{
        $kernel.set_arg(stringify!($arg), $value).expect("Failed to set kernel arg");
    }};
    ($kernel:expr, $arg:ident = $value:expr, $($args:ident = $values:expr),+) => {{
        kernel_set_args!($kernel, $arg = $value);
        kernel_set_args!($kernel, $($args = $values),+);
    }};
}

macro_rules! ref_binop_func {
    ($visablility:vis fn $func:ident(lhs: $this_type:ty, rhs: $other_type:ty) -> res: $output_type:ty {
        $kernel_getter:ident($(
            $kernel_extra_args:ident = $kernel_extra_values:expr
        ),*)
    }) => {
        $visablility fn $func(self: $this_type, rhs: $other_type) -> $output_type {
            let lhs = self;
            let mut res = ocl::Buffer::builder()
                .queue(PRO_QUE.queue().clone())
                .len(R * K)
                .build()
                .expect("Failed to create buffer");

            let kernel = $kernel_getter::<T>();
            kernel.set_arg("lhs", &lhs.data).expect("Failed to set kernel arg lhs");
            kernel.set_arg("rhs", &rhs.data).expect("Failed to set kernel arg rhs");
            kernel.set_arg("res", &mut res).expect("Failed to set kernel arg res");
            kernel_set_args!(kernel, $($kernel_extra_args = $kernel_extra_values),*);

            unsafe {
                kernel
                    .cmd()
                    .global_work_size((K, R))
                    .local_work_size((1, WG_SIZE.min(R)))
                    .enq()
                    .expect("Failed to enqueue kernel");
            }
            Matrix{ data: res }
        }
    };
}

macro_rules! owned_binop_func {
    ($visablility:vis fn $func:ident(lhs: $this_type:ty, rhs: $other_type:ty) -> res: $output_type:ty {
        $kernel_getter:ident($(
            $kernel_extra_args:ident = $kernel_extra_values:expr
        ),*)
    }) => {
        $visablility fn $func(self: $this_type, rhs: $other_type) -> $output_type {
            let mut lhs = self;

            let kernel = $kernel_getter::<T>();
            kernel.set_arg("lhs", &mut lhs.data).expect("Failed to set kernel arg lhs");
            kernel.set_arg("rhs", &rhs.data).expect("Failed to set kernel arg rhs");
            kernel_set_args!(kernel, $($kernel_extra_args = $kernel_extra_values),*);

            unsafe {
                kernel
                    .cmd()
                    .global_work_size((K, R))
                    .local_work_size((1, WG_SIZE.min(R)))
                    .enq()
                    .expect("Failed to enqueue kernel");
            }
            lhs
        }
    };
}

macro_rules! impl_ref_elem_wise_mat_op {
    (impl $op:ident, fn $fn:ident,
        $kernel:ident($(
            $kernel_args:ident = $kernel_value:expr
        ),*)
    ) => {
        impl<'a, T, const R: usize, const K: usize> $op<&Matrix<T, {R}, {K}>> for &'a Matrix<T, {R}, {K}>
            where T: Param + $op<T, Output=T> + Copy
        {
            type Output = Matrix<T, {R}, {K}>;

            ref_binop_func!{
                fn $fn(lhs: &'a Matrix<T, {R}, {K}>, rhs: &Matrix<T, {R}, {K}>) -> res: Matrix<T, {R}, {K}> {
                    $kernel($(
                        $kernel_args = $kernel_value
                    ),*)
                }
            }
        }
    };
}

macro_rules! impl_owned_elem_wise_mat_op {
    (impl $op:ident, fn $fn:ident,
        $kernel:ident($(
            $kernel_args:ident = $kernel_value:expr
        ),*)
    ) => {
        impl<T, const R: usize, const K: usize> $op<&Matrix<T, {R}, {K}>> for Matrix<T, {R}, {K}>
            where T: Param + $op<T, Output=T> + Copy
        {
            type Output = Matrix<T, {R}, {K}>;

            owned_binop_func!{
                fn $fn(lhs: Matrix<T, {R}, {K}>, rhs: &Matrix<T, {R}, {K}>) -> res: Self::Output {
                    $kernel($(
                        $kernel_args = $kernel_value
                    ),*)
                }
            }
        }
    };
}

pub fn generate_kernel_source() -> String {
    use std::io::{BufReader, BufRead};
    use std::fs::File;

    static MUL_SRC: &str = include_str!("mul_kernels.cl");
    static BINOP_SRC: &str = include_str!("binop_kernels.cl");

    let reader = BufReader::new(File::open("elem_types.cfg").expect("Failed to open elem_types.cfg"));

    let mut result = String::new();

    for line in reader.lines() {
        for elem_type in line.expect("Failed to read from elem_types.cfg").split_whitespace() {
            result += &MUL_SRC.replace("{T}", elem_type);
            for (op, op_name) in &[("+", "add"), ("-", "sub"), ("*", "elem_mul"), ("/", "elem_div")] {
                result += &BINOP_SRC
                    .replace("{T}", elem_type)
                    .replace("{op}", op)
                    .replace("{op_name}", op_name);
            }
        }
    }
    result
}
