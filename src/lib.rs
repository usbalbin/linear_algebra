extern crate ocl;

#[macro_use]
extern crate lazy_static;

use ocl::ProQue;
use ocl::Buffer;

pub mod vector;
pub mod matrix;
pub mod traits;
pub mod util;

#[cfg(test)]
mod tests;

use traits::Parameter;
use std::sync::Mutex;
use std::collections::HashMap;
use std::sync::MutexGuard;
use std::ops::Deref;
use std::ops::DerefMut;

struct OclData {
    queue: ocl::ProQue,
    kernels: HashMap<String, Kernels>
}

#[derive(Copy, Clone)]
pub struct KernelParams {
    pub work_group_size: usize,
    pub work_group_count: usize,
    pub global_work_size: usize,
}

pub struct Kernels {
    add_vec_vec: ocl::Kernel,
    sub_vec_vec: ocl::Kernel,
    mul_vec_vec: ocl::Kernel,
    div_vec_vec: ocl::Kernel,
    dot_vec_vec: ocl::Kernel,

    add_assign_vec_vec: ocl::Kernel,
    sub_assign_vec_vec: ocl::Kernel,
    mul_assign_vec_vec: ocl::Kernel,
    div_assign_vec_vec: ocl::Kernel,

    add_vec_scl: ocl::Kernel,
    sub_vec_scl: ocl::Kernel,
    mul_vec_scl: ocl::Kernel,
    div_vec_scl: ocl::Kernel,

    mul_assign_vec_scl: ocl::Kernel,
    div_assign_vec_scl: ocl::Kernel,

    eq_vec: ocl::Kernel,
    sum_vec: ocl::Kernel,
    squared_vec: ocl::Kernel,
    sqrted_vec: ocl::Kernel,
    bitonic_sort_vec: ocl::Kernel,

    //Matrix vec

    mul_vec_mat: ocl::Kernel,
    mul_vec_transpose_mat: ocl::Kernel,

    //Matrix

    mul_mat_mat: ocl::Kernel,

    queue: ocl::Queue,
}

pub struct KernelsGuard<'a, 'b>(MutexGuard<'a, OclData>, &'b str);

impl<'a, 'b> Deref for KernelsGuard<'a, 'b> {
    type Target = Kernels;
    fn deref(&self) -> &Self::Target {
        self.0.kernels.get(self.1).unwrap()
    }
}

impl<'a, 'b> DerefMut for KernelsGuard<'a, 'b> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.kernels.get_mut(self.1).unwrap()
    }
}

lazy_static! {
    pub static ref TYPES: Mutex<Vec<&'static str>> = Mutex::new(vec![
        "uchar",    "char",
        "ushort",   "short",
        "uint",     "int",
        "ulong",    "long",
        "float"
    ]);
}

/// Get OpenCL queue and kernel parameters(work_group_size and global_work_size)
pub fn get_cl_data<T: Parameter>() -> ocl::Queue {
    let data = cl_data::<T>();
    data.queue.queue().clone()
}

/// Get optimal work sizes for kernel
pub fn get_work_sizes(kernel: &ocl::Kernel) -> KernelParams {
    use ocl::enums::{ KernelWorkGroupInfoResult, KernelWorkGroupInfo };

    let result = kernel.wg_info(
        ocl::Device::from(kernel.devices().unwrap()[0]),
        KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple);

    let work_group_size =
        match result {
            KernelWorkGroupInfoResult::PreferredWorkGroupSizeMultiple(s) => s,
            KernelWorkGroupInfoResult::Error(e) => panic!("{}", e),
            _ => panic!(""),
        };
    let work_group_count = 64;                  //TODO: find a way to decide this automatically
    let global_work_size = work_group_count * work_group_size;

    KernelParams{
        work_group_size,
        work_group_count,
        global_work_size
    }
}

/// Get all OpenCL data, internal use only
fn cl_data<'a, T: Parameter>() -> MutexGuard<'a, OclData> {
    lazy_static! {
        static ref CL_DATA: Mutex<OclData> =
        {
            let types = &*TYPES.lock().unwrap();

            let queue = unsafe { setup_queue(types) };

            Mutex::new( OclData {
                queue,
                kernels: HashMap::new(),
            })
        };
    }

    let mut data = CL_DATA.lock().unwrap();
    let ty = T::type_to_str().to_owned();

    if !data.kernels.contains_key(&ty){
        let kernels = unsafe{ setup_kernels::<T>(&data.queue) };
        data.kernels.insert(ty, kernels);
    }

    data
}

fn get_kernels<'a, 'b, T: Parameter>(name: &'b str) -> KernelsGuard<'a, 'b> {
    let d = cl_data::<T>();
    KernelsGuard(d, name)
}

/// Create Kernel object from kernel source in extra_kernels.cl
///
/// Note:
/// In kernel source, use {T} as type when referring to corresponding rust equivalent T.
/// Variable name 'i' is defined as get_global_id(0)
/// All functions should be prefixed with {T}_, for example {T}_add().
/// This is because all source will be compiled once for every type in TYPES vector.
/// IS_{T} will be defined to the current type and can thus be used in preprocessor
/// #ifdef conditions and similar to limit what code is compiled.
pub fn create_kernel<T: Parameter>(kernel_name: &str) -> ocl::Kernel {
    let queue = &mut cl_data::<T>().queue;
    match queue.create_kernel(kernel_name) {
        Ok(kernel) => kernel,
        Err(error) => panic!("Failed to create kernel. Forgot to add kernel source to kernel.cl?: {}", error)
    }
}

fn load_extra_src() -> Result<String, std::io::Error> {
    use std::fs::File;
    use std::io::prelude::*;
    let mut file = File::open("src/extra_kernels.cl")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

unsafe fn setup_kernels<T: Parameter>(queue: &ProQue) -> Kernels {

    #[cfg(test)]
    println!("\nPicked device: {}, {}", queue.device().vendor(), queue.device().name());

    let type_prefix = T::type_to_str().to_owned() + "_";

    let add_vec_vec = queue.create_kernel(&(type_prefix.clone() + "add_vec_vec")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);
    let sub_vec_vec = queue.create_kernel(&(type_prefix.clone() + "sub_vec_vec")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);
    let mul_vec_vec = queue.create_kernel(&(type_prefix.clone() + "mul_vec_vec")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);
    let div_vec_vec = queue.create_kernel(&(type_prefix.clone() + "div_vec_vec")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);
    let dot_vec_vec = queue.create_kernel(&(type_prefix.clone() + "dot_vec_vec")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("a", None)
        .arg_buf_named::<T, Buffer<T>>("b", None)
        .arg_buf_named::<T, Buffer<T>>("results", None)
        .arg_scl_named::<i32>("count", None);


    let add_assign_vec_vec = queue.create_kernel(&(type_prefix.clone() + "add_assign_vec_vec")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);
    let sub_assign_vec_vec = queue.create_kernel(&(type_prefix.clone() + "sub_assign_vec_vec")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);
    let mul_assign_vec_vec = queue.create_kernel(&(type_prefix.clone() + "mul_assign_vec_vec")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);
    let div_assign_vec_vec = queue.create_kernel(&(type_prefix.clone() + "div_assign_vec_vec")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);


    let add_vec_scl = queue.create_kernel(&(type_prefix.clone() + "add_vec_scl")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_scl_named::<T>("B", None);
    let sub_vec_scl = queue.create_kernel(&(type_prefix.clone() + "sub_vec_scl")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_scl_named::<T>("B", None);
    let mul_vec_scl = queue.create_kernel(&(type_prefix.clone() + "mul_vec_scl")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_scl_named::<T>("B", None);
    let div_vec_scl = queue.create_kernel(&(type_prefix.clone() + "div_vec_scl")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_scl_named::<T>("B", None);

    let mul_assign_vec_scl = queue.create_kernel(&(type_prefix.clone() + "mul_assign_vec_scl")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_scl_named::<T>("B", None);
    let div_assign_vec_scl = queue.create_kernel(&(type_prefix.clone() + "div_assign_vec_scl")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_scl_named::<T>("B", None);

    let squared_vec = queue.create_kernel(&(type_prefix.clone() + "squared_vec")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None);

    let sqrted_vec = if T::type_to_str() == "float" || T::type_to_str() == "double" {
        queue.create_kernel(&(type_prefix.clone() + "sqrted_vec")).unwrap()
            .arg_buf_named::<T, Buffer<T>>("C", None)
            .arg_buf_named::<T, Buffer<T>>("A", None)
    } else { //TODO: find better solution to this
        queue.create_kernel(&(type_prefix.clone() + "dummy_kernel")).unwrap()
    };

    let eq_vec = queue.create_kernel(&(type_prefix.clone() + "eq_vec")).unwrap()
        .arg_buf_named::<u8, Buffer<u8>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None);

    let sum_vec = queue.create_kernel(&(type_prefix.clone() + "sum_vec")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("data", None)
        .arg_buf_named::<T, Buffer<T>>("results", None)
        .arg_scl_named::<i32>("count", None);

    let bitonic_sort_vec = queue.create_kernel(&(type_prefix.clone() + "bitonic_sort_vec")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("data", None)
        .arg_scl_named::<i32>("j", None)
        .arg_scl_named::<i32>("k", None);


    //Matrix vec

    let mul_vec_mat = queue.create_kernel(&(type_prefix.clone() + "mul_vec_mat")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None)
        .arg_scl_named::<i32>("B_col_count", None)
        .arg_scl_named::<i32>("A_len", None);

    let mul_vec_transpose_mat = queue.create_kernel(&(type_prefix.clone() + "mul_vec_transpose_mat")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None)
        .arg_scl_named::<i32>("B_col_count", None)
        .arg_scl_named::<i32>("A_len", None);
    //Matrix

    let mul_mat_mat = queue.create_kernel(&(type_prefix.clone() + "mul_mat_mat")).unwrap()
        .arg_buf_named::<T, Buffer<T>>("C", None)
        .arg_buf_named::<T, Buffer<T>>("A", None)
        .arg_buf_named::<T, Buffer<T>>("B", None)
        .arg_scl_named::<i32>("C_col_count", None)
        .arg_scl_named::<i32>("A_col_count", None);

    Kernels {
        add_vec_vec,
        sub_vec_vec,
        mul_vec_vec,
        div_vec_vec,
        dot_vec_vec,

        add_assign_vec_vec,
        sub_assign_vec_vec,
        mul_assign_vec_vec,
        div_assign_vec_vec,

        add_vec_scl,
        sub_vec_scl,
        mul_vec_scl,
        div_vec_scl,

        mul_assign_vec_scl,
        div_assign_vec_scl,

        eq_vec,
        squared_vec,
        sqrted_vec,
        sum_vec,
        bitonic_sort_vec,

        //Matrix vec

        mul_vec_mat,
        mul_vec_transpose_mat,

        //Matrix

        mul_mat_mat,

        queue: queue.queue().clone(),
    }
}

unsafe fn setup_queue(types: &Vec<&str>) -> ProQue {
    let mut builder = ProQue::builder();

    let queue = if let Some((p, d)) = get_gpu() {
        builder
            .platform(p)
            .device(d)
    } else {
        let (p, d) = get_cpu().unwrap();
        builder
            .platform(p)
            .device(d)
    };


    let src = get_src(types);
    let q = match queue
        .src(src)
        .build()
        {
            Err(e) => panic!("Failed to compile kernels with error: {}", e),
            Ok(q) => q,
        };

    q
}

fn get_src(types: &Vec<&str>) -> String {
    let mut res = String::new();
    for ty in types {
        let extra_src = load_extra_src().unwrap_or_default();

        let header = format!("#define IS_{}\n", ty.to_uppercase());
        let src = if ty == &"double" {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
        } else {
            ""
        };
        let src = header + src + include_str!("kernels.cl") + "\n" + &extra_src;
        res += &src.replace("{T}", ty);
        res += &format!("\n#undef IS_{}\n", ty.to_uppercase());
    }

    res
}

fn get_gpu() -> Option<(ocl::Platform, ocl::Device)> {
    if cfg!(feature = "no_gpu") {
        return None;
    }

    let platforms = ocl::Platform::list();
    for platform in platforms {
        let devices =
            match ocl::Device::list(platform, Some(ocl::flags::DEVICE_TYPE_GPU)) {
                Ok(d) => d,
                Err(_) => continue,
            };
        for d in devices {
            return Some((platform, d));
        }
    }
    None
}

fn get_cpu() -> Option<(ocl::Platform, ocl::Device)> {
    let platforms = ocl::Platform::list();
    for platform in platforms {
        let devices =
            match ocl::Device::list(platform, Some(ocl::flags::DEVICE_TYPE_ALL)) {
                Ok(d) => d,
                Err(_) => continue,
            };
        for d in devices {
            return Some((platform, d));
        }
    }
    None
}