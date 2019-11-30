#![feature(const_generics)]

#[macro_use]
extern crate lazy_static;

pub mod matrix;
pub mod vector;

pub use matrix::Matrix;
pub use vector::Vector;
