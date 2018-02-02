# linear_algebra
Linear algebra library for rust. Includes Matrix and Vector, which support the most common math operations. 

The types Vector and Matrix are currently optimized for very large sizes due to the operations being GPU(or multicore CPU) accelerated.
This is done using OpenCL. 

The library is currently in a very early state and even though most of the basic operations probably will stay the same, a lot of other details might change. 
