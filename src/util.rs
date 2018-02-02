


/// Read u64 to from file.
/// NOTE! File will be interpreted in the current systems endianness
pub unsafe fn read_u64(file: &mut ::std::fs::File) -> Result<u64, ::std::io::Error> {
    use std::io::Read;

    let mut elem_size = [0u8; 8];
    file.read_exact(&mut elem_size)?;
    Ok(::std::mem::transmute(elem_size))
}

/// Write u64 to from file.
/// NOTE! File will be interpreted in the current systems endianness
pub fn write_u64(file: &mut ::std::fs::File, x: u64) -> Result<(), ::std::io::Error> {
    use std::io::Write;

    let elem_size: [u8; 8] = unsafe { ::std::mem::transmute(x) };
    file.write_all(&elem_size)
}