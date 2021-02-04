use nalgebra::base::{SliceStorage, SliceStorageMut};
use nalgebra::{Dynamic, Matrix};
use numpy::npyffi;
use numpy::{npyffi::objects::PyArrayObject};
use pyo3::{types::PyAny, AsPyPointer};

/// Compile-time matrix dimension used in errors.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum Dimension {
	Static(usize),
	Dynamic,
}

/// Compile-time shape of a matrix used in errors.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Shape(Dimension, Dimension);

/// Error that can occur when converting from Python to a nalgebra matrix.
#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Error {
	/// The Python object is not a [`numpy.ndarray`](https://numpy.org/devdocs/reference/arrays.ndarray.html).
	WrongObjectType(WrongObjectTypeError),

	/// The input array is not compatible with the requested nalgebra matrix.
	IncompatibleArray(IncompatibleArrayError),

	/// The input array is not properly aligned.
	UnalignedArray(UnalignedArrayError),
}

/// Error indicating that the Python object is not a [`numpy.ndarray`](https://numpy.org/devdocs/reference/arrays.ndarray.html).
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct WrongObjectTypeError {
	pub actual: String,
}

/// Error indicating that the input array is not compatible with the requested nalgebra matrix.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct IncompatibleArrayError {
	pub expected_shape: Shape,
	pub actual_shape: Vec<usize>,
	pub expected_dtype: numpy::DataType,
	pub actual_dtype: String,
}

/// Error indicating that the input array is not properly aligned.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct UnalignedArrayError;

/// Create a nalgebra view from a numpy array.
///
/// The array dtype must match the output type exactly.
/// If desired, you can convert the array to the desired type in Python
/// using [`numpy.ndarray.astype`](https://numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html).
///
/// # Safety
/// This function creates a const slice that references data owned by Python.
/// The user must ensure that the data is not modified through other pointers or references.
#[allow(clippy::needless_lifetimes)]
pub unsafe fn matrix_slice_from_numpy<'a, N, R, C>(
	_py: pyo3::Python,
	input: &'a PyAny,
) -> Result<nalgebra::MatrixSlice<'a, N, R, C, Dynamic, Dynamic>, Error>
where
	N: nalgebra::Scalar + numpy::Element,
	R: nalgebra::Dim,
	C: nalgebra::Dim,
{
	matrix_slice_from_numpy_ptr(input.as_ptr())
}

/// Create a mutable nalgebra view from a numpy array.
///
/// The array dtype must match the output type exactly.
/// If desired, you can convert the array to the desired type in Python
/// using [`numpy.ndarray.astype`](https://numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html).
///
/// # Safety
/// This function creates a mutable slice that references data owned by Python.
/// The user must ensure that no other Rust references to the same data exist.
#[allow(clippy::needless_lifetimes)]
pub unsafe fn matrix_slice_mut_from_numpy<'a, N, R, C>(
	_py: pyo3::Python,
	input: &'a PyAny,
) -> Result<nalgebra::MatrixSliceMut<'a, N, R, C, Dynamic, Dynamic>, Error>
where
	N: nalgebra::Scalar + numpy::Element,
	R: nalgebra::Dim,
	C: nalgebra::Dim,
{
	matrix_slice_mut_from_numpy_ptr(input.as_ptr())
}

/// Create an owning nalgebra matrix from a numpy array.
///
/// The data is copied into the matrix.
///
/// The array dtype must match the output type exactly.
/// If desired, you can convert the array to the desired type in Python
/// using [`numpy.ndarray.astype`](https://numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html).
pub fn matrix_from_numpy<N, R, C>(py: pyo3::Python, input: &PyAny) -> Result<nalgebra::MatrixMN<N, R, C>, Error>
where
	N: nalgebra::Scalar + numpy::Element,
	R: nalgebra::Dim,
	C: nalgebra::Dim,
	nalgebra::base::default_allocator::DefaultAllocator: nalgebra::base::allocator::Allocator<N, R, C>,
{
	Ok(unsafe { matrix_slice_from_numpy::<N, R, C>(py, input) }?.into_owned())
}

/// Same as [`matrix_slice_from_numpy`], but takes a raw [`PyObject`](pyo3::ffi::PyObject) pointer.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn matrix_slice_from_numpy_ptr<'a, N, R, C>(
	array: *mut pyo3::ffi::PyObject,
) -> Result<nalgebra::MatrixSlice<'a, N, R, C, Dynamic, Dynamic>, Error>
where
	N: nalgebra::Scalar + numpy::Element,
	R: nalgebra::Dim,
	C: nalgebra::Dim,
{
	let array = cast_to_py_array(array)?;
	let shape = check_array_compatible::<N, R, C>(array)?;
	check_array_alignment(array)?;

	let row_stride = Dynamic::new(*(*array).strides.add(0) as usize / std::mem::size_of::<N>());
	let col_stride = Dynamic::new(*(*array).strides.add(1) as usize / std::mem::size_of::<N>());
	let storage = SliceStorage::<N, R, C, Dynamic, Dynamic>::from_raw_parts((*array).data as *const N, shape, (row_stride, col_stride));

	Ok(Matrix::from_data(storage))
}

/// Same as [`matrix_slice_mut_from_numpy`], but takes a raw [`PyObject`](pyo3::ffi::PyObject) pointer.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn matrix_slice_mut_from_numpy_ptr<'a, N, R, C>(
	array: *mut pyo3::ffi::PyObject,
) -> Result<nalgebra::MatrixSliceMut<'a, N, R, C, Dynamic, Dynamic>, Error>
where
	N: nalgebra::Scalar + numpy::Element,
	R: nalgebra::Dim,
	C: nalgebra::Dim,
{
	let array = cast_to_py_array(array)?;
	let shape = check_array_compatible::<N, R, C>(array)?;
	check_array_alignment(array)?;

	let row_stride = Dynamic::new(*(*array).strides.add(0) as usize / std::mem::size_of::<N>());
	let col_stride = Dynamic::new(*(*array).strides.add(1) as usize / std::mem::size_of::<N>());
	let storage = SliceStorageMut::<N, R, C, Dynamic, Dynamic>::from_raw_parts((*array).data as *mut N, shape, (row_stride, col_stride));

	Ok(Matrix::from_data(storage))
}

/// Check if an object is numpy array and cast the pointer.
unsafe fn cast_to_py_array(object: *mut pyo3::ffi::PyObject) -> Result<*mut PyArrayObject, WrongObjectTypeError> {
	if npyffi::array::PyArray_Check(object) == 1 {
		Ok(&mut *(object as *mut npyffi::objects::PyArrayObject))
	} else {
		Err(WrongObjectTypeError {
			actual: object_type_string(object),
		})
	}
}

/// Check if a numpy array is compatible and return the runtime shape.
unsafe fn check_array_compatible<N, R, C>(array: *mut PyArrayObject) -> Result<(R, C), IncompatibleArrayError>
where
	N: numpy::Element,
	R: nalgebra::Dim,
	C: nalgebra::Dim,
{
	// Delay semi-expensive construction of error object using a lambda.
	let make_error = || {
		let expected_shape = Shape(
			R::try_to_usize().map(Dimension::Static).unwrap_or(Dimension::Dynamic),
			C::try_to_usize().map(Dimension::Static).unwrap_or(Dimension::Dynamic),
		);
		IncompatibleArrayError {
			expected_shape,
			actual_shape: shape(array),
			expected_dtype: N::DATA_TYPE,
			actual_dtype: data_type_string(array),
		}
	};

	// Input array must have two dimensions.
	if (*array).nd != 2 {
		return Err(make_error());
	}

	let input_rows = *(*array).dimensions.add(0) as usize;
	let input_cols = *(*array).dimensions.add(1) as usize;

	// Check number of rows in input array.
	if R::try_to_usize().map(|expected| input_rows == expected) == Some(false) {
		return Err(make_error());
	}

	// Check number of columns in input array.
	if C::try_to_usize().map(|expected| input_cols == expected) == Some(false) {
		return Err(make_error());
	}

	let t2: u32 = N::DATA_TYPE.into_ffi_dtype() as u32;
	// Check the data type of the input array.
	if npyffi::array::PY_ARRAY_API.PyArray_EquivTypenums((*(*array).descr).type_num, t2 as i32) != 1 {
		return Err(make_error());
	}

	// All good.
	Ok((R::from_usize(input_rows), C::from_usize(input_cols)))
}

unsafe fn check_array_alignment(array: *mut PyArrayObject) -> Result<(), UnalignedArrayError> {
	if (*array).flags & npyffi::flags::NPY_ARRAY_ALIGNED != 0 {
		Ok(())
	} else {
		Err(UnalignedArrayError)
	}
}

/// Get a string representing the type of a Python object.
unsafe fn object_type_string(object: *mut pyo3::ffi::PyObject) -> String {
	let py_type = (*object).ob_type;
	let name = (*py_type).tp_name;
	let name = std::ffi::CStr::from_ptr(name).to_bytes();
	String::from_utf8_lossy(name).into_owned()
}

/// Get a string representing the data type of a numpy array.
unsafe fn data_type_string(array: *mut PyArrayObject) -> String {
	// Convert the dtype to string.
	// Don't forget to call Py_DecRef in all paths if py_name isn't null.
	let py_name = pyo3::ffi::PyObject_Str((*array).descr as *mut pyo3::ffi::PyObject);
	if py_name.is_null() {
		return String::from("<error converting dtype to string>");
	}

	let mut size = 0isize;
	let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(py_name, &mut size as *mut isize);
	if data.is_null() {
		pyo3::ffi::Py_DecRef(py_name);
		return String::from("<invalid UTF-8 in dtype>");
	}

	let name = std::slice::from_raw_parts(data as *mut u8, size as usize);
	let name = String::from_utf8_unchecked(name.to_vec());
	pyo3::ffi::Py_DecRef(py_name);
	name
}

/// Get the shape of a numpy array as [`Vec`].
unsafe fn shape(object: *mut numpy::npyffi::objects::PyArrayObject) -> Vec<usize> {
	let num_dims = (*object).nd;
	let dimensions = std::slice::from_raw_parts((*object).dimensions as *const usize, num_dims as usize);
	dimensions.to_vec()
}

impl From<WrongObjectTypeError> for Error {
	fn from(other: WrongObjectTypeError) -> Self {
		Self::WrongObjectType(other)
	}
}

impl From<IncompatibleArrayError> for Error {
	fn from(other: IncompatibleArrayError) -> Self {
		Self::IncompatibleArray(other)
	}
}

impl From<UnalignedArrayError> for Error {
	fn from(other: UnalignedArrayError) -> Self {
		Self::UnalignedArray(other)
	}
}

impl std::fmt::Display for Dimension {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			Self::Dynamic => write!(f, "Dynamic"),
			Self::Static(x) => write!(f, "{}", x),
		}
	}
}

impl std::fmt::Display for Shape {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		let Self(rows, cols) = self;
		write!(f, "[{}, {}]", rows, cols)
	}
}

impl std::fmt::Display for Error {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			Self::WrongObjectType(e) => write!(f, "{}", e),
			Self::IncompatibleArray(e) => write!(f, "{}", e),
			Self::UnalignedArray(e) => write!(f, "{}", e),
		}
	}
}

impl std::fmt::Display for WrongObjectTypeError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "wrong object type: expected a numpy.ndarray, found {}", self.actual)
	}
}

impl std::fmt::Display for IncompatibleArrayError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(
			f,
			"incompatible array: expected ndarray(shape={}, dtype='{}'), found ndarray(shape={:?}, dtype={:?})",
			self.expected_shape,
			FormatDataType(&self.expected_dtype),
			self.actual_shape,
			self.actual_dtype,
		)
	}
}

impl std::fmt::Display for UnalignedArrayError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "the input array is not properly aligned for this platform")
	}
}

/// Helper to format [`numpy::DataType`] more consistently.
struct FormatDataType<'a>(&'a numpy::DataType);

impl std::fmt::Display for FormatDataType<'_> {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		let Self(dtype) = self;
		match dtype {
			numpy::DataType::Bool => write!(f, "bool"),
			numpy::DataType::Complex32 => write!(f, "complex32"),
			numpy::DataType::Complex64 => write!(f, "complex64"),
			numpy::DataType::Float32 => write!(f, "float32"),
			numpy::DataType::Float64 => write!(f, "float64"),
			numpy::DataType::Int8 => write!(f, "int8"),
			numpy::DataType::Int16 => write!(f, "int16"),
			numpy::DataType::Int32 => write!(f, "int32"),
			numpy::DataType::Int64 => write!(f, "int64"),
			numpy::DataType::Object => write!(f, "object"),
			numpy::DataType::Uint8 => write!(f, "uint8"),
			numpy::DataType::Uint16 => write!(f, "uint16"),
			numpy::DataType::Uint32 => write!(f, "uint32"),
			numpy::DataType::Uint64 => write!(f, "uint64"),
		}
	}
}

impl std::error::Error for Error {}
impl std::error::Error for WrongObjectTypeError {}
impl std::error::Error for IncompatibleArrayError {}
