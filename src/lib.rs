use nalgebra::Matrix;
use numpy::npyffi::objects::PyArrayObject;
use numpy::{npyffi};
use pyo3::AsPyPointer;

/// `nalgebra` buffer type for matrices created by the default allocator.
type Buffer<N, R, C> = <nalgebra::base::DefaultAllocator as nalgebra::base::allocator::Allocator<N, R, C>>::Buffer;

//fn matrix_to_python<'py>(py: pyo3::Python<'py>, matrix: &MatrixD) -> &'py PyArray<f64, Dim> {
//	// TODO: What if matrix is not contiguous?
//	// TODO: Borrow data instead.
//	PyArray::from_slice(py, matrix.as_slice())
//		.reshape(matrix.shape()).unwrap()
//}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum Dimension {
	Static(usize),
	Dynamic,
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Error {
	WrongObjectType(WrongObjectTypeError),
	WrongShape(WrongShapeError),
	WrongDataType(WrongDataTypeError),
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct WrongObjectTypeError {
	pub expected: String,
	pub actual: String,
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct WrongShapeError {
	pub expected: [Dimension; 2],
	pub actual: Vec<usize>
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct WrongDataTypeError {
	pub expected: numpy::NpyDataType,
	pub actual: String,
}

pub fn matrix_from_python<'a, N, R, C>(input: &'a pyo3::types::PyAny) -> Result<nalgebra::Matrix<N, R, C, Buffer<N, R, C>>, Error>
where
	N: nalgebra::Scalar + numpy::TypeNum,
	R: nalgebra::Dim,
	C: nalgebra::Dim,
	nalgebra::base::default_allocator::DefaultAllocator: nalgebra::base::allocator::Allocator<N, R, C>,
{
	unsafe { matrix_from_py_object(input.as_ptr()) }
}

pub unsafe fn matrix_from_py_object<N, R, C>(
	array: *mut pyo3::ffi::PyObject
) -> Result<nalgebra::Matrix<N, R, C, Buffer<N, R, C>>, Error>
where
	N: nalgebra::Scalar + numpy::TypeNum,
	R: nalgebra::Dim,
	C: nalgebra::Dim,
	nalgebra::base::default_allocator::DefaultAllocator: nalgebra::base::allocator::Allocator<N, R, C>,
{
	let array = cast_to_py_array(array)?;
	let [input_rows, input_cols] = check_shape::<R, C>(array)?;
	check_equiv_dtype::<N>(array)?;

	let mut output = Matrix::<N, R, C, _>::new_uninitialized_generic(R::from_usize(input_rows), C::from_usize(input_cols));
	copy_data_to_matrix(&mut output, array);
	Ok(output)
}

unsafe fn copy_data_to_matrix<N, R, C, S>(output: &mut Matrix<N, R, C, S>, array: *mut PyArrayObject)
where
	N: nalgebra::Scalar,
	R: nalgebra::Dim,
	C: nalgebra::Dim,
	S: nalgebra::base::storage::StorageMut<N, R, C>,
{
	let row_stride = *(*array).strides.add(0) as usize;
	let col_stride = *(*array).strides.add(1) as usize;

	let data = (*array).data;

	// TODO: optimize contiguous arrays into a single memcpy.

	for r in 0..output.nrows() {
		for c in 0..output.ncols() {
			output[(r, c)] = *(data.add(r * row_stride + c * col_stride) as *const N);
		}
	}
}

unsafe fn cast_to_py_array(object: *mut pyo3::ffi::PyObject) -> Result<*mut PyArrayObject, WrongObjectTypeError> {
	if npyffi::array::PyArray_Check(object) == 1 {
		Ok(&mut *(object as *mut npyffi::objects::PyArrayObject))
	} else {
		return Err(WrongObjectTypeError {
			actual: object_type_string(object),
			expected: String::from("numpy.ndarray"),
		})
	}
}

unsafe fn check_shape<R, C>(array: *mut PyArrayObject) -> Result<[usize; 2], WrongShapeError>
where
	R: nalgebra::Dim,
	C: nalgebra::Dim,
{
	let expected = [
		R::try_to_usize().map(Dimension::Static).unwrap_or(Dimension::Dynamic),
		C::try_to_usize().map(Dimension::Static).unwrap_or(Dimension::Dynamic),
	];

	if (*array).nd as usize != expected.len() {
		return Err(WrongShapeError {
			expected,
			actual: shape(array),
		});
	}

	let input_rows = *(*array).dimensions.add(0) as usize;
	let input_cols = *(*array).dimensions.add(1) as usize;

	let rows_ok = if let Dimension::Static(expected_rows) = expected[0] {
		input_rows == expected_rows
	} else {
		true
	};

	let cols_ok = if let Dimension::Static(expected_cols) = expected[1] {
		input_cols == expected_cols
	} else {
		true
	};

	if rows_ok && cols_ok {
		Ok([input_rows, input_cols])
	} else {
		Err(WrongShapeError {
			expected,
			actual: shape(array),
		})
	}
}

unsafe fn check_equiv_dtype<N: numpy::TypeNum>(array: *mut PyArrayObject) -> Result<(), WrongDataTypeError> {
	if npyffi::array::PY_ARRAY_API.PyArray_EquivTypenums((*(*array).descr).type_num, N::typenum_default()) == 1 {
		Ok(())
	} else {
		Err(WrongDataTypeError {
			actual: data_type_string(array),
			expected: N::npy_data_type(),
		})
	}
}

unsafe fn object_type_string(object: *mut pyo3::ffi::PyObject) -> String {
	let py_type = (*object).ob_type;
	let name = (*py_type).tp_name;
	let name = std::ffi::CStr::from_ptr(name).to_bytes();
	String::from_utf8_lossy(name).into_owned()
}

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

impl From<WrongShapeError> for Error {
	fn from(other: WrongShapeError) -> Self {
		Self::WrongShape(other)
	}
}

impl From<WrongDataTypeError> for Error {
	fn from(other: WrongDataTypeError) -> Self {
		Self::WrongDataType(other)
	}
}

impl std::fmt::Display for Error {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			Self::WrongObjectType(e) => write!(f, "{}", e),
			Self::WrongShape(e)      => write!(f, "{}", e),
			Self::WrongDataType(e)   => write!(f, "{}", e),
		}
	}
}

impl std::fmt::Display for WrongObjectTypeError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "wrong object type: expected {}, found {}", self.expected, self.actual)
	}
}

impl std::fmt::Display for WrongShapeError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "wrong array shape: expected {:?}, found {:?}", self.expected, self.actual)
	}
}

impl std::fmt::Display for WrongDataTypeError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "wrong array data type: expected {:?}, found {}", self.expected, self.actual)
	}
}

impl std::error::Error for Error {}
impl std::error::Error for WrongObjectTypeError {}
impl std::error::Error for WrongShapeError {}
impl std::error::Error for WrongDataTypeError {}
