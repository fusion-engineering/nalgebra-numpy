use nalgebra::{Matrix, DMatrix, Dynamic};
use numpy::{PyArray, npyffi};
use pyo3::AsPyPointer;

/// `ndarray` dimension type for two-dimensional arrays.
type Dim = ndarray::Dim<[usize; 2]>;

/// `nalgebra` buffer type for matrices created by the default allocator.
type Buffer<N, R, C> = <nalgebra::base::DefaultAllocator as nalgebra::base::allocator::Allocator<N, R, C>>::Buffer;

//fn matrix_to_python<'py>(py: pyo3::Python<'py>, matrix: &MatrixD) -> &'py PyArray<f64, Dim> {
//	// TODO: What if matrix is not contiguous?
//	// TODO: Borrow data instead.
//	PyArray::from_slice(py, matrix.as_slice())
//		.reshape(matrix.shape()).unwrap()
//}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Error {
	NotAnArray,
	WrongDataType, // TODO: add actual and expected data type.
	WrongDimension, // TODO: add actual and expected dimensions.
}

// pub fn dynamic_matrix_from_python<N>(input: &pyo3::types::PyAny) -> Result<DMatrix<N>, Error>
// where
// 	N: nalgebra::Scalar + numpy::types::TypeNum,
// {
// 	matrix_from_python_generic(input, Dynamic::new(0), Dynamic::new(0))
// }

// pub fn matrix_from_python_generic<'a, N, R, C>(input: &'a pyo3::types::PyAny, rows: R, cols: C) -> Result<nalgebra::Matrix<N, R, C, Buffer<N, R, C>>, Error>
// where
// 	N: nalgebra::Scalar + numpy::TypeNum,
// 	R: nalgebra::Dim,
// 	C: nalgebra::Dim,
// 	nalgebra::base::default_allocator::DefaultAllocator: nalgebra::base::allocator::Allocator<N, R, C>,
// {
// 	let input = input.downcast_ref::<PyArray<N, Dim>>()?;

// 	if let Some(rows) = R::try_to_usize() {
// 		if rows != input.shape()[0] {
// 			return Err(Error::WrongDimension);
// 		}
// 	}

// 	if let Some(cols) = C::try_to_usize() {
// 		if cols != input.shape()[1] {
// 			return Err(Error::WrongDimension);
// 		}
// 	}

// 	// TODO: ndarray and numpy are row major, nalgebra i column major.
// 	Ok(nalgebra::Matrix::<N, R, C, Buffer<N, R, C>>::from_iterator_generic(rows, cols, input.as_array().into_iter().map(|x| *x)))
// }

// TODO: See if we can borrow the matrix and create a view.
pub fn matrix_from_python<'a, N, R, C>(input: &'a pyo3::types::PyAny) -> Result<nalgebra::Matrix<N, R, C, Buffer<N, R, C>>, Error>
where
	N: nalgebra::Scalar + numpy::TypeNum,
	R: nalgebra::Dim + nalgebra::DimName,
	C: nalgebra::Dim + nalgebra::DimName,
	R::Value: std::ops::Mul<C::Value>,
	<R::Value as std::ops::Mul<C::Value>>::Output: generic_array::ArrayLength<N>,
{
	unsafe {
		if npyffi::array::PyArray_Check(input.as_ptr()) != 1 {
			return Err(Error::NotAnArray);
		}

		let input = &mut *(input.as_ptr() as *mut npyffi::objects::PyArrayObject);

		if npyffi::array::PY_ARRAY_API.PyArray_EquivTypenums((*input.descr).type_num, N::typenum_default()) != 1 {
			return Err(Error::WrongDataType);
		}

		if input.nd != 2 {
			return Err(Error::WrongDimension);
		}

		let input_rows = input.dimensions.add(0).read_volatile() as usize;
		let input_cols = input.dimensions.add(1).read_volatile() as usize;
		let row_stride = input.strides.add(0).read_volatile() as usize;
		let col_stride = input.strides.add(1).read_volatile() as usize;

		let rows = R::try_to_usize().unwrap();
		let cols = C::try_to_usize().unwrap();

		if input_rows != rows || input_cols != cols {
			return Err(Error::WrongDimension);
		}

		let mut output = Matrix::<N, R, C, _>::new_uninitialized();

		for r in 0..rows {
			for c in 0..cols {
				output[(r, c)] = *(input.data.add(r * row_stride + c * col_stride) as *const N);
			}
		}

		Ok(output)
	}
}
