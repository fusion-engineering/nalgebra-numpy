use nalgebra::Matrix;
use numpy::PyArray;
use pyo3::IntoPy;

/// Copy a nalgebra matrix to a numpy ndarray.
///
/// This does not create a view of the nalgebra matrix.
/// As such, the matrix can be dropped without problem.
pub fn matrix_to_numpy<'py, N, R, C, S>(py: pyo3::Python<'py>, matrix: &Matrix<N, R, C, S>) -> pyo3::PyObject
where
	N: nalgebra::Scalar + numpy::Element,
	R: nalgebra::Dim,
	C: nalgebra::Dim,
	S: nalgebra::storage::Storage<N, R, C>,
{
	let array = PyArray::new(py, (matrix.nrows(), matrix.ncols()), false);
	for r in 0..matrix.nrows() {
		for c in 0..matrix.ncols() {
			unsafe {
				*array.uget_mut((r, c)) = matrix[(r, c)].clone();
			}
		}
	}

	array.into_py(py)
}
