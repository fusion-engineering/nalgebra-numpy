use assert2::{assert, let_assert};
use inline_python::{python, Context};
use nalgebra::{Dynamic, U2, U3};
use nalgebra_numpy::matrix_from_numpy;
use nalgebra_numpy::matrix_slice_mut_from_numpy;

/// Test conversion of a numpy array to a Matrix3<f64>.
#[test]
#[rustfmt::skip]
fn matrix3_f64() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py);

	context.run(python! {
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])
	});

	let matrix = context.globals(py).get_item("matrix").unwrap();
	let_assert!(Ok(matrix) = matrix_from_numpy::<f64, U3, U3>(py, matrix));

	assert!(matrix == nalgebra::Matrix3::new(
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	));
}

/// Test conversion of a numpy array to a Matrix3<f32>.
#[test]
#[rustfmt::skip]
fn matrix3_f32() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py);
	context.run(python! {
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		]).astype(np.float32)
	});

	let matrix = context.globals(py).get_item("matrix").unwrap();
	let_assert!(Ok(matrix) = matrix_from_numpy::<f32, U3, U3>(py, matrix));

	assert!(matrix == nalgebra::Matrix3::new(
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	));
}

/// Test conversion of a numpy array to a DMatrix3<f64>.
#[test]
#[rustfmt::skip]
fn matrixd() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py);
	context.run(python! {
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])
	});

	let matrix = context.globals(py).get_item("matrix").unwrap();

	let_assert!(Ok(matrix) = matrix_from_numpy::<f64, Dynamic, Dynamic>(py, matrix));
	assert!(matrix == nalgebra::DMatrix::from_row_slice(3, 3, &[
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	]));
}

/// Test conversion of a numpy array to a MatrixMN<f64, Dynamic, U3>.
#[test]
#[rustfmt::skip]
fn matrix3d() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py);
	context.run(python! {
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])
	});

	let matrix = context.globals(py).get_item("matrix").unwrap();

	let_assert!(Ok(matrix) = matrix_from_numpy::<f64, Dynamic, U3>(py, matrix));
	assert!(matrix == nalgebra::MatrixMN::<f64, Dynamic, U3>::from_row_slice(&[
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	]));
}

/// Test conversion of a numpy array to a MatrixMN<f64, U3, Dynamic>.
#[test]
#[rustfmt::skip]
fn matrixd3() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py);
	context.run(python! {
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])
	});

	let matrix = context.globals(py).get_item("matrix").unwrap();

	let_assert!(Ok(matrix) = matrix_from_numpy::<f64, U3, Dynamic>(py, matrix));
	assert!(matrix == nalgebra::MatrixMN::<f64, U3, Dynamic>::from_row_slice(&[
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	]));
}

/// Test conversion of a non-coniguous numpy array.
#[test]
#[rustfmt::skip]
fn non_contiguous() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py);
	context.run(python! {
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])[0:2, 0:2];
	});

	let matrix = context.globals(py).get_item("matrix").unwrap();

	let_assert!(Ok(matrix) = matrix_from_numpy::<f64, U2, U2>(py, matrix));
	assert!(matrix == nalgebra::MatrixN::<f64, U2>::new(
		1.0, 2.0,
		4.0, 5.0,
	));
}

/// Test conversion of a column-major numpy array.
#[test]
#[rustfmt::skip]
fn column_major() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py);
	context.run(python! {
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		], order='F');
	});

	let matrix = context.globals(py).get_item("matrix").unwrap();

	let_assert!(Ok(matrix) = matrix_from_numpy::<f64, U3, U3>(py, matrix));
	assert!(matrix == nalgebra::Matrix3::new(
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	));
}

/// Test conversion of a column-major numpy array.
#[test]
#[rustfmt::skip]
fn mutable_view() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py);
	context.run(python! {
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		]);

		assert matrix[1, 2] == 6.0
	});

	let matrix = context.globals(py).get_item("matrix").unwrap();

	let_assert!(Ok(mut matrix) = unsafe { matrix_slice_mut_from_numpy::<f64, U3, U3>(py, matrix) });

	matrix[(1, 2)] = 1337.0;

	// TODO: Do we need to drop the matrix view here to avoid UB?

	context.run(python! {
		assert matrix[1, 2] == 1337
	});
}
