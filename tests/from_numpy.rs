#![feature(proc_macro_hygiene)]

use inline_python::{python, Context};
use nalgebra_numpy::matrix_from_numpy;
use nalgebra_numpy::matrix_slice_mut_from_numpy;
use nalgebra::{Dynamic, U2, U3};
use assert2::assert;

#[macro_use]
mod assert;

/// Test conversion of a numpy array to a Matrix3<f64>.
#[test]
fn matrix3_f64() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py).unwrap();
	python! {
		#![context = &context]
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])
	}

	let matrix = context.globals(py).get_item("matrix").unwrap();
	let matrix : nalgebra::Matrix3<f64> = assert_ok!(matrix_from_numpy(py, matrix));

	assert!(matrix == nalgebra::Matrix3::new(
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	));
}

/// Test conversion of a numpy array to a Matrix3<f32>.
#[test]
fn matrix3_f32() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py).unwrap();
	python! {
		#![context = &context]
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		]).astype(np.float32)
	}

	let matrix = context.globals(py).get_item("matrix").unwrap();
	let matrix : nalgebra::Matrix3<f32> = assert_ok!(matrix_from_numpy(py, matrix));

	assert!(matrix == nalgebra::Matrix3::new(
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	));
}

/// Test conversion of a numpy array to a DMatrix3<f64>.
#[test]
fn matrixd() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py).unwrap();
	python! {
		#![context = &context]
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])
	}

	let matrix = context.globals(py).get_item("matrix").unwrap();

	let matrix : nalgebra::DMatrix<f64> = assert_ok!(matrix_from_numpy(py, matrix));
	assert!(matrix == nalgebra::DMatrix::from_row_slice(3, 3, &[
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	]));
}

/// Test conversion of a numpy array to a MatrixMN<f64, Dynamic, U3>.
#[test]
fn matrix3d() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py).unwrap();
	python! {
		#![context = &context]
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])
	}

	let matrix = context.globals(py).get_item("matrix").unwrap();

	let matrix : nalgebra::MatrixMN<f64, Dynamic, U3> = assert_ok!(matrix_from_numpy(py, matrix));
	assert!(matrix == nalgebra::MatrixMN::<f64, Dynamic, U3>::from_row_slice(&[
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	]));
}

/// Test conversion of a numpy array to a MatrixMN<f64, U3, Dynamic>.
#[test]
fn matrixd3() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py).unwrap();
	python! {
		#![context = &context]
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])
	}

	let matrix = context.globals(py).get_item("matrix").unwrap();

	let matrix : nalgebra::MatrixMN<f64, U3, Dynamic> = assert_ok!(matrix_from_numpy(py, matrix));
	assert!(matrix == nalgebra::MatrixMN::<f64, U3, Dynamic>::from_row_slice(&[
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	]));
}

/// Test conversion of a non-coniguous numpy array.
#[test]
fn non_contiguous() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py).unwrap();
	python! {
		#![context = &context]
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])[0:2, 0:2];
	}

	let matrix = context.globals(py).get_item("matrix").unwrap();

	let matrix : nalgebra::MatrixN<f64, U2> = assert_ok!(matrix_from_numpy(py, matrix));
	assert!(matrix == nalgebra::MatrixN::<f64, U2>::new(
		1.0, 2.0,
		4.0, 5.0,
	));
}

/// Test conversion of a column-major numpy array.
#[test]
fn column_major() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py).unwrap();
	python! {
		#![context = &context]
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		], order='F');
	}

	let matrix = context.globals(py).get_item("matrix").unwrap();

	let matrix : nalgebra::MatrixN<f64, U3> = assert_ok!(matrix_from_numpy(py, matrix));
	assert!(matrix == nalgebra::Matrix3::new(
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	));
}

/// Test conversion of a column-major numpy array.
#[test]
fn mutable_view() {
	let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
	let context = Context::new_with_gil(py).unwrap();
	python! {
		#![context = &context]
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		]);

		assert matrix[1, 2] == 6.0
	}

	let matrix = context.globals(py).get_item("matrix").unwrap();

	let mut matrix : nalgebra::MatrixSliceMut<f64, U3, U3, _, _> = assert_ok!(unsafe { matrix_slice_mut_from_numpy(py, matrix) });

	matrix[(1, 2)] = 1337.0;

	// TODO: Do we need to drop the matrix view here to avoid UB?

	python! {
		#![context = &context]
		assert matrix[1, 2] == 1337
	}
}
