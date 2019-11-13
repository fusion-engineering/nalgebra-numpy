#![feature(proc_macro_hygiene)]

use inline_python::{python, Context};
use nalgebra_numpy::matrix_from_python;
use nalgebra::{Dynamic, U2, U3};

#[macro_use]
mod assert;

/// Test conversion of a numpy array to a Matrix3<f64>.
#[test]
fn matrix3_f64() {
	let gil = pyo3::Python::acquire_gil();
	let context = Context::new_with_gil(gil.python()).unwrap();
	python! {
		#![context = &context]
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])
	}

	let matrix = context.globals(gil.python()).get_item("matrix").unwrap();
	let matrix : nalgebra::Matrix3<f64> = matrix_from_python(matrix).unwrap();

	assert_eq!(matrix, nalgebra::Matrix3::new(
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	));
}

/// Test conversion of a numpy array to a Matrix3<f32>.
#[test]
fn matrix3_f32() {
	let gil = pyo3::Python::acquire_gil();
	let context = Context::new_with_gil(gil.python()).unwrap();
	python! {
		#![context = &context]
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		]).astype(np.float32)
	}

	let matrix = context.globals(gil.python()).get_item("matrix").unwrap();
	let matrix : nalgebra::Matrix3<f32> = matrix_from_python(matrix).unwrap();

	assert_eq!(matrix, nalgebra::Matrix3::new(
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	));
}

/// Test conversion of a numpy array to a DMatrix3<f64>.
#[test]
fn matrixd() {
	let gil = pyo3::Python::acquire_gil();
	let context = Context::new_with_gil(gil.python()).unwrap();
	python! {
		#![context = &context]
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])
	}

	let matrix = context.globals(gil.python()).get_item("matrix").unwrap();

	let matrix : nalgebra::DMatrix<f64> = matrix_from_python(matrix).unwrap();
	assert_eq!(matrix, nalgebra::DMatrix::from_row_slice(3, 3, &[
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	]));
}

/// Test conversion of a numpy array to a MatrixMN<f64, Dynamic, U3>.
#[test]
fn matrix3d() {
	let gil = pyo3::Python::acquire_gil();
	let context = Context::new_with_gil(gil.python()).unwrap();
	python! {
		#![context = &context]
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])
	}

	let matrix = context.globals(gil.python()).get_item("matrix").unwrap();

	let matrix : nalgebra::MatrixMN<f64, Dynamic, U3> = matrix_from_python(matrix).unwrap();
	assert_eq!(matrix, nalgebra::MatrixMN::<f64, Dynamic, U3>::from_row_slice(&[
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	]));
}

/// Test conversion of a numpy array to a MatrixMN<f64, U3, Dynamic>.
#[test]
fn matrixd3() {
	let gil = pyo3::Python::acquire_gil();
	let context = Context::new_with_gil(gil.python()).unwrap();
	python! {
		#![context = &context]
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])
	}

	let matrix = context.globals(gil.python()).get_item("matrix").unwrap();

	let matrix : nalgebra::MatrixMN<f64, U3, Dynamic> = matrix_from_python(matrix).unwrap();
	assert_eq!(matrix, nalgebra::MatrixMN::<f64, U3, Dynamic>::from_row_slice(&[
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	]));
}

/// Test conversion of a non-coniguous numpy array.
#[test]
fn non_contiguous() {
	let gil = pyo3::Python::acquire_gil();
	let context = Context::new_with_gil(gil.python()).unwrap();
	python! {
		#![context = &context]
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])[0:2, 0:2];
	}

	let matrix = context.globals(gil.python()).get_item("matrix").unwrap();

	let matrix : nalgebra::MatrixN<f64, U2> = matrix_from_python(matrix).unwrap();
	assert_eq!(matrix, nalgebra::MatrixN::<f64, U2>::new(
		1.0, 2.0,
		4.0, 5.0,
	));
}

/// Test conversion of a column-major numpy array.
#[test]
fn column_major() {
	let gil = pyo3::Python::acquire_gil();
	let context = Context::new_with_gil(gil.python()).unwrap();
	python! {
		#![context = &context]
		import numpy as np
		matrix = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		], order='F');
	}

	let matrix = context.globals(gil.python()).get_item("matrix").unwrap();

	let matrix : nalgebra::MatrixN<f64, U3> = matrix_from_python(matrix).unwrap();
	assert_eq!(matrix, nalgebra::Matrix3::new(
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	));
}
