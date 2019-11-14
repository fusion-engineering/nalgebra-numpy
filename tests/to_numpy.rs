#![feature(proc_macro_hygiene)]

use nalgebra_numpy::matrix_to_numpy;
use nalgebra::{DMatrix, Matrix3};
use inline_python::python;

#[test]
fn fixed_size() {
	let gil = pyo3::Python::acquire_gil();

	let matrix = matrix_to_numpy(gil.python(), &Matrix3::<i32>::new(
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
	));

	python! {
		from numpy import array_equal
		assert array_equal('matrix, [
			[0, 1, 2],
			[3, 4, 5],
			[6, 7, 8],
		])
	}
}

#[test]
fn dynamic_size() {
	let gil = pyo3::Python::acquire_gil();

	let matrix = matrix_to_numpy(gil.python(), &DMatrix::<f64>::from_row_slice(3, 4, &[
		0.0, 1.0,  2.0,  3.0,
		4.0, 5.0,  6.0,  7.0,
		8.0, 9.0, 10.0, 11.0,
	]));

	python! {
		from numpy import array_equal
		assert array_equal('matrix, [
			[0, 1,  2,  3],
			[4, 5,  6,  7],
			[8, 9, 10, 11],
		])
	}
}
