#![feature(proc_macro_hygiene)]

use inline_python::{python, Context};
use nalgebra_numpy::{Error, matrix_from_python};
use nalgebra::U3;

#[macro_use]
mod assert;

#[test]
fn matrix3_f64() {
	let gil = pyo3::Python::acquire_gil();
	let py  = gil.python();
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

	assert_eq!(matrix_from_python::<f64, U3, U3>(matrix).unwrap(), nalgebra::Matrix3::new(
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	));
}

#[test]
fn matrix3_f32() {
	let gil = pyo3::Python::acquire_gil();
	let py  = gil.python();
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

	assert_eq!(matrix_from_python::<f32, U3, U3>(matrix).unwrap(), nalgebra::Matrix3::new(
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	));
}
