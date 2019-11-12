#![feature(proc_macro_hygiene)]

use inline_python::{python, Context};
use nalgebra_numpy::{Error, matrix_from_python};
use nalgebra::{U1, U2, U3};

#[macro_use]
mod assert;

#[test]
fn wrong_type() {
	let gil = pyo3::Python::acquire_gil();
	let py  = gil.python();
	let context = Context::new_with_gil(py).unwrap();

	python! {
		#![context = &context]
		float = 3.4
		int   = 8
		list  = [1.0, 2.0, 3.0]
	}

	let get_global = |name| context.globals(py).get_item(name).unwrap();

	assert_match!(Err(Error::WrongObjectType) = matrix_from_python::<f64, U1, U1>(get_global("float")));
	assert_match!(Err(Error::WrongObjectType) = matrix_from_python::<i32, U1, U1>(get_global("int")));
	assert_match!(Err(Error::WrongObjectType) = matrix_from_python::<f64, U1, U3>(get_global("list")));
}

#[test]
fn wrong_shape() {
	let gil = pyo3::Python::acquire_gil();
	let py  = gil.python();
	let context = Context::new_with_gil(py).unwrap();

	python! {
		#![context = &context]
		import numpy as np
		matrix  = np.array([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
		]);
	}

	let get_global = |name| context.globals(py).get_item(name).unwrap();

	assert_match!(Ok(_) = matrix_from_python::<f64, U2, U3>(get_global("matrix")));
	assert_match!(Err(Error::WrongShape) = matrix_from_python::<f64, U1, U1>(get_global("matrix")));
	assert_match!(Err(Error::WrongShape) = matrix_from_python::<f64, U3, U2>(get_global("matrix")));
}

#[test]
fn wrong_data_type() {
	let gil = pyo3::Python::acquire_gil();
	let py  = gil.python();
	let context = Context::new_with_gil(py).unwrap();

	python! {
		#![context = &context]
		import numpy as np
		matrix_f32 = np.array([[1.0]]).astype(np.float32);
		matrix_f64 = np.array([[1.0]]).astype(np.float64);
		matrix_i32 = np.array([[1]]).astype(np.int32);
		matrix_i64 = np.array([[1]]).astype(np.int64);
	}

	let get_global = |name| context.globals(py).get_item(name).unwrap();

	assert_match!(Ok(_) = matrix_from_python::<f32, U1, U1>(get_global("matrix_f32")));
	assert_match!(Ok(_) = matrix_from_python::<f64, U1, U1>(get_global("matrix_f64")));
	assert_match!(Ok(_) = matrix_from_python::<i32, U1, U1>(get_global("matrix_i32")));
	assert_match!(Ok(_) = matrix_from_python::<i64, U1, U1>(get_global("matrix_i64")));

	assert_match!(Err(Error::WrongDataType) = matrix_from_python::<f32, U1, U1>(get_global("matrix_f64")));
	assert_match!(Err(Error::WrongDataType) = matrix_from_python::<f32, U1, U1>(get_global("matrix_i32")));
	assert_match!(Err(Error::WrongDataType) = matrix_from_python::<f32, U1, U1>(get_global("matrix_i64")));

	assert_match!(Err(Error::WrongDataType) = matrix_from_python::<f64, U1, U1>(get_global("matrix_f32")));
	assert_match!(Err(Error::WrongDataType) = matrix_from_python::<f64, U1, U1>(get_global("matrix_i32")));
	assert_match!(Err(Error::WrongDataType) = matrix_from_python::<f64, U1, U1>(get_global("matrix_i64")));

	assert_match!(Err(Error::WrongDataType) = matrix_from_python::<i32, U1, U1>(get_global("matrix_f32")));
	assert_match!(Err(Error::WrongDataType) = matrix_from_python::<i32, U1, U1>(get_global("matrix_f64")));
	assert_match!(Err(Error::WrongDataType) = matrix_from_python::<i32, U1, U1>(get_global("matrix_i64")));

	assert_match!(Err(Error::WrongDataType) = matrix_from_python::<i64, U1, U1>(get_global("matrix_f32")));
	assert_match!(Err(Error::WrongDataType) = matrix_from_python::<i64, U1, U1>(get_global("matrix_f64")));
	assert_match!(Err(Error::WrongDataType) = matrix_from_python::<i64, U1, U1>(get_global("matrix_i32")));
}
