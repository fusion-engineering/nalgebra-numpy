use inline_python::python;
use nalgebra::{DMatrix, Matrix3};
use nalgebra_numpy::matrix_to_numpy;

#[test]
#[rustfmt::skip]
fn fixed_size() {
	pyo3::Python::with_gil(|py| {
	let matrix = matrix_to_numpy(py, &Matrix3::<i32>::new(
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
})
}

#[test]
#[rustfmt::skip]
fn dynamic_size() {
	pyo3::Python::with_gil(|py| -> () {
	let matrix = matrix_to_numpy(py, &DMatrix::<f64>::from_row_slice(3, 4, &[
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
	}});
}
