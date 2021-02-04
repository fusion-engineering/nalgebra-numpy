# nalgebra-numpy

This crate provides conversion between [`nalgebra`] and [`numpy`](https://numpy.org/).
It is intended to be used when you want to share nalgebra matrices between Python and Rust code,
for example with [`inline-python`](https://docs.rs/inline-python).

## Conversion from numpy to nalgebra.

It is possible to create either a view or a copy of a numpy array.
You can use [`matrix_from_numpy`] to copy the data into a new matrix,
or one of [`matrix_slice_from_numpy`] or [`matrix_slice_mut_from_numpy`] to create a view.
If a numpy array is not compatible with the requested matrix type,
an error is returned.

Keep in mind though that the borrow checker can not enforce rules on data managed by a Python object.
You could potentially keep an immutable view around in Rust, and then modify the data from Python.
For this reason, creating any view -- even an immutable one -- is unsafe.

## Conversion from nalgebra to numpy.

A nalgebra matrix can also be converted to a numpy array, using [`matrix_to_numpy`].
This function always creates a copy.
Since all nalgebra arrays can be represented as a numpy array,
this directly returns a [`pyo3::PyObject`] rather than a `Result`.

## Examples.

Copy a numpy array to a new fixed size matrix:

```rust
use inline_python::{Context, python};
use nalgebra_numpy::{matrix_from_numpy};

let gil = pyo3::Python::acquire_gil();
let context = Context::new_with_gil(gil.python());
context.run(python! {
    import numpy as np
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
});

let matrix = context.globals(gil.python()).get_item("matrix").unwrap();
let matrix : nalgebra::Matrix3<f64> = matrix_from_numpy(gil.python(), matrix)?;

assert_eq!(matrix, nalgebra::Matrix3::new(
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
));
```

Dynamic matrices are also supported:

```rust
use nalgebra::DMatrix;
#

let matrix : DMatrix<f64> = matrix_from_numpy(gil.python(), matrix)?;
assert_eq!(matrix, DMatrix::from_row_slice(3, 3, &[
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
]));
```

And so are partially dynamic matrices:

```rust
use nalgebra::{MatrixMN, Dynamic, U3};

let matrix : MatrixMN<f64, U3, Dynamic> = matrix_from_numpy(gil.python(), matrix)?;
assert_eq!(matrix, MatrixMN::<f64, U3, Dynamic>::from_row_slice(&[
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
]));
```

A conversion to python object looks as follows:
```rust
use nalgebra_numpy::matrix_to_numpy;
use nalgebra::Matrix3;
use inline_python::python;

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
```

[`nalgebra`]: https://docs.rs/nalgebra
[`matrix_from_numpy`]: https://docs.rs/nalgebra-numpy/latest/nalgebra_numpy/fn.matrix_from_numpy.html
[`matrix_slice_from_numpy`]: https://docs.rs/nalgebra-numpy/latest/nalgebra_numpy/fn.matrix_slice_from_numpy.html
[`matrix_slice_mut_from_numpy`]: https://docs.rs/nalgebra-numpy/latest/nalgebra_numpy/fn.matrix_slice_mut_from_numpy.html
[`matrix_to_numpy`]: https://docs.rs/nalgebra-numpy/latest/nalgebra_numpy/fn.matrix_to_numpy.html
[`pyo3::PyObject`]: https://docs.rs/pyo3/latest/pyo3/type.PyObject.html
