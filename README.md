# nalgebra-numpy

This crate provides conversion between [`nalgebra`] and [`numpy`](https://numpy.org/).

Currently, only the conversion from numpy to nalgebra is implemented,
but the other direction will be added soon.

## Conversion from numpy to nalgebra.

It is possible to create either a view or a copy of a numpy array.
You can use [`matrix_from_python`] to copy the data into a new matrix,
or one of [`matrix_slice_from_python`] or [`matrix_slice_mut_from_python`] to create a view.

Keep in mind though that the borrow checker can not enforce rules on data managed by a Python object.
You could potentially keep an immutable view around in Rust, and then modify the data from Python.
For this reason, creating any view -- even an immutable one -- is unsafe.

## Examples.

Copy a numpy array to a new fixed size matrix:

```rust
#![feature(proc_macro_hygiene)]
use inline_python::{Context, python};
use nalgebra_numpy::{matrix_from_python};

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
let matrix : nalgebra::Matrix3<f64> = matrix_from_python(matrix)?;

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

// <snip>

let matrix : DMatrix<f64> = matrix_from_python(matrix)?;
assert_eq!(matrix, DMatrix::from_row_slice(3, 3, &[
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
]));
```

And so are partially dynamic matrices:

```rust
use nalgebra::{MatrixMN, Dynamic, U3};

// <snip>

let matrix : MatrixMN<f64, U3, Dynamic> = matrix_from_python(matrix)?;
assert_eq!(matrix, MatrixMN::<f64, U3, Dynamic>::from_row_slice(&[
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
]));
```
