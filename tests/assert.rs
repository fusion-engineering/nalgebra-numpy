macro_rules! assert_ok {
	( $expr:expr ) => {{
		let value = $expr;
		::assert2::assert!(let Ok(_) = &value);
		value.unwrap()
	}}
}

#[test]
fn test_assert_ok() {
	let result: Result<(), String> = Ok(());
	assert_ok!(result);
}

#[test]
#[should_panic]
fn test_assert_ok_err() {
	let result: Result<(), String> = Err(String::from("this is an error"));
	assert_ok!(result);
}
