/// Assert that an expression matches a pattern.
///
/// This can be useful to test that a particular enum variant is returned,
/// even if the enum does not implement Eq.
/// It is also useful when you want to test which variant is returned,
/// but you do not care about the contents.
///
/// If the expression does not match the pattern,
/// both the expression and the value it evaluates to are printed.
/// The type of the value must implement [`Debug`].
///
/// ```
/// assert_match!(Err(_) = std::fs::read("/non/existing/path"));
/// assert_match!(Ok(_) = std::fs::read("/dev/null"));
/// ```
macro_rules! assert_match {
	( $pat:pat = $expr:expr) => {{
		let value = $expr;
		if let $pat = &value {
			assert!(true)
		} else {
			eprintln!("failed to match {} = $expr", stringify!($pat));
			eprintln!("  with $expr = {}", stringify!($expr));
			eprintln!("  which evaluates to: {:?}", value);
			panic!("assertion failed");
		}
	}}
}

macro_rules! assert_ok {
	( $expr:expr ) => {{
		let value = $expr;
		match value {
			Ok(x) => x,
			Err(e) => {
				eprintln!("failed to assert that $expr is Ok(...)");
				eprintln!("  with $expr = {}", stringify!($expr));
				eprintln!("  which is an Err(...): {}", e);
				panic!("assertion failed");
			}
		}
	}}
}

#[derive(Debug)]
enum Foo {
	Foo(i32),
	Bar(i32),
}

#[test]
fn test_assert_match() {
	assert_match!(Foo::Foo(10) = Foo::Foo(10));
	assert_match!(Foo::Bar(11) = Foo::Bar(11));
}

#[test]
#[should_panic]
fn test_assert_match_different_variant() {
	assert_match!(Foo::Foo(10) = Foo::Bar(10));
}

#[test]
#[should_panic]
fn test_assert_match_different_value() {
	assert_match!(Foo::Foo(10) = Foo::Foo(11));
}

#[test]
fn test_assert_ok() {
	let result : Result<(), String> = Ok(());
	assert_ok!(result);
}

#[test]
#[should_panic]
fn test_assert_ok_err() {
	let result : Result<(), String> = Err(String::from("this is an error"));
	assert_ok!(result);
}
