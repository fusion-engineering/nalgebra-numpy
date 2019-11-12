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
