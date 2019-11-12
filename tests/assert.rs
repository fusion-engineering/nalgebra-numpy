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
