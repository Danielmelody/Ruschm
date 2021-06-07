use ruschm::{
    error::SchemeError,
    interpreter::Interpreter,
    values::{Number, Value},
};

#[test]
fn tco_unbounded_range() -> Result<(), SchemeError> {
    let mut interpreter = Interpreter::<f32>::new_with_stdlib();
    assert_eq!(
        interpreter.eval(include_str!("./tco_testcases/fib-tail.scm").chars())?,
        Some(Value::Number(Number::Integer(196418)))
    );
    Ok(())
}
