use ruschm::{
    error::SchemeError,
    interpreter::Interpreter,
    values::{Number, Value},
};

#[test]
fn match_list() -> Result<(), SchemeError> {
    let mut interpreter = Interpreter::<f32>::new_with_stdlib();
    assert_eq!(
        interpreter.eval(include_str!("./test_macros/macro_list.scm").chars())?,
        Some(Value::Number(Number::Integer(3)))
    );
    Ok(())
}
