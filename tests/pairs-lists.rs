use ruschm::{environment::StandardEnv, error::SchemeError, interpreter::*};

#[test]
fn pairs() -> Result<(), SchemeError> {
    let source = include_str!("pairs.scm");
    let it = Interpreter::<f32, StandardEnv<_>>::new();
    it.eval(source.chars())?;
    Ok(())
}
