use ruschm::{
    error::ToLocated,
    error::{ErrorData, SchemeError},
    interpreter::{error::LogicError, pair::Pair, Interpreter},
    values::Number,
    values::Type,
    values::Value,
};

#[test]
fn list() -> Result<(), SchemeError> {
    let mut interpreter = Interpreter::<f32>::new_with_stdlib();
    assert_eq!(
        interpreter.eval("(list 1 2 3)".chars())?,
        Some(Value::Pair(Box::new(Pair::new(
            Value::Number(Number::Integer(1)),
            Value::Pair(Box::new(Pair::new(
                Value::Number(Number::Integer(2)),
                Value::Pair(Box::new(Pair::new(
                    Value::Number(Number::Integer(3)),
                    Value::EmptyList
                )))
            )))
        ))))
    );
    Ok(())
}

#[test]
fn apply() -> Result<(), SchemeError> {
    let mut interpreter = Interpreter::<f32>::new_with_stdlib();
    assert_eq!(
        interpreter.eval("(apply + 1 2 '(3 4))".chars())?,
        Some(Value::Number(Number::Integer(10)))
    );
    assert_eq!(
        interpreter.eval("(apply + 1)".chars()),
        Err(ErrorData::Logic(LogicError::TypeMisMatch("1".to_owned(), Type::Pair)).no_locate())
    );
    Ok(())
}

#[test]
fn booleans() -> Result<(), SchemeError> {
    {
        let mut interpreter = Interpreter::<f32>::new_with_stdlib();
        assert_eq!(
            interpreter.eval("(not #f)".chars())?,
            Some(Value::Boolean(true))
        );
        assert_eq!(
            interpreter.eval("(not #t)".chars())?,
            Some(Value::Boolean(false))
        );
        assert_eq!(
            interpreter.eval("(not '())".chars())?,
            Some(Value::Boolean(false))
        );
        assert_eq!(
            interpreter.eval("(not 1)".chars())?,
            Some(Value::Boolean(false))
        );
        assert_eq!(
            interpreter.eval("(not 0)".chars())?,
            Some(Value::Boolean(false))
        );
    }

    {
        let mut interpreter = Interpreter::<f32>::new_with_stdlib();
        assert_eq!(
            interpreter.eval("(boolean? #t)".chars())?,
            Some(Value::Boolean(true))
        );
        assert_eq!(
            interpreter.eval("(boolean? #f)".chars())?,
            Some(Value::Boolean(true))
        );
        assert_eq!(
            interpreter.eval("(boolean? '())".chars())?,
            Some(Value::Boolean(false))
        );
        assert_eq!(
            interpreter.eval("(boolean? 1)".chars())?,
            Some(Value::Boolean(false))
        );
        assert_eq!(
            interpreter.eval("(boolean? 0)".chars())?,
            Some(Value::Boolean(false))
        );
    }

    {
        let mut interpreter = Interpreter::<f32>::new_with_stdlib();
        assert_eq!(
            interpreter.eval("(boolean=? #t #t)".chars())?,
            Some(Value::Boolean(true))
        );
        assert_eq!(
            interpreter.eval("(boolean=? #f #f)".chars())?,
            Some(Value::Boolean(true))
        );
        assert_eq!(
            interpreter.eval("(boolean=? #f #t)".chars())?,
            Some(Value::Boolean(false))
        );
        assert_eq!(
            interpreter.eval("(boolean=? 1 1)".chars()),
            Err(
                ErrorData::Logic(LogicError::TypeMisMatch("1".to_string(), Type::Boolean))
                    .no_locate()
            )
        );
    }
    Ok(())
}
