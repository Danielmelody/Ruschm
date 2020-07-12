use crate::interpreter::*;
use std::collections::HashMap;

pub(crate) fn base_library() -> HashMap<String, ValueType> {
    fn add(arguments: Vec<ValueType>) -> Result<ValueType> {
        arguments
            .iter()
            .try_fold(ValueType::Number(Number::Integer(0)), |a, b| match (a, b) {
                (ValueType::Number(num1), ValueType::Number(num2)) => {
                    Ok(ValueType::Number(num1 + *num2))
                }
                _ => logic_error!("expect a number!"),
            })
    }

    fn sub(arguments: Vec<ValueType>) -> Result<ValueType> {
        let mut iter = arguments.iter();
        let init = match arguments.len() {
            0 => logic_error!("'-' needs at least one argument"),
            1 => ValueType::Number(Number::Integer(0)),
            _ => iter.next().unwrap().clone(),
        };
        iter.try_fold(init, |a, b| match (a, b) {
            (ValueType::Number(num1), ValueType::Number(num2)) => {
                Ok(ValueType::Number(num1 - *num2))
            }
            _ => logic_error!("expect a number!"),
        })
    }
    fn mul(arguments: Vec<ValueType>) -> Result<ValueType> {
        arguments
            .iter()
            .try_fold(ValueType::Number(Number::Integer(1)), |a, b| match (a, b) {
                (ValueType::Number(num1), ValueType::Number(num2)) => {
                    Ok(ValueType::Number(num1 * *num2))
                }
                _ => logic_error!("expect a number!"),
            })
    }

    fn div(arguments: Vec<ValueType>) -> Result<ValueType> {
        let mut iter = arguments.iter();
        let init = match arguments.len() {
            0 => logic_error!("'/' needs at least one argument"),
            1 => ValueType::Number(Number::Integer(1)),
            _ => iter.next().unwrap().clone(),
        };
        iter.try_fold(init, |a, b| match (a, b) {
            (ValueType::Number(num1), ValueType::Number(num2)) => {
                Ok(ValueType::Number((num1 / *num2)?))
            }
            _ => logic_error!("expect a number!"),
        })
    }

    fn max(arguments: Vec<ValueType>) -> Result<ValueType> {
        match arguments.len() {
            0 => logic_error!("max requires at least one argument!"),
            _ => {
                let mut iter = arguments.iter();
                match iter.next() {
                    Some(ValueType::Number(num)) => {
                        iter.try_fold(ValueType::Number(*num), |a, b| match (a, b) {
                            (ValueType::Number(num1), ValueType::Number(num2)) => {
                                Ok(ValueType::Number(upcast_oprands((num1, *num2)).get_max()))
                            }
                            _ => logic_error!("expect a number!"),
                        })
                    }
                    _ => logic_error!("expect a number!"),
                }
            }
        }
    }

    fn min(arguments: Vec<ValueType>) -> Result<ValueType> {
        match arguments.len() {
            0 => logic_error!("min requires at least one argument!"),
            _ => {
                let mut iter = arguments.iter();
                match iter.next() {
                    Some(ValueType::Number(num)) => {
                        iter.try_fold(ValueType::Number(*num), |a, b| match (a, b) {
                            (ValueType::Number(num1), ValueType::Number(num2)) => {
                                Ok(ValueType::Number(upcast_oprands((num1, *num2)).get_min()))
                            }
                            _ => logic_error!("expect a number!"),
                        })
                    }
                    _ => logic_error!("expect a number!"),
                }
            }
        }
    }

    fn sqrt(arguments: Vec<ValueType>) -> Result<ValueType> {
        match arguments.len() {
            1 => Ok(ValueType::Number(Number::Real(
                match arguments.first().unwrap() {
                    ValueType::Number(Number::Integer(num)) => (*num as f64).sqrt(),
                    ValueType::Number(Number::Real(num)) => num.sqrt(),
                    ValueType::Number(Number::Rational(a, b)) => (*a as f64 / *b as f64).sqrt(),
                    other => logic_error!("sqrt requires a number, got {:?}", other),
                },
            ))),
            _ => logic_error!("sqrt takes exactly one argument"),
        }
    }

    [
        (
            "+".to_string(),
            ValueType::Procedure(Procedure::Buildin(add)),
        ),
        (
            "-".to_string(),
            ValueType::Procedure(Procedure::Buildin(sub)),
        ),
        (
            "*".to_string(),
            ValueType::Procedure(Procedure::Buildin(mul)),
        ),
        (
            "/".to_string(),
            ValueType::Procedure(Procedure::Buildin(div)),
        ),
        (
            "max".to_string(),
            ValueType::Procedure(Procedure::Buildin(max)),
        ),
        (
            "min".to_string(),
            ValueType::Procedure(Procedure::Buildin(min)),
        ),
        (
            "sqrt".to_string(),
            ValueType::Procedure(Procedure::Buildin(sqrt)),
        ),
    ]
    .iter()
    .cloned()
    .collect()
}
