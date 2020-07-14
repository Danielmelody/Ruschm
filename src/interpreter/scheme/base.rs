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

    fn equals(arguments: Vec<ValueType>) -> Result<ValueType> {
        if arguments.is_empty() {
            return Ok(ValueType::Boolean(true));
        }
        let first = &arguments[0];
        Ok(ValueType::Boolean(
            arguments.iter().all(|item| item == first),
        ))
    }

    macro_rules! comparision {
        ($name:tt, $operator:tt) => {
            fn $name(arguments: Vec<ValueType>) -> Result<ValueType> {
                match arguments.len() {
                    0 | 1 => Ok(ValueType::Boolean(true)),
                    _ => {
                        let mut iter = arguments.iter();
                        let mut last = iter.next().unwrap();
                        for current in iter {
                            match (last, current) {
                                (ValueType::Number(a), ValueType::Number(b)) => {
                                    if !(a $operator b) {
                                        return Ok(ValueType::Boolean(false));
                                    }
                                    last = current;
                                }
                                _ => logic_error!("great comparision can only between numbers!"),
                            }
                        }
                        Ok(ValueType::Boolean(true))
                    }
                }
            }
        };
    }

    comparision!(greater, >);
    comparision!(greater_equal, >=);
    comparision!(less, <);
    comparision!(less_equal, <=);

    macro_rules! first_of_order {
        ($name:tt, $cmp:tt) => {
            fn $name(arguments: Vec<ValueType>) -> Result<ValueType> {
                match arguments.len() {
                    0 => logic_error!("min requires at least one argument!"),
                    _ => {
                        let mut iter = arguments.iter();
                        match iter.next() {
                            Some(ValueType::Number(num)) => {
                                iter.try_fold(ValueType::Number(*num), |a, b| match (a, b) {
                                    (ValueType::Number(num1), ValueType::Number(num2)) => {
                                        Ok(ValueType::Number(match num1 $cmp *num2 {
                                            true => upcast_oprands((num1, *num2)).lhs(),
                                            false => upcast_oprands((num1, *num2)).rhs(),
                                        }))
                                    }
                                    _ => logic_error!("expect a number!"),
                                })
                            }
                            _ => logic_error!("expect a number!"),
                        }
                    }
                }
            }
        }
    }

    first_of_order!(max, >);
    first_of_order!(min, <);

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

    fn display(arguments: Vec<ValueType>) -> Result<ValueType> {
        Ok(match arguments.len() {
            1 => {
                print!("{}", arguments.first().unwrap());
                ValueType::Void
            }
            _ => logic_error!("display takes exactly one argument"),
        })
    }

    fn newline(arguments: Vec<ValueType>) -> Result<ValueType> {
        Ok(match arguments.len() {
            0 => {
                println!("");
                ValueType::Void
            }
            _ => logic_error!("display takes exactly one argument"),
        })
    }

    macro_rules! function_mapping {
        ($ident:tt, $function:tt) => {
            (
                $ident.to_string(),
                ValueType::Procedure(Procedure::Buildin($function)),
            )
        };
    }

    [
        function_mapping!("+", add),
        function_mapping!("-", sub),
        function_mapping!("*", mul),
        function_mapping!("/", div),
        function_mapping!("=", equals),
        function_mapping!("<", less),
        function_mapping!("<=", less_equal),
        function_mapping!(">", greater),
        function_mapping!(">=", greater_equal),
        function_mapping!("min", min),
        function_mapping!("max", max),
        function_mapping!("sqrt", sqrt),
        function_mapping!("display", display),
        function_mapping!("newline", newline),
    ]
    .iter()
    .cloned()
    .collect()
}
