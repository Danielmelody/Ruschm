use crate::interpreter::*;
use std::collections::HashMap;

pub(crate) fn base_library<'a>() -> HashMap<String, ValueType> {
    fn add(arguments: impl IntoIterator<Item = ValueType>) -> Result<ValueType> {
        arguments
            .into_iter()
            .try_fold(ValueType::Number(Number::Integer(0)), |a, b| match (a, b) {
                (ValueType::Number(num1), ValueType::Number(num2)) => {
                    Ok(ValueType::Number(num1 + num2))
                }
                _ => logic_error!("expect a number!"),
            })
    }

    fn sub(arguments: impl IntoIterator<Item = ValueType>) -> Result<ValueType> {
        let mut iter = arguments.into_iter();
        let init = match iter.next() {
            None => logic_error!("'-' needs at least one argument"),
            Some(first) => match first {
                ValueType::Number(first_num) => match iter.next() {
                    Some(second) => match second {
                        ValueType::Number(second_num) => ValueType::Number(first_num - second_num),
                        _ => logic_error!("expect a number!"),
                    },
                    None => ValueType::Number(Number::Integer(0) - first_num),
                },
                _ => logic_error!("expect a number!"),
            },
        };
        iter.try_fold(init, |a, b| match (a, b) {
            (ValueType::Number(num1), ValueType::Number(num2)) => {
                Ok(ValueType::Number(num1 - num2))
            }
            _ => logic_error!("expect a number!"),
        })
    }

    fn mul(arguments: impl IntoIterator<Item = ValueType>) -> Result<ValueType> {
        let mut iter = arguments.into_iter();
        iter.try_fold(ValueType::Number(Number::Integer(1)), |a, b| match (a, b) {
            (ValueType::Number(num1), ValueType::Number(num2)) => {
                Ok(ValueType::Number(num1 * num2))
            }
            _ => logic_error!("expect a number!"),
        })
    }

    fn div(arguments: impl IntoIterator<Item = ValueType>) -> Result<ValueType> {
        let mut iter = arguments.into_iter();
        let init = match iter.next() {
            None => logic_error!("'-' needs at least one argument"),
            Some(first) => match first {
                ValueType::Number(first_num) => match iter.next() {
                    Some(second) => match second {
                        ValueType::Number(second_num) => {
                            ValueType::Number((first_num / second_num)?)
                        }
                        _ => logic_error!("expect a number!"),
                    },
                    None => ValueType::Number((Number::Integer(1) / first_num)?),
                },
                _ => logic_error!("expect a number!"),
            },
        };
        iter.try_fold(init, |a, b| match (a, b) {
            (ValueType::Number(num1), ValueType::Number(num2)) => {
                Ok(ValueType::Number((num1 / num2)?))
            }
            _ => logic_error!("expect a number!"),
        })
    }

    // fn cond(arguments: impl IntoIterator<Item = ValueType>) -> Result<ValueType> {}

    macro_rules! comparision {
        ($name:tt, $operator:tt) => {
            fn $name(arguments: impl IntoIterator<Item = ValueType>) -> Result<ValueType> {
                let mut iter = arguments.into_iter();
                match iter.next() {
                    None => Ok(ValueType::Boolean(true)),
                    Some(first) => {
                                let mut last = first;
                                for current in iter {
                                    match (last, current) {
                                        (ValueType::Number(a), ValueType::Number(b)) => {
                                            if !(a $operator b) {
                                                return Ok(ValueType::Boolean(false));
                                            }
                                            last = ValueType::Number(b);
                                        }
                                        _ => logic_error!("{} comparision can only between numbers!", stringify!($operator)),
                                    }
                                }
                                Ok(ValueType::Boolean(true))
                            }

                }
            }
        }
    }

    comparision!(equals, ==);
    comparision!(greater, >);
    comparision!(greater_equal, >=);
    comparision!(less, <);
    comparision!(less_equal, <=);

    macro_rules! first_of_order {
        ($name:tt, $cmp:tt) => {
            fn $name(arguments: impl IntoIterator<Item = ValueType>) -> Result<ValueType> {
                let mut iter = arguments.into_iter();
                match iter.next() {
                    None => logic_error!("min requires at least one argument!"),
                    Some(ValueType::Number(num)) => {
                        iter.try_fold(ValueType::Number(num), |a, b| match (a, b) {
                                    (ValueType::Number(num1), ValueType::Number(num2)) => {
                                        Ok(ValueType::Number(match num1 $cmp num2 {
                                            true => upcast_oprands((num1, num2)).lhs(),
                                            false => upcast_oprands((num1, num2)).rhs(),
                                        }))
                                    }
                                    _ => logic_error!("expect a number!"),
                                })
                            },
                    _ => logic_error!("expect a number!"),
                    }
                }
            }
    }

    first_of_order!(max, >);
    first_of_order!(min, <);

    fn sqrt(arguments: impl IntoIterator<Item = ValueType>) -> Result<ValueType> {
        match arguments.into_iter().next() {
            Some(ValueType::Number(number)) => Ok(ValueType::Number(match number {
                Number::Integer(num) => Number::Real((num as f64).sqrt()),
                Number::Real(num) => Number::Real(num.sqrt()),
                Number::Rational(a, b) => Number::Real((a as f64 / b as f64).sqrt()),
            })),
            Some(other) => logic_error!("sqrt requires a number, got {:?}", other),
            _ => logic_error!("sqrt takes exactly one argument"),
        }
    }

    fn vector(arguments: impl IntoIterator<Item = ValueType>) -> Result<ValueType> {
        let vector: Vec<ValueType> = arguments.into_iter().collect();
        Ok(ValueType::Vector(vector))
    }

    fn display(arguments: impl IntoIterator<Item = ValueType>) -> Result<ValueType> {
        Ok(match arguments.into_iter().next() {
            Some(value) => {
                print!("{}", value);
                ValueType::Void
            }
            None => logic_error!("display takes exactly one argument"),
        })
    }

    fn newline(arguments: impl IntoIterator<Item = ValueType>) -> Result<ValueType> {
        Ok(match arguments.into_iter().next() {
            None => {
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
                ValueType::Procedure(Procedure::Buildin(BuildinProcedure($ident, $function))),
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
        function_mapping!("vector", vector),
    ]
    .iter()
    .cloned()
    .collect()
}
