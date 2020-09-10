use crate::environment::IEnvironment;
use crate::interpreter::*;
use std::collections::HashMap;

pub fn base_library<'a, R: RealNumberInternalTrait, E: IEnvironment<R>>(
) -> HashMap<String, Value<R, E>> {
    fn add<R: RealNumberInternalTrait, E: IEnvironment<R>>(
        arguments: impl IntoIterator<Item = Value<R, E>>,
    ) -> Result<Value<R, E>> {
        arguments
            .into_iter()
            .try_fold(Value::Number(Number::Integer(0)), |a, b| match (a, b) {
                (Value::Number(num1), Value::Number(num2)) => Ok(Value::Number(num1 + num2)),
                _ => logic_error!("expect a number!"),
            })
    }

    fn sub<R: RealNumberInternalTrait, E: IEnvironment<R>>(
        arguments: impl IntoIterator<Item = Value<R, E>>,
    ) -> Result<Value<R, E>> {
        let mut iter = arguments.into_iter();
        let init = match iter.next() {
            None => logic_error!("'-' needs at least one argument"),
            Some(first) => match first {
                Value::Number(first_num) => match iter.next() {
                    Some(second) => match second {
                        Value::Number(second_num) => Value::Number(first_num - second_num),
                        _ => logic_error!("expect a number!"),
                    },
                    None => Value::Number(Number::Integer(0) - first_num),
                },
                _ => logic_error!("expect a number!"),
            },
        };
        iter.try_fold(init, |a, b| match (a, b) {
            (Value::Number(num1), Value::Number(num2)) => Ok(Value::Number(num1 - num2)),
            _ => logic_error!("expect a number!"),
        })
    }

    fn mul<R: RealNumberInternalTrait, E: IEnvironment<R>>(
        arguments: impl IntoIterator<Item = Value<R, E>>,
    ) -> Result<Value<R, E>> {
        let mut iter = arguments.into_iter();
        iter.try_fold(Value::Number(Number::Integer(1)), |a, b| match (a, b) {
            (Value::Number(num1), Value::Number(num2)) => Ok(Value::Number(num1 * num2)),
            _ => logic_error!("expect a number!"),
        })
    }

    fn div<R: RealNumberInternalTrait, E: IEnvironment<R>>(
        arguments: impl IntoIterator<Item = Value<R, E>>,
    ) -> Result<Value<R, E>> {
        let mut iter = arguments.into_iter();
        let init = match iter.next() {
            None => logic_error!("'-' needs at least one argument"),
            Some(first) => match first {
                Value::Number(first_num) => match iter.next() {
                    Some(second) => match second {
                        Value::Number(second_num) => Value::Number((first_num / second_num)?),
                        _ => logic_error!("expect a number!"),
                    },
                    None => Value::Number((Number::Integer(1) / first_num)?),
                },
                _ => logic_error!("expect a number!"),
            },
        };
        iter.try_fold(init, |a, b| match (a, b) {
            (Value::Number(num1), Value::Number(num2)) => Ok(Value::Number((num1 / num2)?)),
            _ => logic_error!("expect a number!"),
        })
    }

    // fn cond(arguments: impl IntoIterator<Item = Value<R, E>>) -> Result<Value> {}

    macro_rules! comparision {
        ($name:tt, $operator:tt) => {
            fn $name<R: RealNumberInternalTrait, E: IEnvironment<R>>(
                         arguments: impl IntoIterator<Item = Value<R, E>>
            ) -> Result<Value<R, E>> {
                let mut iter = arguments.into_iter();
                match iter.next() {
                    None => Ok(Value::Boolean(true)),
                    Some(first) => {
                                let mut last = first;
                                for current in iter {
                                    match (last, current) {
                                        (Value::Number(a), Value::Number(b)) => {
                                            if !(a $operator b) {
                                                return Ok(Value::Boolean(false));
                                            }
                                            last = Value::Number(b);
                                        }
                                        _ => logic_error!("{} comparision can only between numbers!", stringify!($operator)),
                                    }
                                }
                                Ok(Value::Boolean(true))
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
            fn $name<R: RealNumberInternalTrait, E: IEnvironment<R>>(
                         arguments: impl IntoIterator<Item = Value<R, E>>
            ) -> Result<Value<R, E>> {
                let mut iter = arguments.into_iter();
                match iter.next() {
                    None => logic_error!("min requires at least one argument!"),
                    Some(Value::Number(num)) => {
                        iter.try_fold(Value::Number(num), |a, b| match (a, b) {
                                    (Value::Number(num1), Value::Number(num2)) => {
                                        Ok(Value::Number(match num1 $cmp num2 {
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

    fn sqrt<R: RealNumberInternalTrait, E: IEnvironment<R>>(
        arguments: impl IntoIterator<Item = Value<R, E>>,
    ) -> Result<Value<R, E>> {
        match arguments.into_iter().next() {
            Some(Value::Number(number)) => Ok(Value::Number(match number {
                Number::Integer(num) => Number::Real(R::from(num).unwrap().sqrt()),
                Number::Real(num) => Number::Real(num.sqrt()),
                Number::Rational(a, b) => {
                    Number::Real(R::from(a).unwrap() / R::from(b).unwrap().sqrt())
                }
            })),
            Some(other) => logic_error!("sqrt requires a number, got {:?}", other),
            _ => logic_error!("sqrt takes exactly one argument"),
        }
    }

    fn vector<R: RealNumberInternalTrait, E: IEnvironment<R>>(
        arguments: impl IntoIterator<Item = Value<R, E>>,
    ) -> Result<Value<R, E>> {
        let vector: Vec<Value<R, E>> = arguments.into_iter().collect();
        Ok(Value::Vector(vector))
    }

    fn display<R: RealNumberInternalTrait, E: IEnvironment<R>>(
        arguments: impl IntoIterator<Item = Value<R, E>>,
    ) -> Result<Value<R, E>> {
        Ok(match arguments.into_iter().next() {
            Some(value) => {
                print!("{}", value);
                Value::Void
            }
            None => logic_error!("display takes exactly one argument"),
        })
    }

    fn newline<R: RealNumberInternalTrait, E: IEnvironment<R>>(
        arguments: impl IntoIterator<Item = Value<R, E>>,
    ) -> Result<Value<R, E>> {
        Ok(match arguments.into_iter().next() {
            None => {
                println!("");
                Value::<R, E>::Void
            }
            _ => logic_error!("display takes exactly one argument"),
        })
    }

    macro_rules! function_mapping {
        ($ident:tt, $parameter_length:expr, $function:tt) => {
            (
                $ident.to_string(),
                Value::Procedure(Procedure::new_buildin_pure(
                    $ident,
                    $parameter_length,
                    $function,
                )),
            )
        };
    }

    [
        function_mapping!("+", None, add),
        function_mapping!("-", None, sub),
        function_mapping!("*", None, mul),
        function_mapping!("/", None, div),
        function_mapping!("=", None, equals),
        function_mapping!("<", None, less),
        function_mapping!("<=", None, less_equal),
        function_mapping!(">", None, greater),
        function_mapping!(">=", None, greater_equal),
        function_mapping!("min", None, min),
        function_mapping!("max", None, max),
        function_mapping!("sqrt", Some(1), sqrt),
        function_mapping!("display", Some(1), display),
        function_mapping!("newline", Some(0), newline),
        function_mapping!("vector", None, vector),
    ]
    .iter()
    .cloned()
    .collect()
}

#[test]
fn buildin_parameters_length() -> Result<()> {
    let buildin_functions = base_library::<f32, StandardEnv<_>>();
    assert!(matches!(
        &buildin_functions["sqrt"],
        Value::Procedure(Procedure::Buildin(sqrt)) if sqrt.parameter_length == Some(1)));
    assert!(matches!(
        &buildin_functions["display"],
        Value::Procedure(Procedure::Buildin(display)) if display.parameter_length == Some(1)));
    assert!(matches!(
        &buildin_functions["newline"],
        Value::Procedure(Procedure::Buildin(newline)) if newline.parameter_length == Some(0)));
    Ok(())
}
