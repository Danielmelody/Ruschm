use crate::interpreter::*;
use crate::parser::ParameterFormals;
use crate::values::*;
use crate::{environment::IEnvironment, values::Value};
use std::collections::HashMap;

fn car<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    let mut iter = arguments.into_iter();
    Ok(iter.next().unwrap().expect_list_or_pair()?.car)
}

fn cdr<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    let mut iter = arguments.into_iter();
    Ok(iter.next().unwrap().expect_list_or_pair()?.cdr)
}

fn cons<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    let mut iter = arguments.into_iter();
    match (iter.next(), iter.next()) {
        (Some(car), Some(cdr)) => Ok(Value::Pair(Box::new(Pair { car, cdr }))),
        _ => unreachable!(),
    }
}

macro_rules! value_test {
    ($variant:pat) => {
        |arguments| {
            let arg = arguments.into_iter().next().unwrap();
            match arg {
                $variant => Ok(Value::Boolean(true)),
                _ => Ok(Value::Boolean(false)),
            }
        }
    };
}

fn eqv<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    let mut iter = arguments.into_iter();
    let a = iter.next().unwrap();
    let b = iter.next().unwrap();
    match (&a, &b) {
        (Value::Vector(a), Value::Vector(b)) => Ok(Value::Boolean(a.ptr_eq(b))),
        (Value::Pair(a), Value::Pair(b)) => Ok(Value::Boolean(
            a.as_ref() as *const Pair<R, E> == b.as_ref() as *const Pair<R, E>,
        )),
        (Value::Number(a), Value::Number(b)) => Ok(Value::Boolean(a.exact_eqv(b))),
        _ => Ok(Value::Boolean(a == b)),
    }
}

#[test]
fn equivalance_predicate() {
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(1)),
            Value::Number(Number::Integer(1)),
        ];
        assert_eq!(eqv(arguments), Ok(Value::Boolean(true)));
    }

    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Pair(Box::new(Pair::new(
                Value::Character('a'),
                Value::Character('b'),
            ))),
            Value::Pair(Box::new(Pair::new(
                Value::Character('a'),
                Value::Character('b'),
            ))),
        ];
        assert_eq!(eqv(arguments), Ok(Value::Boolean(false)));
    }

    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Vector(ValueReference::new_immutable(vec![Value::Character('a')])),
            Value::Vector(ValueReference::new_immutable(vec![Value::Character('a')])),
        ];
        assert_eq!(eqv(arguments), Ok(Value::Boolean(false)));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(1)),
            Value::Number(Number::Integer(1)),
        ];
        assert_eq!(eqv(arguments), Ok(Value::Boolean(true)));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(1)),
            Value::Number(Number::Rational(1, 1)),
        ];
        assert_eq!(eqv(arguments), Ok(Value::Boolean(false)));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![Value::EmptyList, Value::EmptyList];
        assert_eq!(eqv(arguments), Ok(Value::Boolean(true)));
    }
}

fn not<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    match arguments.into_iter().next().unwrap() {
        Value::Boolean(false) => Ok(Value::Boolean(true)),
        _ => Ok(Value::Boolean(false)),
    }
}

fn add<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    arguments
        .into_iter()
        .try_fold(Number::Integer(0), |a, b| Ok(a + b.expect_number()?))
        .map(|num| Value::Number(num))
}

#[test]
fn buildin_add() {
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![];
        assert_eq!(add(arguments), Ok(Value::Number(Number::Integer(0))));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![Value::Number(Number::Integer(2))];
        assert_eq!(add(arguments), Ok(Value::Number(Number::Integer(2))));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(3)),
        ];
        assert_eq!(add(arguments), Ok(Value::Number(Number::Integer(5))));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(3)),
            Value::Number(Number::Integer(4)),
        ];
        assert_eq!(add(arguments), Ok(Value::Number(Number::Integer(9))));
    }
}

fn sub<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    let mut iter = arguments.into_iter();
    let first = iter.next().unwrap().expect_number()?;
    let init = match iter.next() {
        Some(value) => first - value.expect_number()?,
        None => Number::Integer(0) - first,
    };
    iter.try_fold(init, |a, b| Ok(a - b.expect_number()?))
        .map(|num| Value::Number(num))
}

#[test]
fn buildin_sub() {
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![Value::Number(Number::Integer(2))];
        assert_eq!(sub(arguments), Ok(Value::Number(Number::Integer(-2))));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(3)),
        ];
        assert_eq!(sub(arguments), Ok(Value::Number(Number::Integer(-1))));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(3)),
            Value::Number(Number::Integer(4)),
        ];
        assert_eq!(sub(arguments), Ok(Value::Number(Number::Integer(-5))));
    }
}

fn mul<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    arguments
        .into_iter()
        .try_fold(Number::Integer(1), |a, b| Ok(a * b.expect_number()?))
        .map(|num| Value::Number(num))
}

#[test]
fn buildin_mul() {
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![];
        assert_eq!(mul(arguments), Ok(Value::Number(Number::Integer(1))));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![Value::Number(Number::Integer(2))];
        assert_eq!(mul(arguments), Ok(Value::Number(Number::Integer(2))));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(3)),
        ];
        assert_eq!(mul(arguments), Ok(Value::Number(Number::Integer(6))));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(3)),
            Value::Number(Number::Integer(4)),
        ];
        assert_eq!(mul(arguments), Ok(Value::Number(Number::Integer(24))));
    }
}

fn div<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    let mut iter = arguments.into_iter();
    let first = iter.next().unwrap().expect_number()?;
    let init = match iter.next() {
        Some(value) => (first / value.expect_number()?)?,
        None => (Number::Integer(1) / first)?,
    };
    iter.try_fold(init, |a, b| Ok((a / b.expect_number()?)?))
        .map(|num| Value::Number(num))
}

#[test]
fn buildin_div() {
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![Value::Number(Number::Integer(2))];
        assert_eq!(div(arguments), Ok(Value::Number(Number::Real(0.5))));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(8)),
        ];
        assert_eq!(div(arguments), Ok(Value::Number(Number::Real(0.25))));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(8)),
            Value::Number(Number::Real(0.125)),
        ];
        assert_eq!(
            div(arguments),
            Ok(Value::<f32, _>::Number(Number::Real(2.)))
        );
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(0)),
        ];
        assert_eq!(
            div(arguments),
            Err(SchemeError {
                location: None,
                category: ErrorType::Logic,
                message: "division by exact zero".to_string(),
            })
        );
    }
}

macro_rules! numeric_one_argument {
    ($name:tt, $func:tt$(, $err_handle:tt)?) => {
        fn $func<R: RealNumberInternalTrait, E: IEnvironment<R>>(
            arguments: impl IntoIterator<Item = Value<R, E>>,
        ) -> Result<Value<R, E>> {
            Ok(Value::Number(arguments.into_iter().next().unwrap().expect_number()?.$func()$($err_handle)?))
        }
    };
}
numeric_one_argument!("sqrt", sqrt);

numeric_one_argument!("floor", floor);

numeric_one_argument!("ceiling", ceiling);

numeric_one_argument!("exact", exact, ?);
#[test]
fn buildin_numeric_one() {
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> =
            vec![Value::Number(Number::Rational(-49, 3))];
        assert_eq!(floor(arguments), Ok(Value::Number(Number::Integer(-17))));
    }
}

macro_rules! numeric_two_arguments {
    ($name:tt, $func:tt$(, $err_handle:tt)?) => {
        fn $func<R: RealNumberInternalTrait, E: IEnvironment<R>>(
            arguments: impl IntoIterator<Item = Value<R, E>>,
        ) -> Result<Value<R, E>> {
            let mut iter = arguments.into_iter();
            let lhs = iter.next().unwrap().expect_number()?;
            let rhs = iter.next().unwrap().expect_number()?;
            Ok(Value::Number(lhs.$func(rhs)$($err_handle)?))
        }
    };
}

numeric_two_arguments!("floor-quotient", floor_quotient, ?);

numeric_two_arguments!("floor-remainder", floor_remainder, ?);
#[test]
fn buildin_numeric_two() {
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(8)),
            Value::Number(Number::Integer(3)),
        ];
        assert_eq!(
            floor_remainder(arguments),
            Ok(Value::Number(Number::Integer(2)))
        );
    }
}
fn vector<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    let vector: Vec<Value<R, E>> = arguments.into_iter().collect();
    Ok(Value::Vector(ValueReference::new_mutable(vector)))
}

fn make_vector<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    let mut iter = arguments.into_iter();
    let k = iter.next().unwrap().expect_integer()?;
    if k < 0 {
        logic_error!("expect a non-negative length");
    }
    let fill = iter.next().unwrap();
    Ok(Value::Vector(ValueReference::new_mutable(vec![
        fill;
        k as usize
    ])))
}

#[test]
fn buildin_make_vector() {
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> =
            vec![Value::Number(Number::Integer(3)), Value::Boolean(true)];
        assert_eq!(
            make_vector(arguments),
            Ok(Value::Vector(ValueReference::new_mutable(vec![
                Value::Boolean(true),
                Value::Boolean(true),
                Value::Boolean(true)
            ])))
        );
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> =
            vec![Value::Number(Number::Integer(0)), Value::Boolean(true)];
        assert_eq!(
            make_vector(arguments),
            Ok(Value::Vector(ValueReference::new_mutable(vec![])))
        );
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> =
            vec![Value::Number(Number::Integer(-1)), Value::Boolean(true)];
        assert_eq!(
            make_vector(arguments),
            Err(SchemeError {
                location: None,
                category: ErrorType::Logic,
                message: "expect a non-negative length".to_string()
            })
        );
    }
}

fn vector_length<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    let vector = arguments.into_iter().next().unwrap().expect_vector()?;
    let len = vector.as_ref().len();
    Ok(Value::Number(Number::Integer(len as i32)))
}

#[test]
fn buildin_vector_length() {
    {
        let vector: Value<f32, StandardEnv<_>> =
            Value::Vector(ValueReference::new_immutable(vec![
                Value::Number(Number::Integer(5)),
                Value::String("foo".to_string()),
                Value::Number(Number::Rational(5, 3)),
            ]));
        let arguments = vec![vector.clone()];
        assert_eq!(
            vector_length(arguments),
            Ok(Value::Number(Number::Integer(3)))
        );
    }
    {
        let vector: Value<f32, StandardEnv<_>> =
            Value::Vector(ValueReference::new_immutable(vec![]));
        let arguments = vec![vector.clone()];
        assert_eq!(
            vector_length(arguments),
            Ok(Value::Number(Number::Integer(0)))
        );
    }
}

fn vector_ref<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    let mut iter = arguments.into_iter();
    let vector = iter.next().unwrap().expect_vector()?;
    let k = iter.next().unwrap().expect_integer()?;
    let r = match vector.as_ref().get(k as usize) {
        Some(value) => Ok(value.clone()),
        None => logic_error!("vector index out of bound"),
    };
    r
}

#[test]
fn buildin_vector_ref() {
    let vector: Value<f32, StandardEnv<_>> = Value::Vector(ValueReference::new_immutable(vec![
        Value::Number(Number::Integer(5)),
        Value::String("foo".to_string()),
        Value::Number(Number::Rational(5, 3)),
    ]));
    {
        let arguments = vec![vector.clone(), Value::Number(Number::Integer(0))];
        assert_eq!(vector_ref(arguments), Ok(Value::Number(Number::Integer(5))));
    }
    {
        let arguments = vec![vector.clone(), Value::Number(Number::Integer(1))];
        assert_eq!(vector_ref(arguments), Ok(Value::String("foo".to_string())));
    }
    {
        let arguments = vec![vector.clone(), Value::Number(Number::Integer(2))];
        assert_eq!(
            vector_ref(arguments),
            Ok(Value::Number(Number::Rational(5, 3)))
        );
    }
    {
        let arguments = vec![vector.clone(), Value::Number(Number::Integer(3))];
        assert_eq!(
            vector_ref(arguments),
            Err(SchemeError {
                location: None,
                category: ErrorType::Logic,
                message: "vector index out of bound".to_string(),
            })
        );
    }
}

fn vector_set<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    let mut iter = arguments.into_iter();
    let vector = iter.next().unwrap().expect_vector()?;
    let k = iter.next().unwrap().expect_integer()?;
    let obj = iter.next().unwrap();
    match vector.as_mut()?.get_mut(k as usize) {
        None => logic_error!("vector index out of bound"),
        Some(value) => {
            *value = obj;
        }
    }
    Ok(Value::Void)
}

#[test]
fn buildin_vector_set() -> Result<()> {
    let vector: Value<f32, StandardEnv<_>> = Value::Vector(ValueReference::new_mutable(vec![
        Value::Number(Number::Integer(5)),
        Value::String("foo".to_string()),
        Value::Number(Number::Rational(5, 3)),
    ]));
    {
        let arguments = vec![
            vector.clone(),
            Value::Number(Number::Integer(0)),
            Value::Number(Number::Real(3.14)),
        ];
        assert_eq!(vector_set(arguments), Ok(Value::Void));
        assert_eq!(
            vector,
            Value::Vector(ValueReference::new_mutable(vec![
                Value::Number(Number::Real(3.14)),
                Value::String("foo".to_string()),
                Value::Number(Number::Rational(5, 3)),
            ]))
        );
    }
    {
        let arguments = vec![
            vector.clone(),
            Value::Number(Number::Integer(1)),
            Value::Number(Number::Integer(5)),
        ];
        assert_eq!(vector_set(arguments), Ok(Value::Void));
        assert_eq!(
            vector,
            Value::Vector(ValueReference::new_mutable(vec![
                Value::Number(Number::Real(3.14)),
                Value::Number(Number::Integer(5)),
                Value::Number(Number::Rational(5, 3)),
            ]))
        );
    }
    {
        let arguments = vec![
            vector.clone(),
            Value::Number(Number::Integer(2)),
            Value::String("bar".to_string()),
        ];
        assert_eq!(vector_set(arguments), Ok(Value::Void));
        assert_eq!(
            vector,
            Value::Vector(ValueReference::new_mutable(vec![
                Value::Number(Number::Real(3.14)),
                Value::Number(Number::Integer(5)),
                Value::String("bar".to_string()),
            ]))
        );
    }
    {
        let arguments = vec![
            vector.clone(),
            Value::Number(Number::Integer(3)),
            Value::Number(Number::Integer(5)),
        ];
        assert_eq!(
            vector_set(arguments),
            Err(SchemeError {
                location: None,
                category: ErrorType::Logic,
                message: "vector index out of bound".to_string(),
            })
        );
    }
    Ok(())
}

fn display<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    print!("{}", arguments.into_iter().next().unwrap());
    Ok(Value::Void)
}

fn newline<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    _: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    println!("");
    Ok(Value::Void)
}

macro_rules! typed_comparision {
    ($name:tt, $operator:tt, $expect_type: tt) => {
        fn $name<R: RealNumberInternalTrait, E: IEnvironment<R>>(
                        arguments: impl IntoIterator<Item = Value<R, E>>
        ) -> Result<Value<R, E>> {
            let mut iter = arguments.into_iter();
            match iter.next() {
                None => Ok(Value::Boolean(true)),
                Some(first) => {
                    let mut last_num = first.$expect_type()?;
                    for current in iter {
                        let current_num = current.$expect_type()?;
                        if !(last_num $operator current_num) {
                            return Ok(Value::Boolean(false));
                        }
                        last_num = current_num;
                    }
                    Ok(Value::Boolean(true))
                }

            }
        }
    }
}

typed_comparision!(equals, ==, expect_number);
typed_comparision!(greater, >, expect_number);
typed_comparision!(greater_equal, >=, expect_number);
typed_comparision!(less, <, expect_number);
typed_comparision!(less_equal, <=, expect_number);
typed_comparision!(boolean_equal, <=, expect_boolean);

#[test]
fn buildin_greater() {
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![];
        assert_eq!(greater(arguments), Ok(Value::Boolean(true)));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![Value::Number(Number::Integer(2))];
        assert_eq!(greater(arguments), Ok(Value::Boolean(true)));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(2)),
        ];
        assert_eq!(greater(arguments), Ok(Value::Boolean(true)));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(8)),
        ];
        assert_eq!(greater(arguments), Ok(Value::Boolean(false)));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(1)),
        ];
        assert_eq!(greater(arguments), Ok(Value::Boolean(true)));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(2)),
        ];
        assert_eq!(greater(arguments), Ok(Value::Boolean(false)));
    }
}

macro_rules! first_of_order {
    ($name:tt, $cmp:tt) => {
        fn $name<R: RealNumberInternalTrait, E: IEnvironment<R>>(
                        arguments: impl IntoIterator<Item = Value<R, E>>
        ) -> Result<Value<R, E>> {
            let mut iter = arguments.into_iter();
            let init = iter.next().unwrap().expect_number()?;
            iter.try_fold(init, |a, b_value| {
                let b = b_value.expect_number()?;
                let oprand = upcast_oprands((a, b));
                Ok(if a $cmp b {oprand.lhs()} else {oprand.rhs()})
            }).map(|num| Value::Number(num))
            }
        }
}

first_of_order!(max, >);
first_of_order!(min, <);

#[test]
fn buildin_min() {
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![Value::Number(Number::Integer(2))];
        assert_eq!(min(arguments), Ok(Value::Number(Number::Integer(2))));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(2)),
        ];
        assert_eq!(min(arguments), Ok(Value::Number(Number::Integer(2))));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(8)),
        ];
        assert_eq!(min(arguments), Ok(Value::Number(Number::Integer(4))));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(1)),
        ];
        assert_eq!(min(arguments), Ok(Value::Number(Number::Integer(1))));
    }
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(2)),
        ];
        assert_eq!(greater(arguments), Ok(Value::Boolean(false)));
    }
}

pub fn base_library<'a, R: RealNumberInternalTrait, E: IEnvironment<R>>(
) -> HashMap<String, Value<R, E>> {
    macro_rules! function_mapping {
        ($ident:tt, $fixed_parameter:expr, $variadic_parameter:expr, $function:expr) => {
            (
                $ident.to_owned(),
                Value::Procedure(Procedure::new_buildin_pure(
                    $ident,
                    ParameterFormals($fixed_parameter, $variadic_parameter),
                    $function,
                )),
            )
        };
    }

    vec![
        function_mapping!("car", vec!["pair".to_string()], None, car),
        function_mapping!("cdr", vec!["pair".to_string()], None, cdr),
        function_mapping!(
            "eqv?",
            vec!["obj1".to_string(), "obj2".to_string()],
            None,
            eqv
        ),
        function_mapping!(
            "eq?", // Ruschm is pass-by value now, so that eq? is equivalent to eqv?
            vec!["obj1".to_string(), "obj2".to_string()],
            None,
            eqv
        ),
        function_mapping!(
            "cons",
            vec!["car".to_string(), "cdr".to_string()],
            None,
            cons
        ),
        function_mapping!(
            "boolean?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::Boolean(_))
        ),
        function_mapping!(
            "char?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::Character(_))
        ),
        function_mapping!(
            "number?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::Number(_))
        ),
        function_mapping!(
            "string?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::String(_))
        ),
        function_mapping!(
            "symbol?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::Symbol(_))
        ),
        function_mapping!(
            "pair?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::Pair(_))
        ),
        function_mapping!(
            "procedure?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::Procedure(_))
        ),
        function_mapping!(
            "vector?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::Vector(_))
        ),
        function_mapping!("not", vec!["obj".to_string()], None, not),
        function_mapping!(
            "boolean=?",
            vec![],
            Some("booleans".to_string()),
            boolean_equal
        ),
        function_mapping!("+", vec![], Some("x".to_string()), add),
        function_mapping!("-", vec!["x1".to_string()], Some("x".to_string()), sub),
        function_mapping!("*", vec![], Some("x".to_string()), mul),
        function_mapping!("/", vec!["x1".to_string()], Some("x".to_string()), div),
        function_mapping!("=", vec![], Some("x".to_string()), equals),
        function_mapping!("<", vec![], Some("x".to_string()), less),
        function_mapping!("<=", vec![], Some("x".to_string()), less_equal),
        function_mapping!(">", vec![], Some("x".to_string()), greater),
        function_mapping!(">=", vec![], Some("x".to_string()), greater_equal),
        function_mapping!("min", vec!["x1".to_string()], Some("x".to_string()), min),
        function_mapping!("max", vec!["x1".to_string()], Some("x".to_string()), max),
        function_mapping!("sqrt", vec!["x".to_string()], None, sqrt),
        function_mapping!("floor", vec!["x".to_string()], None, floor),
        function_mapping!("ceiling", vec!["x".to_string()], None, ceiling),
        function_mapping!("exact", vec!["x".to_string()], None, exact),
        function_mapping!(
            "floor-quotient",
            vec!["n1".to_string(), "n2".to_string()],
            None,
            floor_quotient
        ),
        function_mapping!(
            "floor-remainder",
            vec!["n1".to_string(), "n2".to_string()],
            None,
            floor_remainder
        ),
        function_mapping!("display", vec!["value".to_string()], None, display),
        function_mapping!("newline", vec![], None, newline),
        function_mapping!("vector", vec![], Some("x".to_string()), vector),
        function_mapping!(
            "make-vector",
            vec!["k".to_string(), "obj".to_string()],
            None,
            make_vector
        ),
        function_mapping!(
            "vector-length",
            vec!["vector".to_string()],
            None,
            vector_length
        ),
        function_mapping!(
            "vector-ref",
            vec!["vector".to_string(), "k".to_string()],
            None,
            vector_ref
        ),
        function_mapping!(
            "vector-set!",
            vec!["vector".to_string(), "k".to_string(), "obj".to_string()],
            None,
            vector_set
        ),
    ]
    .into_iter()
    .collect()
}

#[test]
fn buildin_parameters_length() -> Result<()> {
    let buildin_functions = base_library::<f32, StandardEnv<_>>();
    assert!(matches!(
        &buildin_functions["sqrt"],
        Value::Procedure(Procedure::Buildin(sqrt)) if sqrt.parameters.0.len() == 1));
    assert!(matches!(
        &buildin_functions["display"],
        Value::Procedure(Procedure::Buildin(display)) if display.parameters.0.len() == 1));
    assert!(matches!(
        &buildin_functions["newline"],
        Value::Procedure(Procedure::Buildin(newline)) if newline.parameters.0.len() == 0));
    Ok(())
}
