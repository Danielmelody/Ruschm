use pair::Pair;

use crate::{environment::*, interpreter::*};
use crate::{error::ErrorData, error::ToLocated, parser::ParameterFormals};
use std::rc::Rc;

use crate::values::*;

fn apply<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
    env: Rc<E>,
) -> Result<Value<R, E>> {
    let mut iter = arguments.into_iter();
    let proc = iter.next().unwrap().expect_procedure()?;
    let mut args = iter.collect::<ArgVec<R, E>>();
    if !args.is_empty() {
        let extended = args.pop().unwrap();

        let extended = match extended {
            Value::Pair(p) => p.into_iter().collect::<Result<ArgVec<R, E>>>()?,
            Value::EmptyList => ArgVec::new(),
            other => return error!(LogicError::TypeMisMatch(other.to_string(), Type::Pair))?,
        };
        args.extend(extended);
    }
    for arg in &args {
        print!("{}", arg)
    }
    println!();
    Interpreter::apply_procedure(&proc, args, &env)
}

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
fn builtin_add() {
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
fn builtin_sub() {
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
fn builtin_mul() {
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
fn builtin_div() {
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
            Err(ErrorData::Logic(LogicError::DivisionByZero).no_locate())
        );
    }
}

macro_rules! numeric_one_argument {
    ($func:tt$(, $err_handle:tt)?) => {
        fn $func<R: RealNumberInternalTrait, E: IEnvironment<R>>(
            arguments: impl IntoIterator<Item = Value<R, E>>
        ) -> Result<Value<R, E>> {
            Ok(Value::Number(arguments.into_iter().next().unwrap().expect_number()?.$func()$($err_handle)?))
        }
    };
}

numeric_one_argument!(sqrt);
numeric_one_argument!(exp);
numeric_one_argument!(ln);
numeric_one_argument!(sin);
numeric_one_argument!(cos);
numeric_one_argument!(tan);
numeric_one_argument!(asin);
numeric_one_argument!(acos);
numeric_one_argument!(atan);
numeric_one_argument!(floor);
numeric_one_argument!(ceiling);
numeric_one_argument!(exact, ?);

#[test]
fn builtin_numeric_one() {
    {
        let arguments: Vec<Value<f32, StandardEnv<_>>> =
            vec![Value::Number(Number::Rational(-49, 3))];
        assert_eq!(floor(arguments), Ok(Value::Number(Number::Integer(-17))));
    }
}

macro_rules! numeric_two_arguments {
    ($func:tt$(, $err_handle:tt)?) => {
        fn $func<R: RealNumberInternalTrait, E: IEnvironment<R>>(
            arguments: impl IntoIterator<Item = Value<R, E>>
        ) -> Result<Value<R, E>> {
            let mut iter = arguments.into_iter();
            let lhs = iter.next().unwrap().expect_number()?;
            let rhs = iter.next().unwrap().expect_number()?;
            Ok(Value::Number(lhs.$func(rhs)$($err_handle)?))
        }
    };
}

numeric_two_arguments!(floor_quotient, ?);
numeric_two_arguments!(floor_remainder, ?);
numeric_two_arguments!(log);
numeric_two_arguments!(atan2);

#[test]
fn builtin_numeric_two() {
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
        return error!(LogicError::NegativeLength);
    }
    let fill = iter.next().unwrap();
    Ok(Value::Vector(ValueReference::new_mutable(vec![
        fill;
        k as usize
    ])))
}

#[test]
fn builtin_make_vector() {
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
        assert_eq!(make_vector(arguments), error!(LogicError::NegativeLength));
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
fn builtin_vector_length() {
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
        None => return error!(LogicError::VectorIndexOutOfBounds),
    };
    r
}

#[test]
fn builtin_vector_ref() {
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
            Err(ErrorData::Logic(LogicError::VectorIndexOutOfBounds).no_locate())
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
        None => return Err(ErrorData::Logic(LogicError::VectorIndexOutOfBounds).no_locate()),
        Some(value) => {
            *value = obj;
        }
    }
    Ok(Value::Void)
}

#[test]
fn builtin_vector_set() -> Result<()> {
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
            error!(LogicError::VectorIndexOutOfBounds)
        );
    }
    Ok(())
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
                        arguments: impl IntoIterator<Item = Value<R, E>>,

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
typed_comparision!(boolean_equal, ==, expect_boolean);

#[test]
fn builtin_greater() {
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
                        arguments: impl IntoIterator<Item = Value<R, E>>,

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
fn builtin_min() {
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
pub fn library_map<R: RealNumberInternalTrait, E: IEnvironment<R>>() -> Vec<(String, Value<R, E>)> {
    vec![
        function_mapping!(
            "apply",
            vec!["proc".to_string()],
            Some("args".to_string()),
            apply
        ),
        pure_function_mapping!("car", vec!["pair".to_string()], None, car),
        pure_function_mapping!("cdr", vec!["pair".to_string()], None, cdr),
        pure_function_mapping!(
            "eqv?",
            vec!["obj1".to_string(), "obj2".to_string()],
            None,
            eqv
        ),
        pure_function_mapping!(
            "eq?", // Ruschm is pass-by value now, so that eq? is equivalent to eqv?
            vec!["obj1".to_string(), "obj2".to_string()],
            None,
            eqv
        ),
        pure_function_mapping!(
            "cons",
            vec!["car".to_string(), "cdr".to_string()],
            None,
            cons
        ),
        pure_function_mapping!(
            "boolean?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::Boolean(_))
        ),
        pure_function_mapping!(
            "char?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::Character(_))
        ),
        pure_function_mapping!(
            "number?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::Number(_))
        ),
        pure_function_mapping!(
            "string?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::String(_))
        ),
        pure_function_mapping!(
            "symbol?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::Symbol(_))
        ),
        pure_function_mapping!(
            "pair?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::Pair(_))
        ),
        pure_function_mapping!(
            "procedure?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::Procedure(_))
        ),
        pure_function_mapping!(
            "vector?",
            vec!["obj".to_string()],
            None,
            value_test!(Value::Vector(_))
        ),
        pure_function_mapping!("not", vec!["obj".to_string()], None, not),
        pure_function_mapping!(
            "boolean=?",
            vec![],
            Some("booleans".to_string()),
            boolean_equal
        ),
        pure_function_mapping!("+", vec![], Some("x".to_string()), add),
        pure_function_mapping!("-", vec!["x1".to_string()], Some("x".to_string()), sub),
        pure_function_mapping!("*", vec![], Some("x".to_string()), mul),
        pure_function_mapping!("/", vec!["x1".to_string()], Some("x".to_string()), div),
        pure_function_mapping!("=", vec![], Some("x".to_string()), equals),
        pure_function_mapping!("<", vec![], Some("x".to_string()), less),
        pure_function_mapping!("<=", vec![], Some("x".to_string()), less_equal),
        pure_function_mapping!(">", vec![], Some("x".to_string()), greater),
        pure_function_mapping!(">=", vec![], Some("x".to_string()), greater_equal),
        pure_function_mapping!("min", vec!["x1".to_string()], Some("x".to_string()), min),
        pure_function_mapping!("max", vec!["x1".to_string()], Some("x".to_string()), max),
        pure_function_mapping!("sqrt", vec!["x".to_string()], None, sqrt),
        pure_function_mapping!("exp", vec!["z".to_string()], None, exp),
        pure_function_mapping!("ln", vec!["z".to_string()], None, ln),
        pure_function_mapping!("log", vec!["z1".to_string(), "z2".to_string()], None, log),
        pure_function_mapping!("sin", vec!["z".to_string()], None, sin),
        pure_function_mapping!("cos", vec!["z".to_string()], None, cos),
        pure_function_mapping!("tan", vec!["z".to_string()], None, tan),
        pure_function_mapping!("asin", vec!["z".to_string()], None, asin),
        pure_function_mapping!("acos", vec!["z".to_string()], None, acos),
        pure_function_mapping!("atan", vec!["z".to_string()], None, atan),
        pure_function_mapping!("atan2", vec!["y".to_string(), "x".to_string()], None, atan2),
        pure_function_mapping!("floor", vec!["x".to_string()], None, floor),
        pure_function_mapping!("ceiling", vec!["x".to_string()], None, ceiling),
        pure_function_mapping!("exact", vec!["x".to_string()], None, exact),
        pure_function_mapping!(
            "floor-quotient",
            vec!["n1".to_string(), "n2".to_string()],
            None,
            floor_quotient
        ),
        pure_function_mapping!(
            "floor-remainder",
            vec!["n1".to_string(), "n2".to_string()],
            None,
            floor_remainder
        ),
        //pure_function_mapping!("display", vec!["value".to_string()], None, display),
        pure_function_mapping!("newline", vec![], None, newline),
        pure_function_mapping!("vector", vec![], Some("x".to_string()), vector),
        pure_function_mapping!(
            "make-vector",
            vec!["k".to_string(), "obj".to_string()],
            None,
            make_vector
        ),
        pure_function_mapping!(
            "vector-length",
            vec!["vector".to_string()],
            None,
            vector_length
        ),
        pure_function_mapping!(
            "vector-ref",
            vec!["vector".to_string(), "k".to_string()],
            None,
            vector_ref
        ),
        pure_function_mapping!(
            "vector-set!",
            vec!["vector".to_string(), "k".to_string(), "obj".to_string()],
            None,
            vector_set
        ),
    ]
}

#[test]
fn builtin_parameters_length() -> Result<()> {
    let base_library_map = library_map::<f32, StandardEnv<_>>();
    assert!(matches!(
        &base_library_map.iter().find(|(name, _) |name.as_str() == "sqrt").unwrap().1,
        Value::Procedure(Procedure::Builtin(sqrt)) if sqrt.parameters.0.len() == 1));
    assert!(matches!(
        &base_library_map.iter().find(|(name, _) |name.as_str() == "newline").unwrap().1,
        Value::Procedure(Procedure::Builtin(newline)) if newline.parameters.0.len() == 0));
    Ok(())
}
