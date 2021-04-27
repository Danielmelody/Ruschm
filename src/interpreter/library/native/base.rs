use crate::parser::pair::GenericPair;
use crate::parser::*;
use crate::values::*;
use crate::{environment::*, interpreter::*};
use crate::{error::ErrorData, error::ToLocated};
use std::rc::Rc;

fn apply<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
    env: Rc<Environment<R>>,
) -> Result<Value<R>> {
    let mut iter = arguments.into_iter();
    let proc = iter.next().unwrap().expect_procedure()?;
    let mut args = iter.collect::<ArgVec<R>>();
    if !args.is_empty() {
        let extended = args.pop().unwrap();

        let extended = match extended {
            Value::Pair(p) => p.into_iter().collect::<ArgVec<R>>(),
            other => return error!(LogicError::TypeMisMatch(other.to_string(), Type::Pair))?,
        };
        args.extend(extended);
    }
    Interpreter::apply_procedure(&proc, args, &env)
}

fn car<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
) -> Result<Value<R>> {
    let mut iter = arguments.into_iter();
    match iter.next().unwrap().expect_list()? {
        Pair::Some(car, _) => Ok(car),
        empty => return error!(LogicError::TypeMisMatch(empty.to_string(), Type::Pair)),
    }
}

fn cdr<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
) -> Result<Value<R>> {
    let mut iter = arguments.into_iter();
    match iter.next().unwrap().expect_list()? {
        Pair::Some(_, cdr) => Ok(cdr),
        empty => return error!(LogicError::TypeMisMatch(empty.to_string(), Type::Pair)),
    }
}

fn cons<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
) -> Result<Value<R>> {
    let mut iter = arguments.into_iter();
    let car = iter.next().unwrap();
    let cdr = iter.next().unwrap();
    Ok(Value::Pair(Box::new(Pair::Some(car, cdr))))
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

fn is_pair<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
) -> Result<Value<R>> {
    let arg = arguments.into_iter().next().unwrap();
    match arg {
        Value::Pair(p) if !matches!(p.as_ref(), GenericPair::Empty) => Ok(Value::Boolean(true)),
        _ => Ok(Value::Boolean(false)),
    }
}

fn eqv<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
) -> Result<Value<R>> {
    let mut iter = arguments.into_iter();
    let a = iter.next().unwrap();
    let b = iter.next().unwrap();
    match (&a, &b) {
        (Value::Vector(a), Value::Vector(b)) => Ok(Value::Boolean(a.ptr_eq(b))),
        (Value::Pair(a), Value::Pair(b)) => Ok(Value::Boolean(match (a.as_ref(), b.as_ref()) {
            (GenericPair::Empty, GenericPair::Empty) => true,
            _ => a.as_ref() as *const Pair<R> == b.as_ref() as *const Pair<R>,
        })),
        (Value::Number(a), Value::Number(b)) => Ok(Value::Boolean(a.exact_eqv(b))),
        _ => Ok(Value::Boolean(a == b)),
    }
}

#[test]
fn equivalance_predicate() {
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(1)),
            Value::Number(Number::Integer(1)),
        ];
        assert_eq!(eqv(arguments), Ok(Value::Boolean(true)));
    }

    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Pair(Box::new(Pair::cons(
                Value::Character('a'),
                Value::Character('b'),
            ))),
            Value::Pair(Box::new(Pair::Some(
                Value::Character('a'),
                Value::Character('b'),
            ))),
        ];
        assert_eq!(eqv(arguments), Ok(Value::Boolean(false)));
    }

    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Vector(ValueReference::new_immutable(vec![Value::Character('a')])),
            Value::Vector(ValueReference::new_immutable(vec![Value::Character('a')])),
        ];
        assert_eq!(eqv(arguments), Ok(Value::Boolean(false)));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(1)),
            Value::Number(Number::Integer(1)),
        ];
        assert_eq!(eqv(arguments), Ok(Value::Boolean(true)));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(1)),
            Value::Number(Number::Rational(1, 1)),
        ];
        assert_eq!(eqv(arguments), Ok(Value::Boolean(false)));
    }
    {
        let arguments: Vec<Value<f32>> = vec![Value::from(Pair::Empty), Value::from(Pair::Empty)];
        assert_eq!(eqv(arguments), Ok(Value::Boolean(true)));
    }
}

fn not<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
) -> Result<Value<R>> {
    match arguments.into_iter().next().unwrap() {
        Value::Boolean(false) => Ok(Value::Boolean(true)),
        _ => Ok(Value::Boolean(false)),
    }
}

fn add<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
) -> Result<Value<R>> {
    arguments
        .into_iter()
        .try_fold(Number::Integer(0), |a, b| Ok(a + b.expect_number()?))
        .map(|num| Value::Number(num))
}

#[test]
fn builtin_add() {
    {
        let arguments: Vec<Value<f32>> = vec![];
        assert_eq!(add(arguments), Ok(Value::Number(Number::Integer(0))));
    }
    {
        let arguments: Vec<Value<f32>> = vec![Value::Number(Number::Integer(2))];
        assert_eq!(add(arguments), Ok(Value::Number(Number::Integer(2))));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(3)),
        ];
        assert_eq!(add(arguments), Ok(Value::Number(Number::Integer(5))));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(3)),
            Value::Number(Number::Integer(4)),
        ];
        assert_eq!(add(arguments), Ok(Value::Number(Number::Integer(9))));
    }
}

fn sub<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
) -> Result<Value<R>> {
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
        let arguments: Vec<Value<f32>> = vec![Value::Number(Number::Integer(2))];
        assert_eq!(sub(arguments), Ok(Value::Number(Number::Integer(-2))));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(3)),
        ];
        assert_eq!(sub(arguments), Ok(Value::Number(Number::Integer(-1))));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(3)),
            Value::Number(Number::Integer(4)),
        ];
        assert_eq!(sub(arguments), Ok(Value::Number(Number::Integer(-5))));
    }
}

fn mul<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
) -> Result<Value<R>> {
    arguments
        .into_iter()
        .try_fold(Number::Integer(1), |a, b| Ok(a * b.expect_number()?))
        .map(|num| Value::Number(num))
}

#[test]
fn builtin_mul() {
    {
        let arguments: Vec<Value<f32>> = vec![];
        assert_eq!(mul(arguments), Ok(Value::Number(Number::Integer(1))));
    }
    {
        let arguments: Vec<Value<f32>> = vec![Value::Number(Number::Integer(2))];
        assert_eq!(mul(arguments), Ok(Value::Number(Number::Integer(2))));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(3)),
        ];
        assert_eq!(mul(arguments), Ok(Value::Number(Number::Integer(6))));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(3)),
            Value::Number(Number::Integer(4)),
        ];
        assert_eq!(mul(arguments), Ok(Value::Number(Number::Integer(24))));
    }
}

fn div<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
) -> Result<Value<R>> {
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
        let arguments: Vec<Value<f32>> = vec![Value::Number(Number::Integer(2))];
        assert_eq!(div(arguments), Ok(Value::Number(Number::Real(0.5))));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(8)),
        ];
        assert_eq!(div(arguments), Ok(Value::Number(Number::Real(0.25))));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(8)),
            Value::Number(Number::Real(0.125)),
        ];
        assert_eq!(div(arguments), Ok(Value::<f32>::Number(Number::Real(2.))));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
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
        fn $func<R: RealNumberInternalTrait>(
            arguments: impl IntoIterator<Item = Value<R>>
        ) -> Result<Value<R>> {
            Ok(Value::Number(arguments.into_iter().next().unwrap().expect_number()?.$func()$($err_handle)?))
        }
    };
}

numeric_one_argument!(abs);
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
        let arguments: Vec<Value<f32>> = vec![Value::Number(Number::Rational(-49, 3))];
        assert_eq!(floor(arguments), Ok(Value::Number(Number::Integer(-17))));
    }
}

macro_rules! numeric_two_arguments {
    ($func:tt$(, $err_handle:tt)?) => {
        fn $func<R: RealNumberInternalTrait>(
            arguments: impl IntoIterator<Item = Value<R>>
        ) -> Result<Value<R>> {
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
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(8)),
            Value::Number(Number::Integer(3)),
        ];
        assert_eq!(
            floor_remainder(arguments),
            Ok(Value::Number(Number::Integer(2)))
        );
    }
}
fn vector<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
) -> Result<Value<R>> {
    let vector: Vec<Value<R>> = arguments.into_iter().collect();
    Ok(Value::Vector(ValueReference::new_mutable(vector)))
}

fn make_vector<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
) -> Result<Value<R>> {
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
        let arguments: Vec<Value<f32>> =
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
        let arguments: Vec<Value<f32>> =
            vec![Value::Number(Number::Integer(0)), Value::Boolean(true)];
        assert_eq!(
            make_vector(arguments),
            Ok(Value::Vector(ValueReference::new_mutable(vec![])))
        );
    }
    {
        let arguments: Vec<Value<f32>> =
            vec![Value::Number(Number::Integer(-1)), Value::Boolean(true)];
        assert_eq!(make_vector(arguments), error!(LogicError::NegativeLength));
    }
}

fn vector_length<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
) -> Result<Value<R>> {
    let vector = arguments.into_iter().next().unwrap().expect_vector()?;
    let len = vector.as_ref().len();
    Ok(Value::Number(Number::Integer(len as i32)))
}

#[test]
fn builtin_vector_length() {
    {
        let vector: Value<f32> = Value::Vector(ValueReference::new_immutable(vec![
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
        let vector: Value<f32> = Value::Vector(ValueReference::new_immutable(vec![]));
        let arguments = vec![vector.clone()];
        assert_eq!(
            vector_length(arguments),
            Ok(Value::Number(Number::Integer(0)))
        );
    }
}

fn vector_ref<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
) -> Result<Value<R>> {
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
    let vector: Value<f32> = Value::Vector(ValueReference::new_immutable(vec![
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

fn vector_set<R: RealNumberInternalTrait>(
    arguments: impl IntoIterator<Item = Value<R>>,
) -> Result<Value<R>> {
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
    let vector: Value<f32> = Value::Vector(ValueReference::new_mutable(vec![
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

fn newline<R: RealNumberInternalTrait>(_: impl IntoIterator<Item = Value<R>>) -> Result<Value<R>> {
    println!("");
    Ok(Value::Void)
}

macro_rules! typed_comparision {
    ($name:tt, $operator:tt, $expect_type: tt) => {
        fn $name<R: RealNumberInternalTrait>(
            arguments: impl IntoIterator<Item = Value<R>>,
        ) -> Result<Value<R>> {
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
        let arguments: Vec<Value<f32>> = vec![];
        assert_eq!(greater(arguments), Ok(Value::Boolean(true)));
    }
    {
        let arguments: Vec<Value<f32>> = vec![Value::Number(Number::Integer(2))];
        assert_eq!(greater(arguments), Ok(Value::Boolean(true)));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(2)),
        ];
        assert_eq!(greater(arguments), Ok(Value::Boolean(true)));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(8)),
        ];
        assert_eq!(greater(arguments), Ok(Value::Boolean(false)));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(1)),
        ];
        assert_eq!(greater(arguments), Ok(Value::Boolean(true)));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(2)),
        ];
        assert_eq!(greater(arguments), Ok(Value::Boolean(false)));
    }
}

macro_rules! first_of_order {
    ($name:tt, $cmp:tt) => {
        fn $name<R: RealNumberInternalTrait>(
                        arguments: impl IntoIterator<Item = Value<R>>,

        ) -> Result<Value<R>> {
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
        let arguments: Vec<Value<f32>> = vec![Value::Number(Number::Integer(2))];
        assert_eq!(min(arguments), Ok(Value::Number(Number::Integer(2))));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(2)),
        ];
        assert_eq!(min(arguments), Ok(Value::Number(Number::Integer(2))));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(8)),
        ];
        assert_eq!(min(arguments), Ok(Value::Number(Number::Integer(4))));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(1)),
        ];
        assert_eq!(min(arguments), Ok(Value::Number(Number::Integer(1))));
    }
    {
        let arguments: Vec<Value<f32>> = vec![
            Value::Number(Number::Integer(4)),
            Value::Number(Number::Integer(2)),
            Value::Number(Number::Integer(2)),
        ];
        assert_eq!(greater(arguments), Ok(Value::Boolean(false)));
    }
}

pub fn library_map<R: RealNumberInternalTrait>() -> Vec<(String, Value<R>)> {
    library_map_result().unwrap()
}

fn library_map_result<R: RealNumberInternalTrait>() -> Result<Vec<(String, Value<R>)>> {
    Ok(vec![
        function_mapping!(
            "apply",
            append_variadic_param!(param_fixed!["proc"], "args"),
            apply
        ),
        pure_function_mapping!("car", param_fixed!["pair"], car),
        pure_function_mapping!("cdr", param_fixed!["pair"], cdr),
        pure_function_mapping!("eqv?", param_fixed!["obj1", "obj2"], eqv),
        pure_function_mapping!(
            "eq?", // Ruschm is pass-by value now, so that eq? is equivalent to eqv?
            param_fixed!["obj1", "obj2"],
            eqv
        ),
        pure_function_mapping!("cons", param_fixed!["car", "cdr"], cons),
        pure_function_mapping!(
            "boolean?",
            param_fixed!["obj"],
            value_test!(Value::Boolean(_))
        ),
        pure_function_mapping!(
            "char?",
            param_fixed!["obj"],
            value_test!(Value::Character(_))
        ),
        pure_function_mapping!(
            "number?",
            param_fixed!["obj"],
            value_test!(Value::Number(_))
        ),
        pure_function_mapping!(
            "string?",
            param_fixed!["obj"],
            value_test!(Value::String(_))
        ),
        pure_function_mapping!(
            "symbol?",
            param_fixed!["obj"],
            value_test!(Value::Symbol(_))
        ),
        pure_function_mapping!("pair?", param_fixed!["obj"], is_pair),
        pure_function_mapping!(
            "procedure?",
            param_fixed!["obj"],
            value_test!(Value::Procedure(_))
        ),
        pure_function_mapping!(
            "vector?",
            param_fixed!["obj"],
            value_test!(Value::Vector(_))
        ),
        pure_function_mapping!("not", param_fixed!["obj"], not),
        pure_function_mapping!(
            "boolean=?",
            append_variadic_param!(param_fixed![], "booleans"),
            boolean_equal
        ),
        pure_function_mapping!("+", append_variadic_param!(param_fixed![], "x"), add),
        pure_function_mapping!("-", append_variadic_param!(param_fixed!["x1"], "x"), sub),
        pure_function_mapping!("*", append_variadic_param!(param_fixed![], "x"), mul),
        pure_function_mapping!("/", append_variadic_param!(param_fixed!["x1"], "x"), div),
        pure_function_mapping!("=", append_variadic_param!(param_fixed![], "x"), equals),
        pure_function_mapping!("<", append_variadic_param!(param_fixed![], "x"), less),
        pure_function_mapping!(
            "<=",
            append_variadic_param!(param_fixed![], "x"),
            less_equal
        ),
        pure_function_mapping!(">", append_variadic_param!(param_fixed![], "x"), greater),
        pure_function_mapping!(
            ">=",
            append_variadic_param!(param_fixed![], "x"),
            greater_equal
        ),
        pure_function_mapping!("min", append_variadic_param!(param_fixed!["x1"], "x"), min),
        pure_function_mapping!("max", append_variadic_param!(param_fixed!["x1"], "x"), max),
        pure_function_mapping!("abs", param_fixed!["x"], abs),
        pure_function_mapping!("sqrt", param_fixed!["x"], sqrt),
        pure_function_mapping!("exp", param_fixed!["z"], exp),
        pure_function_mapping!("ln", param_fixed!["z"], ln),
        pure_function_mapping!("log", param_fixed!["z1", "z2"], log),
        pure_function_mapping!("sin", param_fixed!["z"], sin),
        pure_function_mapping!("cos", param_fixed!["z"], cos),
        pure_function_mapping!("tan", param_fixed!["z"], tan),
        pure_function_mapping!("asin", param_fixed!["z"], asin),
        pure_function_mapping!("acos", param_fixed!["z"], acos),
        pure_function_mapping!("atan", param_fixed!["z"], atan),
        pure_function_mapping!("atan2", param_fixed!["y", "x"], atan2),
        pure_function_mapping!("floor", param_fixed!["x"], floor),
        pure_function_mapping!("ceiling", param_fixed!["x"], ceiling),
        pure_function_mapping!("exact", param_fixed!["x"], exact),
        pure_function_mapping!("floor-quotient", param_fixed!["n1", "n2"], floor_quotient),
        pure_function_mapping!("floor-remainder", param_fixed!["n1", "n2"], floor_remainder),
        pure_function_mapping!("newline", param_fixed![], newline),
        pure_function_mapping!(
            "vector",
            append_variadic_param!(param_fixed![], "obj"),
            vector
        ),
        pure_function_mapping!("make-vector", param_fixed!["k", "obj"], make_vector),
        pure_function_mapping!("vector-length", param_fixed!["vector"], vector_length),
        pure_function_mapping!("vector-ref", param_fixed!["vector", "k"], vector_ref),
        pure_function_mapping!(
            "vector-set!",
            param_fixed!["vector", "k", "obj"],
            vector_set
        ),
    ])
}

#[test]
fn builtin_parameters_length() -> Result<()> {
    let sqrt = library_map::<f32>()
        .into_iter()
        .find(|(name, _)| name.as_str() == "sqrt")
        .unwrap()
        .1
        .expect_procedure()?;
    let newline = library_map::<f32>()
        .into_iter()
        .find(|(name, _)| name.as_str() == "newline")
        .unwrap()
        .1
        .expect_procedure()?;

    println!("{}", sqrt.get_parameters());
    assert_eq!(sqrt.get_parameters().len(), (1, false));
    assert_eq!(newline.get_parameters().len(), (0, false));
    Ok(())
}
