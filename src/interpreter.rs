use crate::lexer::TokenGenerator;
use crate::parser::*;
use std::fmt;

#[derive(Debug, PartialEq)]
pub struct LogicError {
    error: String,
}

impl fmt::Display for LogicError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "error: {}", self.error)
    }
}

macro_rules! logic_error {
    ($($arg:tt)*) => (
        return Err(LogicError { error: format!($($arg)*) });
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Number {
    Interger(i64),
    Demicals(f64),
    Rational(i64, i64),
}

enum NumberBinaryOperand {
    Interger(i64, i64),
    Demicals(f64, f64),
    Rational(i64, i64, i64, i64),
}

// Interger => Rational => Demicals
fn upcast_oprands(operand: (Number, Number)) -> NumberBinaryOperand {
    match operand {
        (Number::Rational(dividend, dividor), Number::Demicals(b)) => {
            NumberBinaryOperand::Demicals(dividend as f64 / dividor as f64, b)
        }
        (Number::Demicals(a), Number::Rational(dividend, dividor)) => {
            NumberBinaryOperand::Demicals(a, dividend as f64 / dividor as f64)
        }
        (Number::Interger(a), Number::Demicals(b)) => (NumberBinaryOperand::Demicals(a as f64, b)),
        (Number::Demicals(a), Number::Interger(b)) => (NumberBinaryOperand::Demicals(a, b as f64)),
        (Number::Rational(dividend, dividor), Number::Interger(b)) => {
            NumberBinaryOperand::Rational(dividend, dividor, b, 1)
        }
        (Number::Interger(a), Number::Rational(dividend, dividor)) => {
            NumberBinaryOperand::Rational(a, 1, dividend, dividor)
        }
        (Number::Interger(a), Number::Interger(b)) => (NumberBinaryOperand::Interger(a, b)),
        (Number::Demicals(a), Number::Demicals(b)) => (NumberBinaryOperand::Demicals(a, b)),
        (Number::Rational(a1, a2), Number::Rational(b1, b2)) => {
            NumberBinaryOperand::Rational(a1, a2, b1, b2)
        }
    }
}

impl std::ops::Add<Number> for Number {
    type Output = Number;
    fn add(self, rhs: Number) -> Number {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Interger(a, b) => Number::Interger(a + b),
            NumberBinaryOperand::Demicals(a, b) => Number::Demicals(a + b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => {
                Number::Rational(a1 * b2 + a2 * b1, a2 * b2)
            }
        }
    }
}

impl std::ops::Sub<Number> for Number {
    type Output = Number;
    fn sub(self, rhs: Number) -> Number {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Interger(a, b) => Number::Interger(a - b),
            NumberBinaryOperand::Demicals(a, b) => Number::Demicals(a - b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => {
                Number::Rational(a1 * b2 - a2 * b1, a2 * b2)
            }
        }
    }
}

impl std::ops::Mul<Number> for Number {
    type Output = Number;
    fn mul(self, rhs: Number) -> Number {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Interger(a, b) => Number::Interger(a * b),
            NumberBinaryOperand::Demicals(a, b) => Number::Demicals(a * b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => Number::Rational(a1 * b1, a2 * b2),
        }
    }
}

impl std::ops::Div<Number> for Number {
    type Output = Result<Number, LogicError>;
    fn div(self, rhs: Number) -> Result<Number, LogicError> {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Interger(a, b) => {
                check_division_by_zero(b)?;
                match a % b {
                    0 => Ok(Number::Interger(a / b)),
                    _ => Ok(Number::Rational(a, b)),
                }
            }
            NumberBinaryOperand::Demicals(a, b) => Ok(Number::Demicals(a / b)),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => {
                check_division_by_zero(b1)?;
                check_division_by_zero(a2)?;
                check_division_by_zero(b2)?;
                Ok(Number::Rational(a1 / b1, a2 / b2))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValueType {
    Number(Number),
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ValueType::Number(num) => match num {
                Number::Interger(n) => write!(f, "{}", n),
                Number::Demicals(n) => write!(f, "{}", n),
                Number::Rational(a, b) => write!(f, "{}/{}", a, b),
            },
        }
    }
}

pub fn parse(s: &str) -> Result<Option<Box<Expression>>, SyntaxError> {
    let mut parser = Parser::new(TokenGenerator::new(s.chars()));
    parser.parse()
}

fn check_division_by_zero(num: i64) -> Result<(), LogicError> {
    match num {
        0 => logic_error!("division by exact zero"),
        _ => Ok(()),
    }
}

fn arithmetic_operators(ident: &str, a: Number, b: Number) -> Result<Number, LogicError> {
    Ok(match ident {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        "/" => (a / b)?,
        _ => logic_error!("unrecognized arithmetic {}", ident),
    })
}

fn buildin_procedural(ident: &str, args: &Vec<ValueType>) -> Result<Option<ValueType>, LogicError> {
    let mut iter = args.iter();
    match ident {
        // arithmetic
        "+" | "-" | "*" | "/" => {
            let init = match (ident, args.len()) {
                ("-", 0) => logic_error!("'-' needs at least one argument"),
                ("/", 0) => logic_error!("'/' needs at least one argument"),
                ("-", 1) | ("+", _) => ValueType::Number(Number::Interger(0)),
                ("/", 1) | ("*", _) => ValueType::Number(Number::Interger(1)),
                ("-", _) | ("/", _) => iter.next().unwrap().clone(),
                _ => logic_error!("unrecognized procedure {}", ident),
            };

            let result: Result<ValueType, LogicError> =
                iter.try_fold(init, |ValueType::Number(a), ValueType::Number(b)| {
                    Ok(ValueType::Number(arithmetic_operators(ident, a, *b)?))
                });
            result.map(|val| Some(val))
        }
        _ => Ok(None),
    }
}

pub fn eval(ast: &Expression) -> Result<ValueType, LogicError> {
    Ok(match ast {
        Expression::ProcudureCall(procedure, arguments) => {
            let mut evaluated_args = vec![];
            for arg in arguments {
                evaluated_args.push(eval(arg)?);
            }
            match procedure.as_ref() {
                Expression::Identifier(ident) => {
                    match buildin_procedural(ident.as_str(), &evaluated_args)? {
                        Some(value) => value,
                        None => logic_error!("undefined identifier: {}", ident),
                    }
                }
                _ => logic_error!("expect a procedure here"),
            }
        }
        Expression::Interger(value) => ValueType::Number(Number::Interger(*value)),
        Expression::Demicals(number_literal) => {
            ValueType::Number(Number::Demicals(number_literal.parse::<f64>().unwrap()))
        }
        // TODO: apply gcd here.
        Expression::Rational(a, b) => ValueType::Number(Number::Rational(*a, *b as i64)),
        Expression::Identifier(ident) => logic_error!("undefined identifier: {}", ident),
    })
}

#[test]
fn number() -> Result<(), LogicError> {
    assert_eq!(
        eval(&Expression::Interger(-1))?,
        ValueType::Number(Number::Interger(-1))
    );
    assert_eq!(
        eval(&Expression::Rational(1, 3))?,
        ValueType::Number(Number::Rational(1, 3))
    );
    assert_eq!(
        eval(&Expression::Demicals("-3.45e-7".to_string()))?,
        ValueType::Number(Number::Demicals(-3.45e-7))
    );
    Ok(())
}

#[test]
fn arithmetic() -> Result<(), LogicError> {
    assert_eq!(
        eval(&Expression::ProcudureCall(
            Box::new(Expression::Identifier("+".to_string())),
            vec![
                Box::new(Expression::Interger(1)),
                Box::new(Expression::Interger(2))
            ]
        ))?,
        ValueType::Number(Number::Interger(3))
    );

    assert_eq!(
        eval(&Expression::ProcudureCall(
            Box::new(Expression::Identifier("+".to_string())),
            vec![
                Box::new(Expression::Interger(1)),
                Box::new(Expression::Rational(1, 2))
            ]
        ))?,
        ValueType::Number(Number::Rational(3, 2))
    );

    assert_eq!(
        eval(&Expression::ProcudureCall(
            Box::new(Expression::Identifier("+".to_string())),
            vec![
                Box::new(Expression::Rational(1, 2)),
                Box::new(Expression::Demicals("0.5".to_string())),
            ]
        ))?,
        ValueType::Number(Number::Demicals(1.0)),
    );

    assert_eq!(
        eval(&Expression::ProcudureCall(
            Box::new(Expression::Identifier("/".to_string())),
            vec![
                Box::new(Expression::Interger(1)),
                Box::new(Expression::Interger(0)),
            ]
        )),
        Err(LogicError {
            error: "division by exact zero".to_string()
        }),
    );
    Ok(())
}

#[test]
fn undefined() {
    assert_eq!(
        eval(&Expression::Identifier("foo".to_string())),
        Err(LogicError {
            error: "undefined identifier: foo".to_string(),
        })
    );
}
