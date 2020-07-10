#![allow(dead_code)]
use crate::env::Environment;
use crate::error::*;
use crate::parser::*;
use std::cell::RefCell;
use std::fmt;

type Result<T> = std::result::Result<T, Error>;

macro_rules! logic_error {
    ($($arg:tt)*) => (
        return Err(Error {category: ErrorType::Logic , message: format!($($arg)*) });
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Number {
    Integer(i64),
    Real(f64),
    Rational(i64, i64),
}

enum NumberBinaryOperand {
    Integer(i64, i64),
    Real(f64, f64),
    Rational(i64, i64, i64, i64),
}

// Integer => Rational => Real
fn upcast_oprands(operand: (Number, Number)) -> NumberBinaryOperand {
    match operand {
        (Number::Rational(dividend, dividor), Number::Real(b)) => {
            NumberBinaryOperand::Real(dividend as f64 / dividor as f64, b)
        }
        (Number::Real(a), Number::Rational(dividend, dividor)) => {
            NumberBinaryOperand::Real(a, dividend as f64 / dividor as f64)
        }
        (Number::Integer(a), Number::Real(b)) => (NumberBinaryOperand::Real(a as f64, b)),
        (Number::Real(a), Number::Integer(b)) => (NumberBinaryOperand::Real(a, b as f64)),
        (Number::Rational(dividend, dividor), Number::Integer(b)) => {
            NumberBinaryOperand::Rational(dividend, dividor, b, 1)
        }
        (Number::Integer(a), Number::Rational(dividend, dividor)) => {
            NumberBinaryOperand::Rational(a, 1, dividend, dividor)
        }
        (Number::Integer(a), Number::Integer(b)) => (NumberBinaryOperand::Integer(a, b)),
        (Number::Real(a), Number::Real(b)) => (NumberBinaryOperand::Real(a, b)),
        (Number::Rational(a1, a2), Number::Rational(b1, b2)) => {
            NumberBinaryOperand::Rational(a1, a2, b1, b2)
        }
    }
}

impl std::ops::Add<Number> for Number {
    type Output = Number;
    fn add(self, rhs: Number) -> Number {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Integer(a, b) => Number::Integer(a + b),
            NumberBinaryOperand::Real(a, b) => Number::Real(a + b),
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
            NumberBinaryOperand::Integer(a, b) => Number::Integer(a - b),
            NumberBinaryOperand::Real(a, b) => Number::Real(a - b),
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
            NumberBinaryOperand::Integer(a, b) => Number::Integer(a * b),
            NumberBinaryOperand::Real(a, b) => Number::Real(a * b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => Number::Rational(a1 * b1, a2 * b2),
        }
    }
}

impl std::ops::Div<Number> for Number {
    type Output = Result<Number>;
    fn div(self, rhs: Number) -> Result<Number> {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Integer(a, b) => {
                check_division_by_zero(b)?;
                match a % b {
                    0 => Ok(Number::Integer(a / b)),
                    _ => Ok(Number::Rational(a, b)),
                }
            }
            NumberBinaryOperand::Real(a, b) => Ok(Number::Real(a / b)),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => {
                check_division_by_zero(b1)?;
                check_division_by_zero(a2)?;
                check_division_by_zero(b2)?;
                Ok(Number::Rational(a1 * b2, a2 * b1))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Procedure {
    formals: Vec<String>,
    body: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValueType {
    Number(Number),
    Procedure(Procedure),
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ValueType::Number(num) => match num {
                Number::Integer(n) => write!(f, "{}", n),
                Number::Real(n) => write!(f, "{}", n),
                Number::Rational(a, b) => write!(f, "{}/{}", a, b),
            },
            ValueType::Procedure(_) => write!(f, "Procedure"),
        }
    }
}

fn check_division_by_zero(num: i64) -> Result<()> {
    match num {
        0 => logic_error!("division by exact zero"),
        _ => Ok(()),
    }
}

fn arithmetic_operators(ident: &str, a: Number, b: Number) -> Result<Number> {
    Ok(match ident {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        "/" => (a / b)?,
        _ => logic_error!("unrecognized arithmetic {}", ident),
    })
}

fn buildin_procedural<'a>(ident: &str, args: &Vec<ValueType>) -> Result<Option<ValueType>> {
    let mut iter = args.iter();
    match ident {
        // arithmetic
        "+" | "-" | "*" | "/" => {
            let init = match (ident, args.len()) {
                ("-", 0) => logic_error!("'-' needs at least one argument"),
                ("/", 0) => logic_error!("'/' needs at least one argument"),
                ("-", 1) | ("+", _) => ValueType::Number(Number::Integer(0)),
                ("/", 1) | ("*", _) => ValueType::Number(Number::Integer(1)),
                ("-", _) | ("/", _) => iter.next().unwrap().clone(),
                _ => logic_error!("unrecognized procedure {}", ident),
            };

            let result: Result<ValueType> = iter.try_fold(init, |a, b| match (a, b) {
                (ValueType::Number(num1), ValueType::Number(num2)) => {
                    Ok(ValueType::Number(arithmetic_operators(ident, num1, *num2)?))
                }
                _ => logic_error!("expect a number!"),
            });
            result.map(|val| Some(val))
        }
        _ => Ok(None),
    }
}

// pub fn procedure_call(procedure: &Procedure, args: Vec<ValueType>, env: &Environment) {}

pub struct Interpreter<'a> {
    env: RefCell<Environment<'a>>,
}

impl<'a> Interpreter<'a> {
    pub fn new() -> Self {
        Self {
            env: RefCell::new(Environment::new()),
        }
    }

    pub fn define<'b>(
        &'a self,
        definition: &Definition,
        env: &'b RefCell<Environment<'b>>,
    ) -> Result<()> {
        let Definition(name, expression) = definition;
        let value = self.eval_expression(&expression, env)?;
        env.borrow_mut().define(name.clone(), value);
        Ok(())
    }

    fn eval_scheme_procedure<'b>(
        &'a self,
        formals: &Vec<String>,
        body: &Expression,
        args: impl Iterator<Item = ValueType>,
        parent_env: &'b Environment<'b>,
    ) -> Result<ValueType> {
        let child_env = RefCell::new(Environment::child(parent_env));
        for (param, arg) in formals.iter().zip(args) {
            child_env.borrow_mut().define(param.clone(), arg.clone());
        }
        self.eval_expression(&body, &child_env)
    }

    pub fn eval_root_expression(&'a self, expression: Expression) -> Result<ValueType> {
        self.eval_expression(&expression, &self.env)
    }

    pub fn eval_expression<'b>(
        &'a self,
        expression: &Expression,
        env: &'b RefCell<Environment<'b>>,
    ) -> Result<ValueType> {
        Ok(match expression {
            Expression::ProcedureCall(procedure, arguments) => {
                let mut evaluated_args = vec![];
                for arg in arguments {
                    evaluated_args.push(self.eval_expression(arg, env)?);
                }
                let parent_env = &env.borrow();
                match procedure.as_ref() {
                    Expression::Identifier(ident) => {
                        match buildin_procedural(ident.as_str(), &evaluated_args)? {
                            Some(value) => value,
                            None => match parent_env.get(ident.as_str()) {
                                Some(ValueType::Procedure(procedure)) => self
                                    .eval_scheme_procedure(
                                        &procedure.formals,
                                        &procedure.body,
                                        evaluated_args.into_iter(),
                                        parent_env,
                                    )?,
                                Some(other) => logic_error!("{} is not callable", other),
                                None => {
                                    logic_error!("try to call an undefined identifier: {}", ident)
                                }
                            },
                        }
                    }
                    Expression::Procedure(formals, body) => self.eval_scheme_procedure(
                        formals,
                        body,
                        evaluated_args.into_iter(),
                        parent_env,
                    )?,
                    _ => logic_error!("expect a procedure here"),
                }
            }
            Expression::Procedure(formals, body) => ValueType::Procedure(Procedure {
                formals: formals.clone(),
                body: *body.clone(),
            }),
            Expression::Integer(value) => ValueType::Number(Number::Integer(*value)),
            Expression::Real(number_literal) => {
                ValueType::Number(Number::Real(number_literal.parse::<f64>().unwrap()))
            }
            // TODO: apply gcd here.
            Expression::Rational(a, b) => ValueType::Number(Number::Rational(*a, *b as i64)),
            Expression::Identifier(ident) => match env.borrow().get(ident.as_str()) {
                Some(value) => value.clone(),
                None => logic_error!("undefined identifier: {}", ident),
            },
        })
    }

    pub fn eval_ast<'b>(
        &'a self,
        ast: &Statement,
        env: &'b RefCell<Environment<'b>>,
    ) -> Result<Option<ValueType>> {
        Ok(match ast {
            Statement::ImportDeclaration(_) => {
                // TODO:
                None
            }
            Statement::Definition(definition) => {
                self.define(definition, env)?;
                None
            }
            Statement::Expression(expr) => Some(self.eval_expression(&expr, env)?),
        })
    }

    pub fn eval_root_ast(&'a self, ast: &Statement) -> Result<Option<ValueType>> {
        self.eval_ast(ast, &self.env)
    }

    pub fn eval_program(
        &'a self,
        asts: impl IntoIterator<Item = &'a Statement>,
    ) -> Result<Option<ValueType>> {
        asts.into_iter()
            .try_fold(None, |_, ast| self.eval_root_ast(ast))
    }
}

#[test]
fn number() -> Result<()> {
    let interpreter = Interpreter::new();
    assert_eq!(
        interpreter.eval_root_expression(Expression::Integer(-1))?,
        ValueType::Number(Number::Integer(-1))
    );
    assert_eq!(
        interpreter.eval_root_expression(Expression::Rational(1, 3))?,
        ValueType::Number(Number::Rational(1, 3))
    );
    assert_eq!(
        interpreter.eval_root_expression(Expression::Real("-3.45e-7".to_string()))?,
        ValueType::Number(Number::Real(-3.45e-7))
    );
    Ok(())
}

#[test]
fn arithmetic() -> Result<()> {
    let interpreter = Interpreter::new();
    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("+".to_string())),
            vec![
                Box::new(Expression::Integer(1)),
                Box::new(Expression::Integer(2))
            ]
        ))?,
        ValueType::Number(Number::Integer(3))
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("+".to_string())),
            vec![
                Box::new(Expression::Integer(1)),
                Box::new(Expression::Rational(1, 2))
            ]
        ))?,
        ValueType::Number(Number::Rational(3, 2))
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("+".to_string())),
            vec![
                Box::new(Expression::Rational(1, 2)),
                Box::new(Expression::Real("0.5".to_string())),
            ]
        ))?,
        ValueType::Number(Number::Real(1.0)),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("/".to_string())),
            vec![
                Box::new(Expression::Integer(1)),
                Box::new(Expression::Integer(0)),
            ]
        )),
        Err(Error {
            category: ErrorType::Logic,
            message: "division by exact zero".to_string()
        }),
    );
    Ok(())
}

#[test]
fn undefined() {
    let interpreter = Interpreter::new();
    assert_eq!(
        interpreter.eval_root_expression(Expression::Identifier("foo".to_string())),
        Err(Error {
            category: ErrorType::Logic,
            message: "undefined identifier: foo".to_string(),
        })
    );
}

#[test]
fn variable_definition() -> Result<()> {
    let interpreter = Interpreter::new();
    let program = vec![
        Statement::Definition(Definition("a".to_string(), Expression::Integer(1))),
        Statement::Definition(Definition(
            "b".to_string(),
            Expression::Identifier("a".to_string()),
        )),
        Statement::Expression(Expression::Identifier("b".to_string())),
    ];
    assert_eq!(
        interpreter.eval_program(program.iter())?,
        Some(ValueType::Number(Number::Integer(1)))
    );
    Ok(())
}

#[test]
fn procedure_definition() -> Result<()> {
    let interpreter = Interpreter::new();
    let program = vec![
        Statement::Definition(Definition(
            "add".to_string(),
            Expression::Procedure(
                vec!["x".to_string(), "y".to_string()],
                Box::new(Expression::ProcedureCall(
                    Box::new(Expression::Identifier("+".to_string())),
                    vec![
                        Box::new(Expression::Identifier("x".to_string())),
                        Box::new(Expression::Identifier("y".to_string())),
                    ],
                )),
            ),
        )),
        Statement::Expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("add".to_string())),
            vec![
                Box::new(Expression::Integer(1)),
                Box::new(Expression::Integer(2)),
            ],
        )),
    ];
    assert_eq!(
        interpreter.eval_program(program.iter())?,
        Some(ValueType::Number(Number::Integer(3)))
    );
    Ok(())
}
#[test]
fn lambda_call() -> Result<()> {
    let interpreter = Interpreter::new();
    let program = vec![Statement::Expression(Expression::ProcedureCall(
        Box::new(Expression::Procedure(
            vec!["x".to_string(), "y".to_string()],
            Box::new(Expression::ProcedureCall(
                Box::new(Expression::Identifier("+".to_string())),
                vec![
                    Box::new(Expression::Identifier("x".to_string())),
                    Box::new(Expression::Identifier("y".to_string())),
                ],
            )),
        )),
        vec![
            Box::new(Expression::Integer(1)),
            Box::new(Expression::Integer(2)),
        ],
    ))];
    assert_eq!(
        interpreter.eval_program(program.iter())?,
        Some(ValueType::Number(Number::Integer(3)))
    );
    Ok(())
}

#[test]
fn local_environment() -> Result<()> {
    let interpreter = Interpreter::new();
    let program = vec![
        Statement::Definition(Definition(
            "adda".to_string(),
            Expression::Procedure(
                vec!["x".to_string()],
                Box::new(Expression::ProcedureCall(
                    Box::new(Expression::Identifier("+".to_string())),
                    vec![
                        Box::new(Expression::Identifier("x".to_string())),
                        Box::new(Expression::Identifier("a".to_string())),
                    ],
                )),
            ),
        )),
        Statement::Definition(Definition("a".to_string(), Expression::Integer(1))),
        Statement::Expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("adda".to_string())),
            vec![Box::new(Expression::Integer(2))],
        )),
    ];
    assert_eq!(
        interpreter.eval_program(program.iter())?,
        Some(ValueType::Number(Number::Integer(3)))
    );
    Ok(())
}

#[test]
fn procedure_as_data() -> Result<()> {
    let interpreter = Interpreter::new();
    let program = vec![
        Statement::Definition(Definition(
            "add".to_string(),
            Expression::Procedure(
                vec!["x".to_string(), "y".to_string()],
                Box::new(Expression::ProcedureCall(
                    Box::new(Expression::Identifier("+".to_string())),
                    vec![
                        Box::new(Expression::Identifier("x".to_string())),
                        Box::new(Expression::Identifier("y".to_string())),
                    ],
                )),
            ),
        )),
        Statement::Definition(Definition(
            "apply-op".to_string(),
            Expression::Procedure(
                vec!["op".to_string(), "x".to_string(), "y".to_string()],
                Box::new(Expression::ProcedureCall(
                    Box::new(Expression::Identifier("op".to_string())),
                    vec![
                        Box::new(Expression::Identifier("x".to_string())),
                        Box::new(Expression::Identifier("y".to_string())),
                    ],
                )),
            ),
        )),
        Statement::Expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("apply-op".to_string())),
            vec![
                Box::new(Expression::Identifier("add".to_string())),
                Box::new(Expression::Integer(1)),
                Box::new(Expression::Integer(2)),
            ],
        )),
    ];
    assert_eq!(
        interpreter.eval_program(program.iter())?,
        Some(ValueType::Number(Number::Integer(3)))
    );
    Ok(())
}
