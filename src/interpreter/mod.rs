#![allow(dead_code)]
use crate::environment::Environment;
use crate::error::*;
use crate::lexer::*;
use crate::parser::*;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt;

use std::iter::Iterator;

type Result<T> = std::result::Result<T, Error>;

macro_rules! logic_error {
    ($($arg:tt)*) => (
        return Err(Error {category: ErrorType::Logic , message: format!($($arg)*) });
    )
}

pub mod scheme;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Number {
    Integer(i64),
    Real(f64),
    Rational(i64, i64),
}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Number) -> Option<Ordering> {
        match upcast_oprands((*self, *other)) {
            NumberBinaryOperand::Integer(a, b) => a.partial_cmp(&b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => (a1 * b2).partial_cmp(&(b1 * a2)),
            NumberBinaryOperand::Real(a, b) => a.partial_cmp(&b),
        }
    }
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

impl NumberBinaryOperand {
    pub fn lhs(&self) -> Number {
        match self {
            NumberBinaryOperand::Integer(a, _) => Number::Integer(*a),
            NumberBinaryOperand::Real(a, _) => Number::Real(*a),
            NumberBinaryOperand::Rational(a1, a2, _, _) => Number::Rational(*a1, *a2),
        }
    }

    pub fn rhs(&self) -> Number {
        match self {
            NumberBinaryOperand::Integer(_, b) => Number::Integer(*b),
            NumberBinaryOperand::Real(_, b) => Number::Real(*b),
            NumberBinaryOperand::Rational(_, _, b1, b2) => Number::Rational(*b1, *b2),
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

#[derive(Clone)]
pub struct BuildinProcedure(
    &'static str,
    fn(Box<dyn Iterator<Item = Result<ValueType>> + '_>) -> Result<ValueType>,
);

impl fmt::Display for BuildinProcedure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<build-in procedure ({})>", self.0)
    }
}
impl fmt::Debug for BuildinProcedure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl PartialEq for BuildinProcedure {
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0 && self.1 as usize == rhs.1 as usize
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Procedure {
    User(SchemeProcedure),
    Buildin(BuildinProcedure),
}

impl fmt::Display for Procedure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Procedure::User(procedure) => write!(f, "{}", procedure),
            Procedure::Buildin(fp) => write!(f, "{}", fp),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValueType {
    Number(Number),
    Boolean(bool),
    Datum(Box<Statement>),
    Procedure(Procedure),
    Vector(Vec<ValueType>),
    Void,
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ValueType::Number(num) => match num {
                Number::Integer(n) => write!(f, "{}", n),
                Number::Real(n) => write!(f, "{:?}", n),
                Number::Rational(a, b) => write!(f, "{}/{}", a, b),
            },
            ValueType::Datum(expr) => write!(f, "{}", expr),
            ValueType::Procedure(p) => write!(f, "{}", p),
            ValueType::Void => write!(f, "Void"),
            ValueType::Boolean(true) => write!(f, "#t"),
            ValueType::Boolean(false) => write!(f, "#f"),
            ValueType::Vector(vec) => write!(
                f,
                "#({})",
                vec.iter()
                    .map(|v| format!("{}", v))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        }
    }
}

fn check_division_by_zero(num: i64) -> Result<()> {
    match num {
        0 => logic_error!("division by exact zero"),
        _ => Ok(()),
    }
}

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
        definitions: &Vec<Definition>,
        expressions: &Vec<Expression>,
        args: impl Iterator<Item = Result<ValueType>>,
        parent_env: &'b Environment<'b>,
    ) -> Result<ValueType> {
        let child_env = RefCell::new(Environment::child(parent_env));
        for (param, arg) in formals.iter().zip(args) {
            child_env.borrow_mut().define(param.clone(), arg?);
        }
        for def in definitions {
            self.define(def, &child_env)?;
        }
        match expressions.split_last() {
            Some((last, before_last)) => {
                for expr in before_last {
                    self.eval_expression(expr, &child_env)?;
                }
                self.eval_expression(last, &child_env)
            }
            None => logic_error!("no expression in function body"),
        }
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
            Expression::ProcedureCall(procedure_expr, arguments) => {
                let procedure = self.eval_expression(procedure_expr, env)?;
                let evaluated_args = Box::new(
                    arguments
                        .into_iter()
                        .map(|arg| self.eval_expression(arg, env)),
                );
                let parent_env = &env.borrow();
                match procedure {
                    ValueType::Procedure(Procedure::Buildin(BuildinProcedure(_, fp))) => {
                        fp(evaluated_args)?
                    }
                    ValueType::Procedure(Procedure::User(SchemeProcedure(
                        formals,
                        definitions,
                        expressions,
                    ))) => self.eval_scheme_procedure(
                        &formals,
                        &definitions,
                        &expressions,
                        evaluated_args,
                        parent_env,
                    )?,
                    _ => logic_error!("expect a procedure here"),
                }
            }
            Expression::Vector(vector) => {
                let mut values = Vec::with_capacity(vector.len());
                for expr in vector {
                    values.push(self.eval_expression(expr, env)?);
                }
                ValueType::Vector(values)
            }
            Expression::Procedure(scheme) => ValueType::Procedure(Procedure::User(scheme.clone())),
            Expression::Conditional(cond) => {
                let &(test, consequent, alternative) = &cond.as_ref();
                match self.eval_expression(&test, env)? {
                    ValueType::Boolean(true) => self.eval_expression(&consequent, env)?,
                    ValueType::Boolean(false) => match alternative {
                        Some(alter) => self.eval_expression(&alter, env)?,
                        None => ValueType::Void,
                    },
                    _ => logic_error!("if condition should be a boolean expression"),
                }
            }
            Expression::Datum(datum) => ValueType::Datum(datum.clone()),
            Expression::Boolean(value) => ValueType::Boolean(*value),
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
            Statement::ImportDeclaration(_) => None, // TODO
            Statement::Expression(expr) => Some(self.eval_expression(&expr, env)?),
            Statement::Definition(definition) => {
                self.define(definition, env)?;
                None
            }
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

    pub fn eval(&'a self, char_stream: impl Iterator<Item = char>) -> Result<Option<ValueType>> {
        {
            let mut char_visitor = char_stream.peekable();
            let mut last_value = None;
            loop {
                let token_stream = TokenGenerator::new(&mut char_visitor);
                let result: Result<ParseResult> = token_stream.collect();
                match result?? {
                    Some(ast) => last_value = self.eval_root_ast(&ast)?,
                    None => break Ok(last_value),
                }
            }
        }
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
            vec![Expression::Integer(1), Expression::Integer(2)]
        ))?,
        ValueType::Number(Number::Integer(3))
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("+".to_string())),
            vec![Expression::Integer(1), Expression::Rational(1, 2)]
        ))?,
        ValueType::Number(Number::Rational(3, 2))
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("*".to_string())),
            vec![
                Expression::Rational(1, 2),
                Expression::Real("2.0".to_string()),
            ]
        ))?,
        ValueType::Number(Number::Real(1.0)),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("/".to_string())),
            vec![Expression::Integer(1), Expression::Integer(0)]
        )),
        Err(Error {
            category: ErrorType::Logic,
            message: "division by exact zero".to_string()
        }),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("max".to_string())),
            vec![Expression::Integer(1), Expression::Real("1.3".to_string()),]
        ))?,
        ValueType::Number(Number::Real(1.3)),
    );
    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("min".to_string())),
            vec![Expression::Integer(1), Expression::Real("1.3".to_string()),]
        ))?,
        ValueType::Number(Number::Real(1.0)),
    );
    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("min".to_string())),
            vec![Expression::Identifier("+".to_string()),]
        )),
        Err(Error {
            category: ErrorType::Logic,
            message: "expect a number!".to_string()
        }),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("max".to_string())),
            vec![Expression::Identifier("+".to_string())]
        )),
        Err(Error {
            category: ErrorType::Logic,
            message: "expect a number!".to_string()
        }),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("sqrt".to_string())),
            vec![Expression::Integer(4)]
        ))?,
        ValueType::Number(Number::Real(2.0)),
    );

    match interpreter.eval_root_expression(Expression::ProcedureCall(
        Box::new(Expression::Identifier("sqrt".to_string())),
        vec![Expression::Integer(-4)],
    ))? {
        ValueType::Number(Number::Real(should_be_nan)) => assert!(should_be_nan.is_nan()),
        _ => panic!("sqrt result should be a number"),
    }

    for (cmp, result) in [">", "<", ">=", "<="]
        .iter()
        .zip([false, false, true, true].iter())
    {
        assert_eq!(
            interpreter.eval_root_expression(Expression::ProcedureCall(
                Box::new(Expression::Identifier(cmp.to_string())),
                vec![
                    Expression::Integer(1),
                    Expression::Integer(1),
                    Expression::Integer(1),
                ],
            ))?,
            ValueType::Boolean(*result)
        )
    }

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
fn buildin_procedural() -> Result<()> {
    let interpreter = Interpreter::new();
    let program = vec![
        Statement::Definition(Definition(
            "get-add".to_string(),
            simple_procedure(vec![], Expression::Identifier("+".to_string())),
        )),
        Statement::Expression(Expression::ProcedureCall(
            Box::new(Expression::ProcedureCall(
                Box::new(Expression::Identifier("get-add".to_string())),
                vec![],
            )),
            vec![Expression::Integer(1), Expression::Integer(2)],
        )),
    ];
    assert_eq!(
        interpreter.eval_program(program.iter())?,
        Some(ValueType::Number(Number::Integer(3)))
    );
    Ok(())
}

#[test]
fn procedure_definition() -> Result<()> {
    let interpreter = Interpreter::new();
    let program = vec![
        Statement::Definition(Definition(
            "add".to_string(),
            simple_procedure(
                vec!["x".to_string(), "y".to_string()],
                Expression::ProcedureCall(
                    Box::new(Expression::Identifier("+".to_string())),
                    vec![
                        Expression::Identifier("x".to_string()),
                        Expression::Identifier("y".to_string()),
                    ],
                ),
            ),
        )),
        Statement::Expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("add".to_string())),
            vec![Expression::Integer(1), Expression::Integer(2)],
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
        Box::new(simple_procedure(
            vec!["x".to_string(), "y".to_string()],
            Expression::ProcedureCall(
                Box::new(Expression::Identifier("+".to_string())),
                vec![
                    Expression::Identifier("x".to_string()),
                    Expression::Identifier("y".to_string()),
                ],
            ),
        )),
        vec![Expression::Integer(1), Expression::Integer(2)],
    ))];
    assert_eq!(
        interpreter.eval_program(program.iter())?,
        Some(ValueType::Number(Number::Integer(3)))
    );
    Ok(())
}

#[test]
fn condition() -> Result<()> {
    let interpreter = Interpreter::new();
    let program = vec![Statement::Expression(Expression::Conditional(Box::new((
        Expression::Boolean(true),
        Expression::Integer(1),
        Some(Expression::Integer(2)),
    ))))];
    assert_eq!(
        interpreter.eval_program(program.iter())?,
        Some(ValueType::Number(Number::Integer(1)))
    );
    Ok(())
}

#[test]
fn local_environment() -> Result<()> {
    let interpreter = Interpreter::new();
    let program = vec![
        Statement::Definition(Definition(
            "adda".to_string(),
            simple_procedure(
                vec!["x".to_string()],
                Expression::ProcedureCall(
                    Box::new(Expression::Identifier("+".to_string())),
                    vec![
                        Expression::Identifier("x".to_string()),
                        Expression::Identifier("a".to_string()),
                    ],
                ),
            ),
        )),
        Statement::Definition(Definition("a".to_string(), Expression::Integer(1))),
        Statement::Expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("adda".to_string())),
            vec![Expression::Integer(2)],
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
            simple_procedure(
                vec!["x".to_string(), "y".to_string()],
                Expression::ProcedureCall(
                    Box::new(Expression::Identifier("+".to_string())),
                    vec![
                        Expression::Identifier("x".to_string()),
                        Expression::Identifier("y".to_string()),
                    ],
                ),
            ),
        )),
        Statement::Definition(Definition(
            "apply-op".to_string(),
            simple_procedure(
                vec!["op".to_string(), "x".to_string(), "y".to_string()],
                Expression::ProcedureCall(
                    Box::new(Expression::Identifier("op".to_string())),
                    vec![
                        Expression::Identifier("x".to_string()),
                        Expression::Identifier("y".to_string()),
                    ],
                ),
            ),
        )),
        Statement::Expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("apply-op".to_string())),
            vec![
                Expression::Identifier("add".to_string()),
                Expression::Integer(1),
                Expression::Integer(2),
            ],
        )),
    ];
    assert_eq!(
        interpreter.eval_program(program.iter())?,
        Some(ValueType::Number(Number::Integer(3)))
    );
    Ok(())
}
