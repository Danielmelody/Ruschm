#![allow(dead_code)]
use crate::environment::*;
use crate::error::*;
use crate::lexer::*;
use crate::parser::*;
use num_traits::real::Real;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt;
use std::iter::Iterator;
use std::marker::PhantomData;
use std::rc::Rc;

type Result<T> = std::result::Result<T, SchemeError>;

macro_rules! logic_error {
    ($($arg:tt)*) => (
        return Err(SchemeError {category: ErrorType::Logic , message: format!($($arg)*) });
    )
}

pub mod scheme;

pub trait RealNumberInternalTrait: fmt::Display + fmt::Debug + Real
where
    Self: std::marker::Sized,
{
}

impl<T: fmt::Display + fmt::Debug + Real> RealNumberInternalTrait for T {}
#[derive(Debug, Clone, Copy)]
pub enum Number<InternalReal: RealNumberInternalTrait> {
    Integer(i32),
    Real(InternalReal),
    Rational(i32, i32),
}

impl<InternalReal: RealNumberInternalTrait> fmt::Display for Number<InternalReal> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Number::Integer(n) => write!(f, "{}", n),
            Number::Real(n) => write!(f, "{:?}", n),
            Number::Rational(a, b) => write!(f, "{}/{}", a, b),
        }
    }
}

impl<InternalReal: RealNumberInternalTrait> PartialEq for Number<InternalReal> {
    fn eq(&self, other: &Number<InternalReal>) -> bool {
        match upcast_oprands((*self, *other)) {
            NumberBinaryOperand::Integer(a, b) => a.eq(&b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => (a1 * b2).eq(&(b1 * a2)),
            NumberBinaryOperand::Real(a, b) => a.eq(&b),
        }
    }
}

impl<InternalReal: RealNumberInternalTrait> PartialOrd for Number<InternalReal> {
    fn partial_cmp(&self, other: &Number<InternalReal>) -> Option<Ordering> {
        match upcast_oprands((*self, *other)) {
            NumberBinaryOperand::Integer(a, b) => a.partial_cmp(&b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => (a1 * b2).partial_cmp(&(b1 * a2)),
            NumberBinaryOperand::Real(a, b) => a.partial_cmp(&b),
        }
    }
}

enum NumberBinaryOperand<InternalReal> {
    Integer(i32, i32),
    Real(InternalReal, InternalReal),
    Rational(i32, i32, i32, i32),
}

// Integer => Rational => Real
fn upcast_oprands<InternalReal: RealNumberInternalTrait>(
    operand: (Number<InternalReal>, Number<InternalReal>),
) -> NumberBinaryOperand<InternalReal> {
    match operand {
        (Number::Rational(dividend, dividor), Number::Real(b)) => NumberBinaryOperand::Real(
            InternalReal::from(dividend).unwrap() / InternalReal::from(dividor).unwrap(),
            b,
        ),
        (Number::Real(a), Number::Rational(dividend, dividor)) => NumberBinaryOperand::Real(
            a,
            InternalReal::from(dividend).unwrap() / InternalReal::from(dividor).unwrap(),
        ),
        (Number::Integer(a), Number::Real(b)) => {
            NumberBinaryOperand::Real(InternalReal::from(a).unwrap(), b)
        }
        (Number::Real(a), Number::Integer(b)) => {
            NumberBinaryOperand::Real(a, InternalReal::from(b).unwrap())
        }
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

impl<InternalReal: RealNumberInternalTrait> NumberBinaryOperand<InternalReal> {
    pub fn lhs(&self) -> Number<InternalReal> {
        match self {
            NumberBinaryOperand::Integer(a, _) => Number::Integer(*a),
            NumberBinaryOperand::Real(a, _) => Number::Real(*a),
            NumberBinaryOperand::Rational(a1, a2, _, _) => Number::Rational(*a1, *a2),
        }
    }

    pub fn rhs(&self) -> Number<InternalReal> {
        match self {
            NumberBinaryOperand::Integer(_, b) => Number::Integer(*b),
            NumberBinaryOperand::Real(_, b) => Number::Real(*b),
            NumberBinaryOperand::Rational(_, _, b1, b2) => Number::Rational(*b1, *b2),
        }
    }
}

impl<InternalReal: RealNumberInternalTrait> std::ops::Add<Number<InternalReal>>
    for Number<InternalReal>
{
    type Output = Number<InternalReal>;
    fn add(self, rhs: Number<InternalReal>) -> Number<InternalReal> {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Integer(a, b) => Number::Integer(a + b),
            NumberBinaryOperand::Real(a, b) => Number::Real(a + b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => {
                Number::Rational(a1 * b2 + a2 * b1, a2 * b2)
            }
        }
    }
}

impl<InternalReal: RealNumberInternalTrait> std::ops::Sub<Number<InternalReal>>
    for Number<InternalReal>
{
    type Output = Number<InternalReal>;
    fn sub(self, rhs: Number<InternalReal>) -> Number<InternalReal> {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Integer(a, b) => Number::Integer(a - b),
            NumberBinaryOperand::Real(a, b) => Number::Real(a - b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => {
                Number::Rational(a1 * b2 - a2 * b1, a2 * b2)
            }
        }
    }
}

impl<InternalReal: RealNumberInternalTrait> std::ops::Mul<Number<InternalReal>>
    for Number<InternalReal>
{
    type Output = Number<InternalReal>;
    fn mul(self, rhs: Number<InternalReal>) -> Number<InternalReal> {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Integer(a, b) => Number::Integer(a * b),
            NumberBinaryOperand::Real(a, b) => Number::Real(a * b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => Number::Rational(a1 * b1, a2 * b2),
        }
    }
}

impl<InternalReal: RealNumberInternalTrait> std::ops::Div<Number<InternalReal>>
    for Number<InternalReal>
{
    type Output = Result<Number<InternalReal>>;
    fn div(self, rhs: Number<InternalReal>) -> Result<Number<InternalReal>> {
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

pub type ArgVec<InternalReal> = SmallVec<[Value<InternalReal>; 4]>;

#[derive(Clone)]
pub struct BuildinProcedure<InternalReal: RealNumberInternalTrait>(
    &'static str,
    fn(ArgVec<InternalReal>) -> Result<Value<InternalReal>>,
);

impl<InternalReal: RealNumberInternalTrait> fmt::Display for BuildinProcedure<InternalReal> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<build-in procedure ({})>", self.0)
    }
}
impl<InternalReal: RealNumberInternalTrait> fmt::Debug for BuildinProcedure<InternalReal> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<InternalReal: RealNumberInternalTrait> PartialEq for BuildinProcedure<InternalReal> {
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0 && self.1 as usize == rhs.1 as usize
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Procedure<InternalReal: RealNumberInternalTrait> {
    User(SchemeProcedure),
    Buildin(BuildinProcedure<InternalReal>),
}

impl<InternalReal: RealNumberInternalTrait> fmt::Display for Procedure<InternalReal> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Procedure::User(procedure) => write!(f, "{}", procedure),
            Procedure::Buildin(fp) => write!(f, "{}", fp),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value<InternalReal: RealNumberInternalTrait> {
    Number(Number<InternalReal>),
    Boolean(bool),
    Character(char),
    String(String),
    Datum(Box<Statement>),
    Procedure(Procedure<InternalReal>),
    Vector(Vec<Value<InternalReal>>),
    Void,
}

impl<InternalReal: RealNumberInternalTrait> fmt::Display for Value<InternalReal> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Number(num) => write!(f, "{}", num),
            Value::Datum(expr) => write!(f, "{}", expr),
            Value::Procedure(p) => write!(f, "{}", p),
            Value::Void => write!(f, "Void"),
            Value::Boolean(true) => write!(f, "#t"),
            Value::Boolean(false) => write!(f, "#f"),
            Value::Character(c) => write!(f, "#\\{}", c),
            Value::String(ref s) => write!(f, "\"{}\"", s),
            Value::Vector(vec) => write!(
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

fn check_division_by_zero(num: i32) -> Result<()> {
    match num {
        0 => logic_error!("division by exact zero"),
        _ => Ok(()),
    }
}

pub struct Interpreter<Real: RealNumberInternalTrait, Environment: IEnvironment<Real>> {
    pub env: Rc<RefCell<Environment>>,
    _marker: PhantomData<Real>,
}

impl<InternalReal: RealNumberInternalTrait, Environment: IEnvironment<InternalReal>>
    Interpreter<InternalReal, Environment>
{
    pub fn new() -> Self {
        Self {
            env: Rc::new(RefCell::new(Environment::new())),
            _marker: PhantomData,
        }
    }

    fn apply_scheme_procedure(
        formals: &Vec<String>,
        internal_definitions: &Vec<Definition>,
        expressions: &Vec<Expression>,
        args: ArgVec<InternalReal>,
        parent_env: &Rc<RefCell<Environment>>,
    ) -> Result<Value<InternalReal>> {
        let child_env = Rc::new(RefCell::new(Environment::child(parent_env.clone())));
        for (param, arg) in formals.iter().zip(args.into_iter()) {
            child_env.borrow_mut().define(param.clone(), arg);
        }
        for Definition(name, expr) in internal_definitions {
            child_env
                .borrow_mut()
                .define(name.clone(), Self::eval_expression(&expr, &child_env)?)
        }
        match expressions.split_last() {
            Some((last, before_last)) => {
                for expr in before_last {
                    Self::eval_expression(expr, &child_env)?;
                }
                Self::eval_expression(last, &child_env)
            }
            None => logic_error!("no expression in function body"),
        }
    }

    pub(self) fn eval_root_expression(
        &self,
        expression: Expression,
    ) -> Result<Value<InternalReal>> {
        Self::eval_expression(&expression, &self.env)
    }

    pub fn apply_procedure(
        procedure: &Procedure<InternalReal>,
        evaluated_args: ArgVec<InternalReal>,
        env: &Rc<RefCell<Environment>>,
    ) -> Result<Value<InternalReal>> {
        Ok(match procedure {
            Procedure::Buildin(BuildinProcedure(_, function_pointer)) => {
                function_pointer(evaluated_args)?
            }
            Procedure::User(SchemeProcedure(formals, definitions, expressions)) => {
                Self::apply_scheme_procedure(
                    &formals,
                    &definitions,
                    &expressions,
                    evaluated_args,
                    env,
                )?
            }
        })
    }

    pub fn eval_expression(
        expression: &Expression,
        env: &Rc<RefCell<Environment>>,
    ) -> Result<Value<InternalReal>> {
        Ok(match expression {
            Expression::ProcedureCall(procedure_expr, arguments) => {
                let first = Self::eval_expression(procedure_expr, env)?;
                let evaluated_args: Result<ArgVec<InternalReal>> = arguments
                    .into_iter()
                    .map(|arg| Self::eval_expression(arg, env))
                    .collect();
                match first {
                    Value::Procedure(procedure) => {
                        Self::apply_procedure(&procedure, evaluated_args?, env)?
                    }
                    _ => logic_error!("expect a procedure here"),
                }
            }
            Expression::Vector(vector) => {
                let mut values = Vec::with_capacity(vector.len());
                for expr in vector {
                    values.push(Self::eval_expression(expr, env)?);
                }
                Value::Vector(values)
            }
            Expression::Character(c) => Value::Character(*c),
            Expression::String(string) => Value::String(string.clone()),
            Expression::Procedure(scheme) => Value::Procedure(Procedure::User(scheme.clone())),
            Expression::Conditional(cond) => {
                let &(test, consequent, alternative) = &cond.as_ref();
                match Self::eval_expression(&test, env)? {
                    Value::Boolean(true) => Self::eval_expression(&consequent, env)?,
                    Value::Boolean(false) => match alternative {
                        Some(alter) => Self::eval_expression(&alter, env)?,
                        None => Value::Void,
                    },
                    _ => logic_error!("if condition should be a boolean expression"),
                }
            }
            Expression::Datum(datum) => Value::Datum(datum.clone()),
            Expression::Boolean(value) => Value::Boolean(*value),
            Expression::Integer(value) => Value::Number(Number::Integer(*value)),
            Expression::Real(number_literal) => Value::Number(Number::Real(
                InternalReal::from(number_literal.parse::<f64>().unwrap()).unwrap(),
            )),
            // TODO: apply gcd here.
            Expression::Rational(a, b) => Value::Number(Number::Rational(*a, *b as i32)),
            Expression::Identifier(ident) => match env.borrow().get(ident.as_str()) {
                Some(value) => value.clone(),
                None => logic_error!("undefined identifier: {}", ident),
            },
        })
    }

    pub fn eval_ast(
        ast: &Statement,
        env: Rc<RefCell<Environment>>,
    ) -> Result<Option<Value<InternalReal>>> {
        Ok(match ast {
            Statement::ImportDeclaration(_) => None, // TODO
            Statement::Expression(expr) => Some(Self::eval_expression(&expr, &env)?),
            Statement::Definition(Definition(name, expr)) => {
                let value = Self::eval_expression(&expr, &env)?;
                env.borrow_mut().define(name.clone(), value);
                None
            }
        })
    }

    pub fn eval_root_ast(&self, ast: &Statement) -> Result<Option<Value<InternalReal>>> {
        Self::eval_ast(ast, self.env.clone())
    }

    pub fn eval_program<'a>(
        &self,
        asts: impl IntoIterator<Item = &'a Statement>,
    ) -> Result<Option<Value<InternalReal>>> {
        asts.into_iter()
            .try_fold(None, |_, ast| self.eval_root_ast(&ast))
    }

    pub fn eval(
        &self,
        char_stream: impl Iterator<Item = char>,
    ) -> Result<Option<Value<InternalReal>>> {
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
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    assert_eq!(
        interpreter.eval_root_expression(Expression::Integer(-1))?,
        Value::Number(Number::Integer(-1))
    );
    assert_eq!(
        interpreter.eval_root_expression(Expression::Rational(1, 3))?,
        Value::Number(Number::Rational(1, 3))
    );
    assert_eq!(
        interpreter.eval_root_expression(Expression::Real("-3.45e-7".to_string()))?,
        Value::Number(Number::Real(-3.45e-7))
    );
    Ok(())
}

#[test]
fn arithmetic() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("+".to_string())),
            vec![Expression::Integer(1), Expression::Integer(2)]
        ))?,
        Value::Number(Number::Integer(3))
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("+".to_string())),
            vec![Expression::Integer(1), Expression::Rational(1, 2)]
        ))?,
        Value::Number(Number::Rational(3, 2))
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("*".to_string())),
            vec![
                Expression::Rational(1, 2),
                Expression::Real("2.0".to_string()),
            ]
        ))?,
        Value::Number(Number::Real(1.0)),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("/".to_string())),
            vec![Expression::Integer(1), Expression::Integer(0)]
        )),
        Err(SchemeError {
            category: ErrorType::Logic,
            message: "division by exact zero".to_string()
        }),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("max".to_string())),
            vec![Expression::Integer(1), Expression::Real("1.3".to_string()),]
        ))?,
        Value::Number(Number::Real(1.3)),
    );
    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("min".to_string())),
            vec![Expression::Integer(1), Expression::Real("1.3".to_string()),]
        ))?,
        Value::Number(Number::Real(1.0)),
    );
    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("min".to_string())),
            vec![Expression::Identifier("+".to_string()),]
        )),
        Err(SchemeError {
            category: ErrorType::Logic,
            message: "expect a number!".to_string()
        }),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("max".to_string())),
            vec![Expression::Identifier("+".to_string())]
        )),
        Err(SchemeError {
            category: ErrorType::Logic,
            message: "expect a number!".to_string()
        }),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("sqrt".to_string())),
            vec![Expression::Integer(4)]
        ))?,
        Value::Number(Number::Real(2.0)),
    );

    match interpreter.eval_root_expression(Expression::ProcedureCall(
        Box::new(Expression::Identifier("sqrt".to_string())),
        vec![Expression::Integer(-4)],
    ))? {
        Value::Number(Number::Real(should_be_nan)) => {
            assert!(num_traits::Float::is_nan(should_be_nan))
        }
        _ => panic!("sqrt result should be a number"),
    }

    for (cmp, result) in [">", "<", ">=", "<=", "="]
        .iter()
        .zip([false, false, true, true, true].iter())
    {
        assert_eq!(
            interpreter.eval_root_expression(Expression::ProcedureCall(
                Box::new(Expression::Identifier(cmp.to_string())),
                vec![
                    Expression::Integer(1),
                    Expression::Rational(1, 1),
                    Expression::Real("1.0".to_string()),
                ],
            ))?,
            Value::Boolean(*result)
        )
    }

    Ok(())
}

#[test]
fn undefined() {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    assert_eq!(
        interpreter.eval_root_expression(Expression::Identifier("foo".to_string())),
        Err(SchemeError {
            category: ErrorType::Logic,
            message: "undefined identifier: foo".to_string(),
        })
    );
}

#[test]
fn variable_definition() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
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
        Some(Value::Number(Number::Integer(1)))
    );
    Ok(())
}

#[test]
fn buildin_procedural() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
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
        Some(Value::Number(Number::Integer(3)))
    );
    Ok(())
}

#[test]
fn procedure_definition() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
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
        Some(Value::Number(Number::Integer(3)))
    );
    Ok(())
}

#[test]
fn lambda_call() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
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
        Some(Value::Number(Number::Integer(3)))
    );
    Ok(())
}

#[test]
fn condition() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    let program = vec![Statement::Expression(Expression::Conditional(Box::new((
        Expression::Boolean(true),
        Expression::Integer(1),
        Some(Expression::Integer(2)),
    ))))];
    assert_eq!(
        interpreter.eval_program(program.iter())?,
        Some(Value::Number(Number::Integer(1)))
    );
    Ok(())
}

#[test]
fn local_environment() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
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
        Some(Value::Number(Number::Integer(3)))
    );
    Ok(())
}

#[test]
fn procedure_as_data() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
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
        Some(Value::Number(Number::Integer(3)))
    );
    Ok(())
}
