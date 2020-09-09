#![allow(dead_code)]
use crate::environment::*;
use crate::error::*;
use crate::lexer::*;
use crate::parser::*;
use num_traits::real::Real;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::fmt;
use std::iter::Iterator;
use std::marker::PhantomData;
use std::rc::Rc;

type Result<T> = std::result::Result<T, SchemeError>;

pub mod scheme;

pub trait RealNumberInternalTrait: fmt::Display + fmt::Debug + Real
where
    Self: std::marker::Sized,
{
}

impl<T: fmt::Display + fmt::Debug + Real> RealNumberInternalTrait for T {}
#[derive(Debug, Clone, Copy)]
pub enum Number<R: RealNumberInternalTrait> {
    Integer(i32),
    Real(R),
    Rational(i32, i32),
    // _marker: PhantomData<E>,
}

impl<R: RealNumberInternalTrait> fmt::Display for Number<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Number::Integer(n) => write!(f, "{}", n),
            Number::Real(n) => write!(f, "{:?}", n),
            Number::Rational(a, b) => write!(f, "{}/{}", a, b),
        }
    }
}

impl<R: RealNumberInternalTrait> PartialEq for Number<R> {
    fn eq(&self, other: &Number<R>) -> bool {
        match upcast_oprands((*self, *other)) {
            NumberBinaryOperand::Integer(a, b) => a.eq(&b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => (a1 * b2).eq(&(b1 * a2)),
            NumberBinaryOperand::Real(a, b) => a.eq(&b),
        }
    }
}

impl<R: RealNumberInternalTrait> PartialOrd for Number<R> {
    fn partial_cmp(&self, other: &Number<R>) -> Option<Ordering> {
        match upcast_oprands((*self, *other)) {
            NumberBinaryOperand::Integer(a, b) => a.partial_cmp(&b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => (a1 * b2).partial_cmp(&(b1 * a2)),
            NumberBinaryOperand::Real(a, b) => a.partial_cmp(&b),
        }
    }
}

enum NumberBinaryOperand<R: RealNumberInternalTrait> {
    Integer(i32, i32),
    Real(R, R),
    Rational(i32, i32, i32, i32),
}

// Integer => Rational => Real
fn upcast_oprands<R: RealNumberInternalTrait>(
    operand: (Number<R>, Number<R>),
) -> NumberBinaryOperand<R> {
    match operand {
        (Number::Rational(dividend, dividor), Number::Real(b)) => {
            NumberBinaryOperand::Real(R::from(dividend).unwrap() / R::from(dividor).unwrap(), b)
        }
        (Number::Real(a), Number::Rational(dividend, dividor)) => {
            NumberBinaryOperand::Real(a, R::from(dividend).unwrap() / R::from(dividor).unwrap())
        }
        (Number::Integer(a), Number::Real(b)) => NumberBinaryOperand::Real(R::from(a).unwrap(), b),
        (Number::Real(a), Number::Integer(b)) => NumberBinaryOperand::Real(a, R::from(b).unwrap()),
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

impl<R: RealNumberInternalTrait> NumberBinaryOperand<R> {
    pub fn lhs(&self) -> Number<R> {
        match self {
            NumberBinaryOperand::Integer(a, _) => Number::Integer(*a),
            NumberBinaryOperand::Real(a, _) => Number::Real(*a),
            NumberBinaryOperand::Rational(a1, a2, _, _) => Number::Rational(*a1, *a2),
        }
    }

    pub fn rhs(&self) -> Number<R> {
        match self {
            NumberBinaryOperand::Integer(_, b) => Number::Integer(*b),
            NumberBinaryOperand::Real(_, b) => Number::Real(*b),
            NumberBinaryOperand::Rational(_, _, b1, b2) => Number::Rational(*b1, *b2),
        }
    }
}

impl<R: RealNumberInternalTrait> std::ops::Add<Number<R>> for Number<R> {
    type Output = Number<R>;
    fn add(self, rhs: Number<R>) -> Number<R> {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Integer(a, b) => Number::Integer(a + b),
            NumberBinaryOperand::Real(a, b) => Number::Real(a + b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => {
                Number::Rational(a1 * b2 + a2 * b1, a2 * b2)
            }
        }
    }
}

impl<R: RealNumberInternalTrait> std::ops::Sub<Number<R>> for Number<R> {
    type Output = Number<R>;
    fn sub(self, rhs: Number<R>) -> Number<R> {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Integer(a, b) => Number::Integer(a - b),
            NumberBinaryOperand::Real(a, b) => Number::Real(a - b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => {
                Number::Rational(a1 * b2 - a2 * b1, a2 * b2)
            }
        }
    }
}

impl<R: RealNumberInternalTrait> std::ops::Mul<Number<R>> for Number<R> {
    type Output = Number<R>;
    fn mul(self, rhs: Number<R>) -> Number<R> {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Integer(a, b) => Number::Integer(a * b),
            NumberBinaryOperand::Real(a, b) => Number::Real(a * b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => Number::Rational(a1 * b1, a2 * b2),
        }
    }
}

impl<R: RealNumberInternalTrait> std::ops::Div<Number<R>> for Number<R> {
    type Output = Result<Number<R>>;
    fn div(self, rhs: Number<R>) -> Result<Number<R>> {
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

pub type ArgVec<R, E> = SmallVec<[Value<R, E>; 4]>;

#[derive(Clone)]
pub struct BuildinProcedure<R: RealNumberInternalTrait, E: IEnvironment<R>> {
    pub name: &'static str,
    pub parameter_length: Option<usize>,
    pub pointer: fn(ArgVec<R, E>) -> Result<Value<R, E>>,
}

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> fmt::Display for BuildinProcedure<R, E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<build-in procedure ({})>", self.name)
    }
}
impl<R: RealNumberInternalTrait, E: IEnvironment<R>> fmt::Debug for BuildinProcedure<R, E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> PartialEq for BuildinProcedure<R, E> {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
            && self.parameter_length == rhs.parameter_length
            && self.pointer as usize == rhs.pointer as usize
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Procedure<R: RealNumberInternalTrait, E: IEnvironment<R>> {
    User(SchemeProcedure, Rc<E>),
    Buildin(BuildinProcedure<R, E>),
}

// impl<R: RealNumberInternalTrait, E: IEnvironment<R>> PartialEq for Procedure<R, E> {
//     fn eq(&self, other: &Self) -> bool {
//         self.body == other.body
//     }
// }

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> Procedure<R, E> {
    pub fn get_parameter_length(&self) -> Option<usize> {
        match &self {
            Procedure::User(user, ..) => Some(user.0.len()),
            Procedure::Buildin(buildin) => buildin.parameter_length,
        }
    }
}

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> fmt::Display for Procedure<R, E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            Procedure::User(procedure, ..) => write!(f, "{}", procedure),
            Procedure::Buildin(fp) => write!(f, "{}", fp),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value<R: RealNumberInternalTrait, E: IEnvironment<R>> {
    Number(Number<R>),
    Boolean(bool),
    Character(char),
    String(String),
    Datum(Box<Statement>),
    Procedure(Procedure<R, E>),
    Vector(Vec<Value<R, E>>),
    Void,
}

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> fmt::Display for Value<R, E> {
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

#[derive(Debug, Clone, PartialEq)]
enum TailExpressionResult<R: RealNumberInternalTrait, E: IEnvironment<R>> {
    TailCall(Expression, Vec<Expression>, Rc<E>),
    Value(Value<R, E>),
}

pub struct Interpreter<R: RealNumberInternalTrait, E: IEnvironment<R>> {
    pub env: Rc<E>,
    _marker: PhantomData<R>,
}

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> Interpreter<R, E> {
    pub fn new() -> Self {
        Self {
            env: Rc::new(E::new()),
            _marker: PhantomData,
        }
    }

    fn apply_scheme_procedure(
        formals: Vec<String>,
        internal_definitions: Vec<Definition>,
        mut expressions: Vec<Expression>,
        closure: Rc<E>,
        args: ArgVec<R, E>,
    ) -> Result<TailExpressionResult<R, E>> {
        let local_env = Rc::new(E::new_child(closure.clone()));
        for (param, arg) in formals.iter().zip(args.into_iter()) {
            local_env.define(param.clone(), arg);
        }
        for Definition(name, expr) in internal_definitions {
            let value = Self::eval_expression(&expr, &local_env)?;
            local_env.define(name.clone(), value)
        }
        let last = expressions.pop();
        for expr in expressions {
            Self::eval_expression(&expr, &local_env)?;
        }
        match last {
            Some(last) => Ok(Self::eval_tail_expression(last, local_env)?),
            None => logic_error!("no expression in function body"),
        }
    }

    pub(self) fn eval_root_expression(&self, expression: Expression) -> Result<Value<R, E>> {
        Self::eval_expression(&expression, &self.env)
    }

    #[allow(unused_assignments)]
    pub fn apply_procedure<'a>(
        initial_procedure_expr: &Expression,
        initial_arguments: &[Expression],
        initial_env: &Rc<E>,
    ) -> Result<Value<R, E>> {
        let mut procedure_expr = initial_procedure_expr;
        let mut current_procedure_expr = None;
        let mut arguments = initial_arguments;
        let mut current_arguments = None;
        let mut env = initial_env;
        let mut current_env: Option<Rc<E>> = None;
        loop {
            let first = Self::eval_expression(procedure_expr, env)?;
            let evaluated_args_result: Result<ArgVec<R, E>> = arguments
                .iter()
                .map(|arg| Self::eval_expression(arg, env))
                .collect();
            let (procedure, evaluated_args) = match first {
                Value::Procedure(procedure) => (procedure, evaluated_args_result?),
                _ => logic_error!("expect a procedure here"),
            };
            match procedure {
                Procedure::Buildin(BuildinProcedure { pointer, .. }) => {
                    break pointer(evaluated_args)
                }
                Procedure::User(SchemeProcedure(formals, definitions, expressions), closure) => {
                    let apply_result = Self::apply_scheme_procedure(
                        formals,
                        definitions,
                        expressions,
                        closure,
                        evaluated_args,
                    )?;
                    match apply_result {
                        TailExpressionResult::TailCall(
                            tail_procedure_expr,
                            tail_arguments,
                            last_env,
                        ) => {
                            current_procedure_expr = Some(tail_procedure_expr);
                            procedure_expr = current_procedure_expr.as_ref().unwrap();
                            current_arguments = Some(tail_arguments);
                            arguments = current_arguments.as_ref().unwrap();
                            current_env = Some(last_env);
                            env = current_env.as_ref().unwrap();
                        }
                        TailExpressionResult::Value(return_value) => {
                            break Ok(return_value);
                        }
                    };
                }
            };
        }
    }

    fn eval_tail_expression(
        expression: Expression,
        env: Rc<E>,
    ) -> Result<TailExpressionResult<R, E>> {
        Ok(match expression {
            Expression::ProcedureCall(procedure_expr, arguments) => {
                TailExpressionResult::TailCall(*procedure_expr, arguments, env)
            }
            Expression::Conditional(cond) => {
                let (test, consequent, alternative) = *cond;
                match Self::eval_expression(&test, &env)? {
                    Value::Boolean(true) => Self::eval_tail_expression(consequent, env)?,
                    Value::Boolean(false) => match alternative {
                        Some(alter) => Self::eval_tail_expression(alter, env)?,
                        None => TailExpressionResult::Value(Value::Void),
                    },
                    _ => logic_error!("if condition should be a boolean expression"),
                }
            }
            other => TailExpressionResult::Value(Self::eval_expression(&other, &env)?),
        })
    }

    pub fn eval_expression(expression: &Expression, env: &Rc<E>) -> Result<Value<R, E>> {
        Ok(match expression {
            Expression::ProcedureCall(procedure_expr, arguments) => {
                Self::apply_procedure(procedure_expr, arguments, env)?
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
            Expression::Assignment(name, value_expr) => {
                let value = Self::eval_expression(value_expr, env)?;
                env.set(name, value)?;
                Value::Void
            }
            Expression::Procedure(scheme) => {
                Value::Procedure(Procedure::User(scheme.clone(), env.clone()))
            }
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
                R::from(number_literal.parse::<f64>().unwrap()).unwrap(),
            )),
            // TODO: apply gcd here.
            Expression::Rational(a, b) => Value::Number(Number::Rational(*a, *b as i32)),
            Expression::Identifier(ident) => match env.get(ident.as_str()) {
                Some(value) => value.clone(),
                None => logic_error!("undefined identifier: {}", ident),
            },
        })
    }

    pub fn eval_ast(ast: &Statement, env: Rc<E>) -> Result<Option<Value<R, E>>> {
        Ok(match ast {
            Statement::ImportDeclaration(_) => None, // TODO
            Statement::Expression(expr) => Some(Self::eval_expression(&expr, &env)?),
            Statement::Definition(Definition(name, expr)) => {
                let value = Self::eval_expression(&expr, &env)?;
                env.define(name.clone(), value);
                None
            }
        })
    }

    pub fn eval_root_ast(&self, ast: &Statement) -> Result<Option<Value<R, E>>> {
        Self::eval_ast(ast, self.env.clone())
    }

    pub fn eval_program<'a>(
        &self,
        asts: impl IntoIterator<Item = &'a Statement>,
    ) -> Result<Option<Value<R, E>>> {
        asts.into_iter()
            .try_fold(None, |_, ast| self.eval_root_ast(&ast))
    }

    pub fn eval(&self, char_stream: impl Iterator<Item = char>) -> Result<Option<Value<R, E>>> {
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
fn variable_assignment() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    let program = vec![
        Statement::Definition(Definition("a".to_string(), Expression::Integer(1))),
        Statement::Expression(Expression::Assignment(
            "a".to_string(),
            Box::new(Expression::Integer(2)),
        )),
        Statement::Expression(Expression::Identifier("a".to_string())),
    ];
    assert_eq!(
        interpreter.eval_program(program.iter())?,
        Some(Value::Number(Number::Integer(2)))
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
fn closure() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    let program = vec![
        Statement::Definition(Definition(
            "counter-creator".to_string(),
            Expression::Procedure(SchemeProcedure(
                vec![],
                vec![Definition("current".to_string(), Expression::Integer(0))],
                vec![Expression::Procedure(SchemeProcedure(
                    vec![],
                    vec![],
                    vec![
                        Expression::Assignment(
                            "current".to_string(),
                            Box::new(Expression::ProcedureCall(
                                Box::new(Expression::Identifier("+".to_string())),
                                vec![
                                    Expression::Identifier("current".to_string()),
                                    Expression::Integer(1),
                                ],
                            )),
                        ),
                        Expression::Identifier("current".to_string()),
                    ],
                ))],
            )),
        )),
        Statement::Definition(Definition(
            "counter".to_string(),
            Expression::ProcedureCall(
                Box::new(Expression::Identifier("counter-creator".to_string())),
                vec![],
            ),
        )),
        Statement::Expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("counter".to_string())),
            vec![],
        )),
        Statement::Expression(Expression::ProcedureCall(
            Box::new(Expression::Identifier("counter".to_string())),
            vec![],
        )),
    ];
    assert_eq!(
        interpreter.eval_program(program.iter())?,
        Some(Value::Number(Number::Integer(2)))
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

#[test]
fn eval_tail_expression() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    {
        let expression = Expression::Integer(3);
        assert_eq!(
            Interpreter::eval_tail_expression(expression, interpreter.env.clone())?,
            TailExpressionResult::Value(Value::Number(Number::Integer(3)))
        );
    }
    {
        let expression = Expression::ProcedureCall(
            Box::new(Expression::Identifier("+".to_string())),
            vec![Expression::Integer(2), Expression::Integer(5)],
        );
        assert_eq!(
            Interpreter::eval_tail_expression(expression, interpreter.env.clone())?,
            TailExpressionResult::TailCall(
                Expression::Identifier("+".to_string()),
                vec![Expression::Integer(2), Expression::Integer(5)],
                interpreter.env.clone()
            )
        );
    }
    {
        let expression = Expression::Conditional(Box::new((
            Expression::Boolean(true),
            Expression::ProcedureCall(
                Box::new(Expression::Identifier("+".to_string())),
                vec![Expression::Integer(2), Expression::Integer(5)],
            ),
            None,
        )));
        assert_eq!(
            Interpreter::eval_tail_expression(expression, interpreter.env.clone())?,
            TailExpressionResult::TailCall(
                Expression::Identifier("+".to_string()),
                vec![Expression::Integer(2), Expression::Integer(5)],
                interpreter.env.clone()
            )
        );
    }
    {
        let expression = Expression::Conditional(Box::new((
            Expression::Boolean(false),
            Expression::ProcedureCall(
                Box::new(Expression::Identifier("+".to_string())),
                vec![Expression::Integer(2), Expression::Integer(5)],
            ),
            Some(Expression::Integer(4)),
        )));
        assert_eq!(
            Interpreter::eval_tail_expression(expression, interpreter.env.clone())?,
            TailExpressionResult::Value(Value::Number(Number::Integer(4)))
        );
    }
    {
        let expression = Expression::Conditional(Box::new((
            Expression::Boolean(false),
            Expression::Integer(4),
            Some(Expression::ProcedureCall(
                Box::new(Expression::Identifier("+".to_string())),
                vec![Expression::Integer(2), Expression::Integer(5)],
            )),
        )));
        assert_eq!(
            Interpreter::eval_tail_expression(expression, interpreter.env.clone())?,
            TailExpressionResult::TailCall(
                Expression::Identifier("+".to_string()),
                vec![Expression::Integer(2), Expression::Integer(5)],
                interpreter.env.clone()
            )
        );
    }
    Ok(())
}
