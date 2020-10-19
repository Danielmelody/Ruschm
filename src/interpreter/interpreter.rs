#![allow(dead_code)]
use crate::values::ArgVec;
use crate::{
    environment::*, values::BuildinProcedure, values::Number, values::RealNumberInternalTrait,
    values::ValueReference,
};
use crate::{error::*, values::Value};
use crate::{parser::*, values::Procedure};
use std::iter::Iterator;
use std::marker::PhantomData;
use std::rc::Rc;

use super::pair::Pair;
use super::Result;

#[derive(Debug, Clone, PartialEq)]
enum TailExpressionResult<'a, R: RealNumberInternalTrait, E: IEnvironment<R>> {
    TailCall(&'a Expression, &'a [Expression], Rc<E>),
    Value(Value<R, E>),
}

pub struct Interpreter<R: RealNumberInternalTrait, E: IEnvironment<R>> {
    pub env: Rc<E>,
    _marker: PhantomData<R>,
}

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> Interpreter<R, E> {
    pub fn new() -> Self {
        let interpreter = Self {
            env: Rc::new(E::new()),
            _marker: PhantomData,
        };

        interpreter
            .eval(include_str!("./scheme/base.scm").chars())
            .unwrap();
        interpreter
    }

    fn apply_scheme_procedure<'a>(
        formals: &ParameterFormals,
        internal_definitions: &[Definition],
        expressions: &'a [Expression],
        closure: Rc<E>,
        args: ArgVec<R, E>,
    ) -> Result<TailExpressionResult<'a, R, E>> {
        let local_env = Rc::new(E::new_child(closure.clone()));
        let mut arg_iter = args.into_iter();
        for (param, arg) in formals.0.iter().zip(arg_iter.by_ref()) {
            local_env.define(param.clone(), arg);
        }
        if let Some(variadic) = &formals.1 {
            let list = arg_iter.collect();
            local_env.define(variadic.clone(), list);
        }
        for DefinitionBody(name, expr) in internal_definitions.into_iter().map(|d| &d.data) {
            let value = Self::eval_expression(&expr, &local_env)?;
            local_env.define(name.clone(), value)
        }
        match expressions.split_last() {
            Some((last, other)) => {
                for expr in other {
                    Self::eval_expression(&expr, &local_env)?;
                }
                Self::eval_tail_expression(last, local_env)
            }
            None => logic_error!("no expression in function body"),
        }
    }

    pub(self) fn eval_root_expression(&self, expression: Expression) -> Result<Value<R, E>> {
        Self::eval_expression(&expression, &self.env)
    }

    fn eval_procedure_call(
        procedure_expr: &Expression,
        arguments: &[Expression],
        env: &Rc<E>,
    ) -> Result<(Procedure<R, E>, ArgVec<R, E>)> {
        let first = Self::eval_expression(procedure_expr, env)?;
        let evaluated_args_result: Result<ArgVec<R, E>> = arguments
            .iter()
            .map(|arg| Self::eval_expression(arg, env))
            .collect();
        Ok(match first {
            Value::Procedure(procedure) => (procedure, evaluated_args_result?),
            _ => logic_error!("expect a procedure here"),
        })
    }

    pub fn apply_procedure<'a>(
        initial_procedure: &Procedure<R, E>,
        mut args: ArgVec<R, E>,
        env: &Rc<E>,
    ) -> Result<Value<R, E>> {
        let formals = initial_procedure.get_parameters();
        if args.len() < formals.0.len() || (args.len() > formals.0.len() && formals.1.is_none()) {
            logic_error!(
                "expect {}{} arguments, got {}. parameter list is: {}",
                if formals.1.is_some() { "at least " } else { "" },
                formals.0.len(),
                args.len(),
                formals,
            );
        }
        let mut current_procedure = None;
        loop {
            match if current_procedure.is_none() {
                initial_procedure
            } else {
                current_procedure.as_ref().unwrap()
            } {
                Procedure::Buildin(BuildinProcedure { pointer, .. }) => {
                    break pointer.apply(args, env);
                }
                Procedure::User(SchemeProcedure(formals, definitions, expressions), closure) => {
                    let apply_result = Self::apply_scheme_procedure(
                        formals,
                        definitions,
                        expressions,
                        closure.clone(),
                        args,
                    )?;
                    match apply_result {
                        TailExpressionResult::TailCall(
                            tail_procedure_expr,
                            tail_arguments,
                            last_env,
                        ) => {
                            let (tail_procedure, tail_args) = Self::eval_procedure_call(
                                tail_procedure_expr,
                                tail_arguments,
                                &last_env,
                            )?;
                            current_procedure = Some(tail_procedure);
                            args = tail_args;
                        }
                        TailExpressionResult::Value(return_value) => {
                            break Ok(return_value);
                        }
                    };
                }
            };
        }
    }

    fn eval_tail_expression<'a>(
        expression: &'a Expression,
        env: Rc<E>,
    ) -> Result<TailExpressionResult<'a, R, E>> {
        Ok(match &expression.data {
            ExpressionBody::ProcedureCall(procedure_expr, arguments) => {
                TailExpressionResult::TailCall(procedure_expr.as_ref(), arguments, env)
            }
            ExpressionBody::Conditional(cond) => {
                let (test, consequent, alternative) = cond.as_ref();
                match Self::eval_expression(&test, &env)? {
                    Value::Boolean(true) => Self::eval_tail_expression(consequent, env)?,
                    Value::Boolean(false) => match alternative {
                        Some(alter) => Self::eval_tail_expression(alter, env)?,
                        None => TailExpressionResult::Value(Value::Void),
                    },
                    _ => logic_error!("if condition should be a boolean expression"),
                }
            }
            _ => TailExpressionResult::Value(Self::eval_expression(&expression, &env)?),
        })
    }

    pub fn read_literal(expression: &Expression, env: &Rc<E>) -> Result<Value<R, E>> {
        match &expression.data {
            ExpressionBody::Real(_)
            | ExpressionBody::Rational(..)
            | ExpressionBody::Integer(_)
            | ExpressionBody::String(..)
            | ExpressionBody::Character(..) => Self::eval_expression(expression, env),
            ExpressionBody::Identifier(name) => Ok(Value::Symbol(name.clone())),
            ExpressionBody::List(list) => {
                let last_second = list.iter().rev().skip(1).next();
                match last_second {
                    Some(Expression {
                        data: ExpressionBody::Period,
                        location,
                    }) => match list.iter().rev().skip(2).next() {
                        Some(_) => {
                            let mut car = list.iter().rev().skip(2);
                            car.try_fold(
                                Self::read_literal(list.last().unwrap(), env)?,
                                |pair, current| {
                                    Ok(Value::Pair(Box::new(Pair {
                                        car: Self::read_literal(current, env)?,
                                        cdr: pair,
                                    })))
                                },
                            )
                        }
                        None => logic_error_with_location!(*location, "unexpect dot"),
                    },
                    _ => Ok(list
                        .iter()
                        .rev()
                        .map(|i| Self::read_literal(i, env))
                        .collect::<Result<_>>()?),
                }
            }
            ExpressionBody::Vector(vec) => Ok(Value::Vector(ValueReference::new_immutable(
                vec.iter()
                    .map(|i| Self::read_literal(i, env))
                    .collect::<Result<_>>()?,
            ))),
            ExpressionBody::Quote(inner) => Ok(vec![
                Self::read_literal(inner, env)?,
                Value::Symbol("quote".to_string()),
            ]
            .into_iter()
            .collect()),
            o => unreachable!("expression should not be {:?}", o),
        }
    }

    pub fn eval_expression(expression: &Expression, env: &Rc<E>) -> Result<Value<R, E>> {
        Ok(match &expression.data {
            ExpressionBody::ProcedureCall(procedure_expr, arguments) => {
                let first = Self::eval_expression(procedure_expr, env)?;
                let evaluated_args: Result<ArgVec<R, E>> = arguments
                    .into_iter()
                    .map(|arg| Self::eval_expression(arg, env))
                    .collect();
                match first {
                    Value::Procedure(procedure) => {
                        Self::apply_procedure(&procedure, evaluated_args?, env)?
                    }
                    _ => logic_error_with_location!(expression.location, "expect a procedure here"),
                }
            }
            ExpressionBody::List(_) => {
                unreachable!("expression list should be converted to list value")
            }
            ExpressionBody::Vector(_) => Self::read_literal(&expression, env)?,
            ExpressionBody::Character(c) => Value::Character(*c),
            ExpressionBody::String(string) => Value::String(string.clone()),
            ExpressionBody::Period => {
                logic_error_with_location!(expression.location, "unexpect dot")
            }
            ExpressionBody::Assignment(name, value_expr) => {
                let value = Self::eval_expression(value_expr, env)?;
                env.set(name, value)?;
                Value::Void
            }
            ExpressionBody::Procedure(scheme) => {
                Value::Procedure(Procedure::User(scheme.clone(), env.clone()))
            }
            ExpressionBody::Conditional(cond) => {
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
            ExpressionBody::Quote(inner) => Self::read_literal(inner.as_ref(), env)?,
            ExpressionBody::Boolean(value) => Value::Boolean(*value),
            ExpressionBody::Integer(value) => Value::Number(Number::Integer(*value)),
            ExpressionBody::Real(number_literal) => Value::Number(Number::Real(
                R::from(number_literal.parse::<f64>().unwrap()).unwrap(),
            )),
            // TODO: apply gcd here.
            ExpressionBody::Rational(a, b) => Value::Number(Number::Rational(*a, *b as i32)),
            ExpressionBody::Identifier(ident) => match env.get(ident.as_str()) {
                Some(value) => value.clone(),
                None => logic_error_with_location!(
                    expression.location,
                    "undefined identifier: {}",
                    ident
                ),
            },
        })
    }

    pub fn eval_ast(ast: &Statement, env: Rc<E>) -> Result<Option<Value<R, E>>> {
        Ok(match ast {
            Statement::ImportDeclaration(_) => None, // TODO
            Statement::Expression(expr) => Some(Self::eval_expression(&expr, &env)?),
            Statement::Definition(Definition {
                data: DefinitionBody(name, expr),
                ..
            }) => {
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
            let lexer = Lexer::from_char_stream(char_stream);
            let mut parser = Parser::from_lexer(lexer);
            parser.try_fold(None, |_, statement| self.eval_root_ast(&statement?))
        }
    }
}

#[test]
fn number() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    assert_eq!(
        interpreter.eval_root_expression(Expression::from_data(ExpressionBody::Integer(-1)))?,
        Value::Number(Number::Integer(-1))
    );
    assert_eq!(
        interpreter.eval_root_expression(Expression::from_data(ExpressionBody::Rational(1, 3)))?,
        Value::Number(Number::Rational(1, 3))
    );
    assert_eq!(
        interpreter.eval_root_expression(Expression::from_data(ExpressionBody::Real(
            "-3.45e-7".to_string()
        )))?,
        Value::Number(Number::Real(-3.45e-7))
    );
    Ok(())
}

#[test]
fn arithmetic() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    assert_eq!(
        interpreter.eval_root_expression(Expression::from_data(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "+".to_string()
            ))),
            vec![
                Expression::from_data(ExpressionBody::Integer(1)),
                Expression::from_data(ExpressionBody::Integer(2))
            ]
        )))?,
        Value::Number(Number::Integer(3))
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::from_data(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "+".to_string()
            ))),
            vec![
                Expression::from_data(ExpressionBody::Integer(1)),
                Expression::from_data(ExpressionBody::Rational(1, 2))
            ]
        )))?,
        Value::Number(Number::Rational(3, 2))
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::from_data(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "*".to_string()
            ))),
            vec![
                Expression::from_data(ExpressionBody::Rational(1, 2)),
                Expression::from_data(ExpressionBody::Real("2.0".to_string())),
            ]
        )))?,
        Value::Number(Number::Real(1.0)),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::from_data(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "/".to_string()
            ))),
            vec![
                Expression::from_data(ExpressionBody::Integer(1)),
                Expression::from_data(ExpressionBody::Integer(0))
            ]
        ))),
        Err(SchemeError {
            category: ErrorType::Logic,
            message: "division by exact zero".to_string(),
            location: None
        }),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::from_data(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "max".to_string()
            ))),
            vec![
                Expression::from_data(ExpressionBody::Integer(1)),
                Expression::from_data(ExpressionBody::Real("1.3".to_string())),
            ]
        )))?,
        Value::Number(Number::Real(1.3)),
    );
    assert_eq!(
        interpreter.eval_root_expression(Expression::from_data(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "min".to_string()
            ))),
            vec![
                Expression::from_data(ExpressionBody::Integer(1)),
                Expression::from_data(ExpressionBody::Real("1.3".to_string())),
            ]
        )))?,
        Value::Number(Number::Real(1.0)),
    );
    assert_eq!(
        interpreter.eval_root_expression(Expression::from_data(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "min".to_string()
            ))),
            vec![Expression::from_data(ExpressionBody::String(
                "a".to_string()
            ))]
        ))),
        Err(SchemeError {
            category: ErrorType::Logic,
            message: "expect a number, got \"a\"".to_string(),
            location: None
        }),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::from_data(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "max".to_string()
            ))),
            vec![Expression::from_data(ExpressionBody::String(
                "a".to_string()
            ))]
        ))),
        Err(SchemeError {
            category: ErrorType::Logic,
            message: "expect a number, got \"a\"".to_string(),
            location: None
        }),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::from_data(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "sqrt".to_string()
            ))),
            vec![Expression::from_data(ExpressionBody::Integer(4))]
        )))?,
        Value::Number(Number::Real(2.0)),
    );

    match interpreter.eval_root_expression(Expression::from_data(ExpressionBody::ProcedureCall(
        Box::new(Expression::from_data(ExpressionBody::Identifier(
            "sqrt".to_string(),
        ))),
        vec![Expression::from_data(ExpressionBody::Integer(-4))],
    )))? {
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
            interpreter.eval_root_expression(Expression::from_data(
                ExpressionBody::ProcedureCall(
                    Box::new(Expression::from_data(ExpressionBody::Identifier(
                        cmp.to_string()
                    ))),
                    vec![
                        Expression::from_data(ExpressionBody::Integer(1)),
                        Expression::from_data(ExpressionBody::Rational(1, 1)),
                        Expression::from_data(ExpressionBody::Real("1.0".to_string())),
                    ],
                )
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
        interpreter.eval_root_expression(Expression::from_data(ExpressionBody::Identifier(
            "foo".to_string()
        ))),
        Err(SchemeError {
            category: ErrorType::Logic,
            message: "undefined identifier: foo".to_string(),
            location: None
        })
    );
}

#[test]
fn variable_definition() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    let program = vec![
        Statement::Definition(Definition::from_data(DefinitionBody(
            "a".to_string(),
            Expression::from_data(ExpressionBody::Integer(1)),
        ))),
        Statement::Definition(Definition::from_data(DefinitionBody(
            "b".to_string(),
            Expression::from_data(ExpressionBody::Identifier("a".to_string())),
        ))),
        Statement::Expression(Expression::from_data(ExpressionBody::Identifier(
            "b".to_string(),
        ))),
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
        Statement::Definition(Definition::from_data(DefinitionBody(
            "a".to_string(),
            Expression::from_data(ExpressionBody::Integer(1)),
        ))),
        Statement::Expression(Expression::from_data(ExpressionBody::Assignment(
            "a".to_string(),
            Box::new(Expression::from_data(ExpressionBody::Integer(2))),
        ))),
        Statement::Expression(Expression::from_data(ExpressionBody::Identifier(
            "a".to_string(),
        ))),
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
        Statement::Definition(Definition::from_data(DefinitionBody(
            "get-add".to_string(),
            simple_procedure(
                ParameterFormals::new(),
                Expression::from_data(ExpressionBody::Identifier("+".to_string())),
            ),
        ))),
        Statement::Expression(Expression::from_data(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::ProcedureCall(
                Box::new(Expression::from_data(ExpressionBody::Identifier(
                    "get-add".to_string(),
                ))),
                vec![],
            ))),
            vec![
                Expression::from_data(ExpressionBody::Integer(1)),
                Expression::from_data(ExpressionBody::Integer(2)),
            ],
        ))),
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
        Statement::Definition(Definition::from_data(DefinitionBody(
            "add".to_string(),
            simple_procedure(
                ParameterFormals(vec!["x".to_string(), "y".to_string()], None),
                Expression::from_data(ExpressionBody::ProcedureCall(
                    Box::new(Expression::from_data(ExpressionBody::Identifier(
                        "+".to_string(),
                    ))),
                    vec![
                        Expression::from_data(ExpressionBody::Identifier("x".to_string())),
                        Expression::from_data(ExpressionBody::Identifier("y".to_string())),
                    ],
                )),
            ),
        ))),
        Statement::Expression(Expression::from_data(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "add".to_string(),
            ))),
            vec![
                Expression::from_data(ExpressionBody::Integer(1)),
                Expression::from_data(ExpressionBody::Integer(2)),
            ],
        ))),
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
    let program = vec![Statement::Expression(Expression::from_data(
        ExpressionBody::ProcedureCall(
            Box::new(simple_procedure(
                ParameterFormals(
                    vec!["x".to_string(), "y".to_string()],
                    Some("z".to_string()),
                ),
                Expression::from_data(ExpressionBody::ProcedureCall(
                    Box::new(Expression::from_data(ExpressionBody::Identifier(
                        "+".to_string(),
                    ))),
                    vec![
                        Expression::from_data(ExpressionBody::Identifier("x".to_string())),
                        Expression::from_data(ExpressionBody::Identifier("y".to_string())),
                    ],
                )),
            )),
            vec![
                Expression::from_data(ExpressionBody::Integer(1)),
                Expression::from_data(ExpressionBody::Integer(2)),
                Expression::from_data(ExpressionBody::String("something-else".to_string())),
            ],
        ),
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
        Statement::Definition(Definition::from_data(DefinitionBody(
            "counter-creator".to_string(),
            Expression::from_data(ExpressionBody::Procedure(SchemeProcedure(
                ParameterFormals::new(),
                vec![Definition::from_data(DefinitionBody(
                    "current".to_string(),
                    Expression::from_data(ExpressionBody::Integer(0)),
                ))],
                vec![Expression::from_data(ExpressionBody::Procedure(
                    SchemeProcedure(
                        ParameterFormals::new(),
                        vec![],
                        vec![
                            Expression::from_data(ExpressionBody::Assignment(
                                "current".to_string(),
                                Box::new(Expression::from_data(ExpressionBody::ProcedureCall(
                                    Box::new(Expression::from_data(ExpressionBody::Identifier(
                                        "+".to_string(),
                                    ))),
                                    vec![
                                        Expression::from_data(ExpressionBody::Identifier(
                                            "current".to_string(),
                                        )),
                                        Expression::from_data(ExpressionBody::Integer(1)),
                                    ],
                                ))),
                            )),
                            Expression::from_data(ExpressionBody::Identifier(
                                "current".to_string(),
                            )),
                        ],
                    ),
                ))],
            ))),
        ))),
        Statement::Definition(Definition::from_data(DefinitionBody(
            "counter".to_string(),
            Expression::from_data(ExpressionBody::ProcedureCall(
                Box::new(Expression::from_data(ExpressionBody::Identifier(
                    "counter-creator".to_string(),
                ))),
                vec![],
            )),
        ))),
        Statement::Expression(Expression::from_data(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "counter".to_string(),
            ))),
            vec![],
        ))),
        Statement::Expression(Expression::from_data(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "counter".to_string(),
            ))),
            vec![],
        ))),
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
    assert_eq!(
        interpreter.eval_program(
            vec![Statement::Expression(Expression::from_data(
                ExpressionBody::Conditional(Box::new((
                    Expression::from_data(ExpressionBody::Boolean(true)),
                    Expression::from_data(ExpressionBody::Integer(1)),
                    Some(Expression::from_data(ExpressionBody::Integer(2))),
                ))),
            ))]
            .iter()
        )?,
        Some(Value::Number(Number::Integer(1)))
    );
    assert_eq!(
        interpreter.eval_program(
            vec![Statement::Expression(Expression::from_data(
                ExpressionBody::Conditional(Box::new((
                    Expression::from_data(ExpressionBody::Boolean(false)),
                    Expression::from_data(ExpressionBody::Integer(1)),
                    Some(Expression::from_data(ExpressionBody::Integer(2))),
                ))),
            ))]
            .iter()
        )?,
        Some(Value::Number(Number::Integer(2)))
    );
    Ok(())
}

#[test]
fn local_environment() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    let program = vec![
        Statement::Definition(Definition::from_data(DefinitionBody(
            "adda".to_string(),
            simple_procedure(
                ParameterFormals(vec!["x".to_string()], None),
                Expression::from_data(ExpressionBody::ProcedureCall(
                    Box::new(Expression::from_data(ExpressionBody::Identifier(
                        "+".to_string(),
                    ))),
                    vec![
                        Expression::from_data(ExpressionBody::Identifier("x".to_string())),
                        Expression::from_data(ExpressionBody::Identifier("a".to_string())),
                    ],
                )),
            ),
        ))),
        Statement::Definition(Definition::from_data(DefinitionBody(
            "a".to_string(),
            Expression::from_data(ExpressionBody::Integer(1)),
        ))),
        Statement::Expression(Expression::from_data(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "adda".to_string(),
            ))),
            vec![Expression::from_data(ExpressionBody::Integer(2))],
        ))),
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
        Statement::Definition(Definition::from_data(DefinitionBody(
            "add".to_string(),
            simple_procedure(
                ParameterFormals(vec!["x".to_string(), "y".to_string()], None),
                Expression::from_data(ExpressionBody::ProcedureCall(
                    Box::new(Expression::from_data(ExpressionBody::Identifier(
                        "+".to_string(),
                    ))),
                    vec![
                        Expression::from_data(ExpressionBody::Identifier("x".to_string())),
                        Expression::from_data(ExpressionBody::Identifier("y".to_string())),
                    ],
                )),
            ),
        ))),
        Statement::Definition(Definition::from_data(DefinitionBody(
            "apply-op".to_string(),
            simple_procedure(
                ParameterFormals(
                    vec!["op".to_string(), "x".to_string(), "y".to_string()],
                    None,
                ),
                Expression::from_data(ExpressionBody::ProcedureCall(
                    Box::new(Expression::from_data(ExpressionBody::Identifier(
                        "op".to_string(),
                    ))),
                    vec![
                        Expression::from_data(ExpressionBody::Identifier("x".to_string())),
                        Expression::from_data(ExpressionBody::Identifier("y".to_string())),
                    ],
                )),
            ),
        ))),
        Statement::Expression(Expression::from_data(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "apply-op".to_string(),
            ))),
            vec![
                Expression::from_data(ExpressionBody::Identifier("add".to_string())),
                Expression::from_data(ExpressionBody::Integer(1)),
                Expression::from_data(ExpressionBody::Integer(2)),
            ],
        ))),
    ];
    assert_eq!(
        interpreter.eval_program(program.iter())?,
        Some(Value::Number(Number::Integer(3)))
    );
    Ok(())
}

// #[test]
// fn eval_tail_expression() -> Result<()> {
//     let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
//     {
//         let expression = Expression::from_data(ExpressionBody::Integer(3));
//         assert_eq!(
//             Interpreter::eval_tail_expression(&expression, interpreter.env.clone())?,
//             TailExpressionResult::Value(Value::Number(Number::Integer(3)))
//         );
//     }
//     {
//         let expression = Expression::from_data(ExpressionBody::ProcedureCall(
//             Box::new(Expression::from_data(ExpressionBody::Identifier(
//                 "+".to_string(),
//             ))),
//             vec![
//                 Expression::from_data(ExpressionBody::Integer(2)),
//                 Expression::from_data(ExpressionBody::Integer(5)),
//             ],
//         ));
//         assert_eq!(
//             Interpreter::eval_tail_expression(&expression, interpreter.env.clone())?,
//             TailExpressionResult::TailCall(
//                 &Expression::from_data(ExpressionBody::Identifier("+".to_string())),
//                 &convert_located(vec![ExpressionBody::Integer(2), ExpressionBody::Integer(5)]),
//                 interpreter.env.clone()
//             )
//         );
//     }
//     {
//         let expression = Expression::from_data(ExpressionBody::Conditional(Box::new((
//             Expression::from_data(ExpressionBody::Boolean(true)),
//             Expression::from_data(ExpressionBody::ProcedureCall(
//                 Box::new(Expression::from_data(ExpressionBody::Identifier(
//                     "+".to_string(),
//                 ))),
//                 convert_located(vec![ExpressionBody::Integer(2), ExpressionBody::Integer(5)]),
//             )),
//             None,
//         ))));
//         assert_eq!(
//             Interpreter::eval_tail_expression(&expression, interpreter.env.clone())?,
//             TailExpressionResult::TailCall(
//                 &Expression::from_data(ExpressionBody::Identifier("+".to_string())),
//                 &convert_located(vec![ExpressionBody::Integer(2), ExpressionBody::Integer(5)]),
//                 interpreter.env.clone()
//             )
//         );
//     }
//     {
//         let expression = Expression::from_data(ExpressionBody::Conditional(Box::new((
//             Expression::from_data(ExpressionBody::Boolean(false)),
//             Expression::from_data(ExpressionBody::ProcedureCall(
//                 Box::new(Expression::from_data(ExpressionBody::Identifier(
//                     "+".to_string(),
//                 ))),
//                 convert_located(vec![ExpressionBody::Integer(2), ExpressionBody::Integer(5)]),
//             )),
//             Some(Expression::from_data(ExpressionBody::Integer(4))),
//         ))));
//         assert_eq!(
//             Interpreter::eval_tail_expression(&expression, interpreter.env.clone())?,
//             TailExpressionResult::Value(Value::Number(Number::Integer(4)))
//         );
//     }
//     {
//         let expression = Expression::from_data(ExpressionBody::Conditional(Box::new((
//             Expression::from_data(ExpressionBody::Boolean(false)),
//             Expression::from_data(ExpressionBody::Integer(4)),
//             Some(Expression::from_data(ExpressionBody::ProcedureCall(
//                 Box::new(Expression::from_data(ExpressionBody::Identifier(
//                     "+".to_string(),
//                 ))),
//                 convert_located(vec![ExpressionBody::Integer(2), ExpressionBody::Integer(5)]),
//             ))),
//         ))));
//         assert_eq!(
//             Interpreter::eval_tail_expression(&expression, interpreter.env.clone())?,
//             TailExpressionResult::TailCall(
//                 &Expression::from_data(ExpressionBody::Identifier("+".to_string())),
//                 &convert_located(vec![ExpressionBody::Integer(2), ExpressionBody::Integer(5)]),
//                 interpreter.env.clone()
//             )
//         );
//     }
//     Ok(())
// }

#[test]
fn datum_literal() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    assert_eq!(
        Interpreter::eval_expression(
            &l(ExpressionBody::Quote(Box::new(l(ExpressionBody::Integer(
                1,
            ))))),
            &interpreter.env,
        )?,
        Value::Number(Number::Integer(1))
    );
    assert_eq!(
        Interpreter::eval_expression(
            &l(ExpressionBody::Quote(Box::new(l(
                ExpressionBody::Identifier("a".to_string())
            )))),
            &interpreter.env,
        )?,
        Value::Symbol("a".to_string())
    );
    assert_eq!(
        Interpreter::eval_expression(
            &l(ExpressionBody::Quote(Box::new(l(ExpressionBody::List(
                vec![l(ExpressionBody::Integer(1),)]
            ))))),
            &interpreter.env,
        )?,
        Value::Pair(Box::new(Pair {
            car: Value::Number(Number::Integer(1)),
            cdr: Value::EmptyList
        }))
    );
    assert_eq!(
        Interpreter::eval_expression(
            &l(ExpressionBody::Quote(Box::new(l(ExpressionBody::Vector(
                vec![l(ExpressionBody::Identifier("a".to_string()),)]
            ))))),
            &interpreter.env,
        )?,
        Value::Vector(ValueReference::new_immutable(vec![Value::Symbol(
            "a".to_string()
        )]))
    );
    Ok(())
}
