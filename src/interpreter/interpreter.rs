#![allow(dead_code)]

#[cfg(test)]
use crate::values::{Transformer, Type};
use crate::{
    environment::*, library_name, values::BuiltinProcedure, values::Number,
    values::RealNumberInternalTrait, values::ValueReference,
};
use crate::{error::*, values::Value};
use crate::{file::file_char_stream, values::ArgVec};
use crate::{parser::*, values::Procedure};
use error::SyntaxError;

use std::{collections::HashMap, ops::Deref, path::Path, rc::Rc};
use std::{collections::HashSet, iter::Iterator};
use std::{marker::PhantomData, path::PathBuf};

use super::library::LibraryName;
use super::library::{native, Library};
use super::Result;
use super::{error::LogicError, pair::Pair};
pub enum LibraryFactory<R: RealNumberInternalTrait, E: IEnvironment<R>> {
    Native(LibraryName, fn() -> Vec<(String, Value<R, E>)>),
    AST(Located<LibraryDefinition>),
}
#[test]
fn library_factory() -> Result<()> {
    let mut it = Interpreter::<f32, StandardEnv<f32>>::new();
    it.register_library_factory(LibraryFactory::Native(library_name!("foo"), || {
        vec![("a".to_string(), Value::Void)]
    }));
    assert_eq!(
        it.get_library(library_name!("foo").into()),
        Ok(Library::new(
            library_name!("foo").into(),
            vec![("a".to_string(), Value::Void)]
        ))
    );
    it.register_library_factory(LibraryFactory::from_char_stream(
        &library_name!("foo"),
        "(define-library (foo) (export a) (begin (define a 1)))".chars(),
    )?);
    assert_eq!(
        it.get_library(library_name!("foo").into()),
        Ok(Library::new(
            library_name!("foo").into(),
            vec![("a".to_string(), Value::Number(Number::Integer(1)))]
        ))
    );
    Ok(())
}

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> LibraryFactory<R, E> {
    pub fn get_library_name(&self) -> &LibraryName {
        match self {
            LibraryFactory::Native(name, _) => name,
            LibraryFactory::AST(library_definition) => &library_definition.0,
        }
    }
    pub fn from_char_stream(
        expect_library_name: &LibraryName,
        char_stream: impl Iterator<Item = char>,
    ) -> Result<Self> {
        let lexer = Lexer::from_char_stream(char_stream);
        let parser = Parser::from_lexer(lexer);
        for statement in parser {
            if let Statement::LibraryDefinition(library_definition) = statement? {
                if library_definition.0.deref() == expect_library_name {
                    return Ok(Self::AST(library_definition));
                }
            }
        }
        error!(LogicError::LibraryNotFound(expect_library_name.clone()))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum TailExpressionResult<'a, R: RealNumberInternalTrait, E: IEnvironment<R>> {
    TailCall(TailCall<'a, R, E>),
    Value(Value<R, E>),
}
#[derive(Debug, Clone, PartialEq)]
enum TailCall<'a, R: RealNumberInternalTrait, E: IEnvironment<R>> {
    Ref(&'a Expression, &'a [Expression], Rc<E>),
    Owned(Expression, Vec<Expression>, Rc<E>, PhantomData<R>),
}
impl<'a, R: RealNumberInternalTrait, E: IEnvironment<R>> TailCall<'a, R, E> {
    pub fn as_ref(&'a self) -> (&'a Expression, &'a [Expression], &Rc<E>) {
        match self {
            TailCall::Ref(procedure_expr, arguments, env) => (procedure_expr, arguments, env),
            TailCall::Owned(procedure_expr, arguments, env, _) => (procedure_expr, arguments, env),
        }
    }
}
pub struct LibraryLoader<R: RealNumberInternalTrait, E: IEnvironment<R>> {
    lib_factories: HashMap<LibraryName, Rc<LibraryFactory<R, E>>>,
}

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> LibraryLoader<R, E> {
    pub fn new() -> Self {
        Self {
            lib_factories: HashMap::new(),
        }
    }
    pub fn register_library_factory(&mut self, library_factory: LibraryFactory<R, E>) {
        self.lib_factories.insert(
            library_factory.get_library_name().clone(),
            Rc::new(library_factory),
        );
    }
    pub fn with_lib_factory(mut self, library_factory: LibraryFactory<R, E>) -> Self {
        self.register_library_factory(library_factory);
        self
    }
}

pub struct Interpreter<R: RealNumberInternalTrait, E: IEnvironment<R>> {
    pub env: Rc<E>,
    lib_loader: LibraryLoader<R, E>,
    imported_library: HashSet<LibraryName>,
    import_end: bool, // indicate program's import declaration part end
    pub program_directory: Option<PathBuf>,
    _marker: PhantomData<R>,
}

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> Interpreter<R, E> {
    pub fn new() -> Self {
        let mut interpreter = Self {
            env: Rc::new(E::new()),
            lib_loader: LibraryLoader::new(),
            imported_library: HashSet::new(),
            import_end: false,
            program_directory: None,
            _marker: PhantomData,
        };
        interpreter.register_stdlib_factories();
        interpreter
    }

    pub fn new_with_stdlib() -> Self {
        let mut interpreter = Interpreter::<R, E>::new();
        interpreter.eval("(import (scheme base))".chars()).unwrap();
        interpreter.eval("(import (scheme write))".chars()).unwrap();
        interpreter
    }

    pub fn append_lib_loader(&mut self, lib_loader: LibraryLoader<R, E>) {
        self.lib_loader
            .lib_factories
            .extend(lib_loader.lib_factories.into_iter());
    }
    pub fn register_library_factory(&mut self, library_factory: LibraryFactory<R, E>) {
        self.lib_loader.register_library_factory(library_factory);
    }

    fn register_stdlib_factories(&mut self) {
        self.register_library_factory(LibraryFactory::Native(
            library_name!("ruschm", "base"),
            native::base::library_map,
        ));
        self.register_library_factory(LibraryFactory::Native(
            library_name!("ruschm", "write"),
            native::write::library_map,
        ));
        self.register_library_factory(
            LibraryFactory::from_char_stream(
                &library_name!("scheme", "base"),
                include_str!("library/include/scheme/base.sld").chars(),
            )
            .unwrap(),
        );
        self.register_library_factory(
            LibraryFactory::from_char_stream(
                &library_name!("scheme", "write"),
                include_str!("library/include/scheme/write.sld").chars(),
            )
            .unwrap(),
        );
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
            None => unreachable!(),
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
        let evaluated_args_result = arguments
            .iter()
            .map(|arg| Self::eval_expression(arg, env))
            .collect::<Result<ArgVec<_, _>>>()?;
        Ok((first.expect_procedure()?, evaluated_args_result))
    }

    pub fn apply_procedure<'a>(
        initial_procedure: &Procedure<R, E>,
        mut args: ArgVec<R, E>,
        env: &Rc<E>,
    ) -> Result<Value<R, E>> {
        let formals = initial_procedure.get_parameters();
        if args.len() < formals.0.len() || (args.len() > formals.0.len() && formals.1.is_none()) {
            return error!(LogicError::ArgumentMissMatch(formals.clone(), args.len()));
        }
        let mut current_procedure = None;
        loop {
            match if current_procedure.is_none() {
                initial_procedure
            } else {
                current_procedure.as_ref().unwrap()
            } {
                Procedure::Builtin(BuiltinProcedure { pointer, .. }) => {
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
                        TailExpressionResult::TailCall(tail_call) => {
                            let (tail_procedure_expr, tail_arguments, last_env) =
                                tail_call.as_ref();
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
                if let Some(expanded) =
                    Self::try_expand_expression(procedure_expr, arguments, &env)?
                {
                    Self::eval_owned_tail_expression(expanded, env)?
                } else {
                    TailExpressionResult::TailCall(TailCall::Ref(
                        procedure_expr.as_ref(),
                        arguments,
                        env,
                    ))
                }
            }
            ExpressionBody::Conditional(cond) => {
                let (test, consequent, alternative) = cond.as_ref();
                let condition = Self::eval_expression(&test, &env)?.expect_boolean()?;
                if condition {
                    Self::eval_tail_expression(consequent, env)?
                } else {
                    match alternative {
                        Some(alter) => Self::eval_tail_expression(alter, env)?,
                        None => TailExpressionResult::Value(Value::Void),
                    }
                }
            }
            _ => TailExpressionResult::Value(Self::eval_expression(&expression, &env)?),
        })
    }
    // during eval_tail_expression, some expression will expand to owned expression
    // it has to repeat this logic for owned expression, no other workaround
    fn eval_owned_tail_expression<'a>(
        expression: Expression,
        env: Rc<E>,
    ) -> Result<TailExpressionResult<'a, R, E>> {
        Ok(match &expression.data {
            ExpressionBody::ProcedureCall(..) | ExpressionBody::Conditional(_) => {
                match expression.extract_data() {
                    ExpressionBody::ProcedureCall(procedure_expr, arguments) => {
                        if let Some(expanded) =
                            Self::try_expand_expression(procedure_expr.as_ref(), &arguments, &env)?
                        {
                            Self::eval_owned_tail_expression(expanded, env)?
                        } else {
                            TailExpressionResult::TailCall(TailCall::Owned(
                                *procedure_expr,
                                arguments,
                                env,
                                PhantomData,
                            ))
                        }
                    }
                    ExpressionBody::Conditional(cond) => {
                        let (test, consequent, alternative) = *cond;
                        let condition = Self::eval_expression(&test, &env)?.expect_boolean()?;
                        if condition {
                            Self::eval_owned_tail_expression(consequent, env)?
                        } else {
                            match alternative {
                                Some(alter) => Self::eval_owned_tail_expression(alter, env)?,
                                None => TailExpressionResult::Value(Value::Void),
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            _ => TailExpressionResult::Value(Self::eval_expression(&expression, &env)?),
        })
    }

    pub fn read_literal(expression: &Expression, env: &Rc<E>) -> Result<Value<R, E>> {
        match &expression.data {
            ExpressionBody::Primitive(primitive) => Self::eval_primitive(primitive),
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
                        None => {
                            return Err(ErrorData::Logic(LogicError::MetaCircularSyntax(
                                SyntaxError::UnexpectedToken(TokenData::Period),
                            ))
                            .locate(*location))
                        }
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
            o => unreachable!("expression should not be {:?}", o),
        }
    }

    fn eval_primitive(datum: &Primitive) -> Result<Value<R, E>> {
        Ok(match &datum {
            Primitive::Character(c) => Value::Character(*c),
            Primitive::String(string) => Value::String(string.clone()),
            Primitive::Boolean(value) => Value::Boolean(*value),
            Primitive::Integer(value) => Value::Number(Number::Integer(*value)),
            Primitive::Real(number_literal) => Value::Number(Number::Real(
                R::from(number_literal.parse::<f64>().unwrap()).unwrap(),
            )),
            // TODO: apply gcd here.
            Primitive::Rational(a, b) => Value::Number(Number::Rational(*a, *b as i32)),
        })
    }
    fn try_expand_expression(
        procedure_expr: &Expression,
        arguments: &[Expression],
        env: &Rc<E>,
    ) -> Result<Option<Expression>> {
        Ok(
            if let Value::Transformer(transformer) = Self::eval_expression(procedure_expr, env)? {
                Some({
                    let location = procedure_expr.location;
                    let mut transformed = transformer
                        .transform(arguments.iter().cloned().collect::<Vec<_>>())?
                        .into_iter();
                    let expression = if let Some(statement) = transformed.next() {
                        statement.expect_expression()?
                    } else {
                        return located_error!(
                            SyntaxError::ExpectSomething("expression".to_string()),
                            location
                        );
                    };

                    if transformed.next().is_some() {
                        panic!("transformer expand to multiple statements unsupported")
                    }
                    expression
                })
            } else {
                None
            },
        )
    }
    pub fn eval_expression(expression: &Expression, env: &Rc<E>) -> Result<Value<R, E>> {
        Ok(match &expression.data {
            ExpressionBody::Primitive(datum) => Self::eval_primitive(datum)?,
            ExpressionBody::List(_) => {
                unreachable!("expression list should be converted to list value")
            }
            ExpressionBody::Vector(_) => Self::read_literal(&expression, env)?,
            ExpressionBody::ProcedureCall(procedure_expr, arguments) => {
                if let Some(expanded) = Self::try_expand_expression(procedure_expr, arguments, env)?
                {
                    Self::eval_expression(&expanded, env)?
                } else {
                    let procedure =
                        Self::eval_expression(procedure_expr, env)?.expect_procedure()?;
                    let evaluated_args: Result<ArgVec<_, _>> = arguments
                        .into_iter()
                        .map(|arg| Self::eval_expression(arg, env))
                        .collect();
                    Self::apply_procedure(&procedure, evaluated_args?, env)?
                }
            }
            ExpressionBody::Period => {
                return located_error!(
                    LogicError::UnexpectedExpression(expression.clone()),
                    expression.location
                );
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
                if Self::eval_expression(&test, &env)?.expect_boolean()? {
                    Self::eval_expression(&consequent, env)?
                } else {
                    match alternative {
                        Some(alter) => Self::eval_expression(&alter, env)?,
                        None => Value::Void,
                    }
                }
            }
            ExpressionBody::Quote(inner) => Self::read_literal(inner.as_ref(), env)?,

            ExpressionBody::Identifier(ident) => match env.get(ident.as_str()) {
                Some(value) => value.clone(),
                None => {
                    return located_error!(
                        LogicError::UnboundedSymbol(ident.clone()),
                        expression.location
                    )
                }
            },
        })
    }

    pub fn eval_import(&mut self, imports: &ImportDeclaration, env: Rc<E>) -> Result<()> {
        let mut definitions = HashMap::new();
        for import in &imports.0 {
            definitions.extend(self.eval_import_set(import)?.into_iter());
        }
        for (name, value) in definitions {
            env.define(name, value);
        }
        Ok(())
    }
    fn file_library_factory(&self, name: &Located<LibraryName>) -> Result<LibraryFactory<R, E>> {
        let base_directory = if let Some(program_directory) = &self.program_directory {
            program_directory.clone()
        } else {
            std::env::current_dir()?
        };
        // TODO: file extension, file system variants
        let path = base_directory
            .join(name.deref().path())
            .with_extension("sld");
        if path.exists() {
            let char_stream = file_char_stream(&path)?;
            LibraryFactory::from_char_stream(name.deref(), char_stream)
        } else {
            located_error!(
                LogicError::LibraryNotFound(name.deref().clone()),
                name.location.clone()
            )
        }
    }
    fn new_library(&mut self, factory: &LibraryFactory<R, E>) -> Result<Library<R, E>> {
        match factory {
            LibraryFactory::Native(name, f) => Ok(Library::new(name.clone().into(), f())),
            LibraryFactory::AST(library_definition) => {
                self.eval_library_definition(library_definition.deref())
            }
        }
    }
    pub fn get_library(&mut self, name: Located<LibraryName>) -> Result<Library<R, E>> {
        let factory = match self.lib_loader.lib_factories.get(&name) {
            Some(factory) => factory,
            None => {
                let new_factory = self.file_library_factory(&name)?;
                self.lib_loader
                    .lib_factories
                    .entry(name.deref().clone())
                    .or_insert(Rc::new(new_factory))
            }
        }
        .clone();
        self.new_library(&factory)
    }
    pub fn eval_import_set(&mut self, import: &ImportSet) -> Result<Vec<(String, Value<R, E>)>> {
        match &import.data {
            ImportSetBody::Direct(lib_name) => {
                if self
                    .imported_library
                    .insert(lib_name.clone().extract_data())
                {
                    let library = self.get_library(lib_name.clone())?;
                    self.imported_library.remove(lib_name);
                    Ok(library
                        .iter_definitions()
                        .map(|(name, value)| (name.clone(), value.clone()))
                        .collect())
                } else {
                    located_error!(
                        LogicError::LibraryImportCyclic(lib_name.clone().extract_data()),
                        lib_name.location
                    )
                }
            }
            ImportSetBody::Only(import_set, identifiers) => {
                let id_set = identifiers.into_iter().collect::<HashSet<_>>();
                Ok(self
                    .eval_import_set(import_set.as_ref())?
                    .into_iter()
                    .filter(|(name, _)| id_set.contains(name))
                    .collect())
            }
            ImportSetBody::Except(import_set, identifiers) => {
                let id_set = identifiers.into_iter().collect::<HashSet<_>>();
                Ok(self
                    .eval_import_set(import_set.as_ref())?
                    .into_iter()
                    .filter(|(name, _)| !id_set.contains(name))
                    .collect())
            }
            ImportSetBody::Prefix(import_set, prefix) => Ok(self
                .eval_import_set(import_set.as_ref())?
                .into_iter()
                .map(|(name, value)| (format!("{}{}", prefix, name), value))
                .collect()),
            ImportSetBody::Rename(import_set, renames) => {
                let id_map = renames
                    .into_iter()
                    .map(|(from, to)| (from, to))
                    .collect::<HashMap<_, _>>();
                Ok(self
                    .eval_import_set(import_set.as_ref())?
                    .into_iter()
                    .map(|(name, value)| match id_map.get(&name) {
                        Some(to) => ((*to).clone(), value),
                        None => (name, value),
                    })
                    .collect())
            }
        }
    }

    pub fn eval_expression_statement_or_definition(
        &mut self,
        statement: &Statement,
        env: Rc<E>,
    ) -> Result<Option<Value<R, E>>> {
        Ok(match statement {
            Statement::Expression(expr) => Some(Self::eval_expression(&expr, &env)?),
            Statement::Definition(Definition {
                data: DefinitionBody(name, expr),
                ..
            }) => {
                let value = Self::eval_expression(&expr, &env)?;
                env.define(name.clone(), value);
                None
            }
            Statement::SyntaxDefinition(_) => todo!("defining new syntax is not supported"),
            _ => error!(SyntaxError::ExpectSomething(
                "expression/definition".to_string()
            ))?,
        })
    }

    pub fn eval_ast(&mut self, ast: &Statement, env: Rc<E>) -> Result<Option<Value<R, E>>> {
        if !self.import_end {
            Ok(match ast {
                Statement::ImportDeclaration(imports) => {
                    self.eval_import(imports, env)?;
                    None
                }
                Statement::LibraryDefinition(library_definition) => {
                    return located_error!(
                        SyntaxError::ExpectSomething(
                            "import declaration/expression/definition".to_string()
                        ),
                        library_definition.location.clone()
                    );
                }
                other => {
                    self.import_end = true;
                    self.eval_expression_statement_or_definition(other, env)?
                }
            })
        } else {
            self.eval_expression_statement_or_definition(ast, env)
        }
    }

    pub fn eval_root_ast(&mut self, ast: &Statement) -> Result<Option<Value<R, E>>> {
        self.eval_ast(ast, self.env.clone())
    }

    pub fn eval_library_definition<'a>(
        &mut self,
        library_definition: &LibraryDefinition,
    ) -> Result<Library<R, E>> {
        let name = library_definition.0.clone();
        let mut definitions = HashMap::new();
        let mut final_exports = Vec::new();
        let lib_env = Rc::new(E::new());
        for declaration in &library_definition.1 {
            match &declaration.data {
                LibraryDeclaration::ImportDeclaration(imports) => {
                    self.eval_import(imports, lib_env.clone())?;
                }
                LibraryDeclaration::Export(exports) => final_exports.extend(exports.iter()),
                LibraryDeclaration::Begin(statements) => {
                    for statement in statements.iter() {
                        self.eval_expression_statement_or_definition(statement, lib_env.clone())?;
                    }
                }
            }
        }
        for export in final_exports {
            let (from, to) = match &export.data {
                ExportSpec::Direct(identifier) => (identifier, identifier),
                ExportSpec::Rename(from, to) => (from, to),
            };
            match lib_env.get(from) {
                Some(value) => {
                    definitions.insert(to.clone(), value.clone());
                }
                None => located_error!(LogicError::UnboundedSymbol(from.clone()), export.location)?,
            }
        }
        Ok(Library::new(name, definitions))
    }
    pub fn eval_program<'a>(
        &mut self,
        asts: impl IntoIterator<Item = &'a Statement>,
    ) -> Result<Option<Value<R, E>>> {
        asts.into_iter()
            .try_fold(None, |_, ast| self.eval_root_ast(&ast))
    }

    pub fn eval(&mut self, char_stream: impl Iterator<Item = char>) -> Result<Option<Value<R, E>>> {
        {
            let lexer = Lexer::from_char_stream(char_stream);
            let mut parser = Parser::from_lexer(lexer);
            parser.try_fold(None, |_, statement| self.eval_root_ast(&statement?))
        }
    }
    pub fn eval_file(&mut self, path: PathBuf) -> Result<Option<Value<R, E>>> {
        self.program_directory = path.clone().parent().map(Path::to_owned);
        self.eval(file_char_stream(&path)?)
    }
}

#[test]
fn number() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new_with_stdlib();
    assert_eq!(
        interpreter
            .eval_root_expression(ExpressionBody::Primitive(Primitive::Integer(-1)).into())?,
        Value::Number(Number::Integer(-1))
    );
    assert_eq!(
        interpreter
            .eval_root_expression(ExpressionBody::Primitive(Primitive::Rational(1, 3)).into())?,
        Value::Number(Number::Rational(1, 3))
    );
    assert_eq!(
        interpreter.eval_root_expression(
            ExpressionBody::Primitive(Primitive::Real("-3.45e-7".to_string()).into()).into()
        )?,
        Value::Number(Number::Real(-3.45e-7))
    );
    Ok(())
}

#[test]
fn arithmetic() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new_with_stdlib();

    assert_eq!(
        interpreter.eval_root_expression(Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
                "+".to_string()
            ))),
            vec![
                ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                ExpressionBody::Primitive(Primitive::Integer(2)).into()
            ]
        )))?,
        Value::Number(Number::Integer(3))
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
                "+".to_string()
            ))),
            vec![
                ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                ExpressionBody::Primitive(Primitive::Rational(1, 2)).into()
            ]
        )))?,
        Value::Number(Number::Rational(3, 2))
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
                "*".to_string()
            ))),
            vec![
                ExpressionBody::Primitive(Primitive::Rational(1, 2)).into(),
                ExpressionBody::Primitive(Primitive::Real("2.0".to_string()).into()).into(),
            ]
        )))?,
        Value::Number(Number::Real(1.0)),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
                "/".to_string()
            ))),
            vec![
                ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                ExpressionBody::Primitive(Primitive::Integer(0)).into()
            ]
        ))),
        Err(ErrorData::Logic(LogicError::DivisionByZero).no_locate()),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
                "max".to_string()
            ))),
            vec![
                ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                ExpressionBody::Primitive(Primitive::Real("1.3".to_string()).into()).into(),
            ]
        )))?,
        Value::Number(Number::Real(1.3)),
    );
    assert_eq!(
        interpreter.eval_root_expression(Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
                "min".to_string()
            ))),
            vec![
                ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                ExpressionBody::Primitive(Primitive::Real("1.3".to_string()).into()).into(),
            ]
        )))?,
        Value::Number(Number::Real(1.0)),
    );
    assert_eq!(
        interpreter.eval_root_expression(Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
                "min".to_string()
            ))),
            vec![ExpressionBody::Primitive(Primitive::String("a".to_string()).into()).into()]
        ))),
        Err(ErrorData::Logic(LogicError::TypeMisMatch("a".to_string(), Type::Number)).no_locate()),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
                "max".to_string()
            ))),
            vec![ExpressionBody::Primitive(Primitive::String("a".to_string()).into()).into()]
        ))),
        Err(ErrorData::Logic(LogicError::TypeMisMatch("a".to_string(), Type::Number)).no_locate()),
    );

    assert_eq!(
        interpreter.eval_root_expression(Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
                "sqrt".to_string()
            ))),
            vec![ExpressionBody::Primitive(Primitive::Integer(4)).into()]
        )))?,
        Value::Number(Number::Real(2.0)),
    );

    match interpreter.eval_root_expression(Expression::from(ExpressionBody::ProcedureCall(
        Box::new(Expression::from(ExpressionBody::Identifier(
            "sqrt".to_string(),
        ))),
        vec![ExpressionBody::Primitive(Primitive::Integer(-4)).into()],
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
            interpreter.eval_root_expression(Expression::from(ExpressionBody::ProcedureCall(
                Box::new(Expression::from(ExpressionBody::Identifier(
                    cmp.to_string()
                ))),
                vec![
                    ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                    ExpressionBody::Primitive(Primitive::Rational(1, 1)).into(),
                    ExpressionBody::Primitive(Primitive::Real("1.0".to_string()).into()).into(),
                ],
            )))?,
            Value::Boolean(*result)
        )
    }

    Ok(())
}

#[test]
fn undefined() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new_with_stdlib();

    assert_eq!(
        interpreter.eval_root_expression(Expression::from(ExpressionBody::Identifier(
            "foo".to_string()
        ))),
        Err(ErrorData::Logic(LogicError::UnboundedSymbol("foo".to_string())).no_locate())
    );
    Ok(())
}

#[test]
fn variable_definition() -> Result<()> {
    let mut interpreter = Interpreter::<f32, StandardEnv<f32>>::new();

    let program = vec![
        Statement::Definition(Definition::from(DefinitionBody(
            "a".to_string(),
            ExpressionBody::Primitive(Primitive::Integer(1)).into(),
        ))),
        Statement::Definition(Definition::from(DefinitionBody(
            "b".to_string(),
            Expression::from(ExpressionBody::Identifier("a".to_string())),
        ))),
        Statement::Expression(Expression::from(ExpressionBody::Identifier(
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
    let mut interpreter = Interpreter::<f32, StandardEnv<f32>>::new();

    let program = vec![
        Statement::Definition(Definition::from(DefinitionBody(
            "a".to_string(),
            ExpressionBody::Primitive(Primitive::Integer(1)).into(),
        ))),
        Statement::Expression(Expression::from(ExpressionBody::Assignment(
            "a".to_string(),
            Box::new(ExpressionBody::Primitive(Primitive::Integer(2)).into()),
        ))),
        Statement::Expression(Expression::from(ExpressionBody::Identifier(
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
fn builtin_procedural() -> Result<()> {
    let mut interpreter = Interpreter::<f32, StandardEnv<f32>>::new_with_stdlib();

    let program = vec![
        Statement::Definition(Definition::from(DefinitionBody(
            "get-add".to_string(),
            simple_procedure(
                ParameterFormals::new(),
                Expression::from(ExpressionBody::Identifier("+".to_string())),
            ),
        ))),
        Statement::Expression(Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::ProcedureCall(
                Box::new(Expression::from(ExpressionBody::Identifier(
                    "get-add".to_string(),
                ))),
                vec![],
            ))),
            vec![
                ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                ExpressionBody::Primitive(Primitive::Integer(2)).into(),
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
    let mut interpreter = Interpreter::<f32, StandardEnv<f32>>::new_with_stdlib();

    let program = vec![
        Statement::Definition(Definition::from(DefinitionBody(
            "add".to_string(),
            simple_procedure(
                ParameterFormals(vec!["x".to_string(), "y".to_string()], None),
                Expression::from(ExpressionBody::ProcedureCall(
                    Box::new(Expression::from(ExpressionBody::Identifier(
                        "+".to_string(),
                    ))),
                    vec![
                        Expression::from(ExpressionBody::Identifier("x".to_string())),
                        Expression::from(ExpressionBody::Identifier("y".to_string())),
                    ],
                )),
            ),
        ))),
        Statement::Expression(Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
                "add".to_string(),
            ))),
            vec![
                ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                ExpressionBody::Primitive(Primitive::Integer(2)).into(),
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
fn procedure_debug() -> Result<()> {
    let mut interpreter = Interpreter::<f32, StandardEnv<f32>>::new();

    let program = vec![Statement::Expression(simple_procedure(
        ParameterFormals(vec!["x".to_string(), "y".to_string()], None),
        Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
                "+".to_string(),
            ))),
            vec![
                Expression::from(ExpressionBody::Identifier("x".to_string())),
                Expression::from(ExpressionBody::Identifier("y".to_string())),
            ],
        )),
    ))];
    // If impl SchemeProcedure with default Debug, this println will end up with an infinite recursion
    println!("{:?}", interpreter.eval_program(program.iter())?);
    Ok(())
}

#[test]
fn lambda_call() -> Result<()> {
    let mut interpreter = Interpreter::<f32, StandardEnv<f32>>::new_with_stdlib();

    let program = vec![Statement::Expression(Expression::from(
        ExpressionBody::ProcedureCall(
            Box::new(simple_procedure(
                ParameterFormals(
                    vec!["x".to_string(), "y".to_string()],
                    Some("z".to_string()),
                ),
                Expression::from(ExpressionBody::ProcedureCall(
                    Box::new(Expression::from(ExpressionBody::Identifier(
                        "+".to_string(),
                    ))),
                    vec![
                        Expression::from(ExpressionBody::Identifier("x".to_string())),
                        Expression::from(ExpressionBody::Identifier("y".to_string())),
                    ],
                )),
            )),
            vec![
                ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                ExpressionBody::Primitive(Primitive::Integer(2)).into(),
                ExpressionBody::Primitive(Primitive::String("something-else".to_string()).into())
                    .into(),
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
    let mut interpreter = Interpreter::<f32, StandardEnv<f32>>::new_with_stdlib();

    let program = vec![
        Statement::Definition(Definition::from(DefinitionBody(
            "counter-creator".to_string(),
            Expression::from(ExpressionBody::Procedure(SchemeProcedure(
                ParameterFormals::new(),
                vec![Definition::from(DefinitionBody(
                    "current".to_string(),
                    ExpressionBody::Primitive(Primitive::Integer(0)).into(),
                ))],
                vec![Expression::from(ExpressionBody::Procedure(
                    SchemeProcedure(
                        ParameterFormals::new(),
                        vec![],
                        vec![
                            Expression::from(ExpressionBody::Assignment(
                                "current".to_string(),
                                Box::new(Expression::from(ExpressionBody::ProcedureCall(
                                    Box::new(Expression::from(ExpressionBody::Identifier(
                                        "+".to_string(),
                                    ))),
                                    vec![
                                        Expression::from(ExpressionBody::Identifier(
                                            "current".to_string(),
                                        )),
                                        ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                                    ],
                                ))),
                            )),
                            Expression::from(ExpressionBody::Identifier("current".to_string())),
                        ],
                    ),
                ))],
            ))),
        ))),
        Statement::Definition(Definition::from(DefinitionBody(
            "counter".to_string(),
            Expression::from(ExpressionBody::ProcedureCall(
                Box::new(Expression::from(ExpressionBody::Identifier(
                    "counter-creator".to_string(),
                ))),
                vec![],
            )),
        ))),
        Statement::Expression(Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
                "counter".to_string(),
            ))),
            vec![],
        ))),
        Statement::Expression(Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
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
    let mut interpreter = Interpreter::<f32, StandardEnv<f32>>::new();

    assert_eq!(
        interpreter.eval_program(
            vec![Statement::Expression(Expression::from(
                ExpressionBody::Conditional(Box::new((
                    ExpressionBody::Primitive(Primitive::Boolean(true)).into(),
                    ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                    Some(ExpressionBody::Primitive(Primitive::Integer(2)).into()),
                ))),
            ))]
            .iter()
        )?,
        Some(Value::Number(Number::Integer(1)))
    );
    assert_eq!(
        interpreter.eval_program(
            vec![Statement::Expression(Expression::from(
                ExpressionBody::Conditional(Box::new((
                    ExpressionBody::Primitive(Primitive::Boolean(false)).into(),
                    ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                    Some(ExpressionBody::Primitive(Primitive::Integer(2)).into()),
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
    let mut interpreter = Interpreter::<f32, StandardEnv<f32>>::new_with_stdlib();

    let program = vec![
        Statement::Definition(Definition::from(DefinitionBody(
            "adda".to_string(),
            simple_procedure(
                ParameterFormals(vec!["x".to_string()], None),
                Expression::from(ExpressionBody::ProcedureCall(
                    Box::new(Expression::from(ExpressionBody::Identifier(
                        "+".to_string(),
                    ))),
                    vec![
                        Expression::from(ExpressionBody::Identifier("x".to_string())),
                        Expression::from(ExpressionBody::Identifier("a".to_string())),
                    ],
                )),
            ),
        ))),
        Statement::Definition(Definition::from(DefinitionBody(
            "a".to_string(),
            ExpressionBody::Primitive(Primitive::Integer(1)).into(),
        ))),
        Statement::Expression(Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
                "adda".to_string(),
            ))),
            vec![ExpressionBody::Primitive(Primitive::Integer(2)).into()],
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
    let mut interpreter = Interpreter::<f32, StandardEnv<f32>>::new_with_stdlib();

    let program = vec![
        Statement::Definition(Definition::from(DefinitionBody(
            "add".to_string(),
            simple_procedure(
                ParameterFormals(vec!["x".to_string(), "y".to_string()], None),
                Expression::from(ExpressionBody::ProcedureCall(
                    Box::new(Expression::from(ExpressionBody::Identifier(
                        "+".to_string(),
                    ))),
                    vec![
                        Expression::from(ExpressionBody::Identifier("x".to_string())),
                        Expression::from(ExpressionBody::Identifier("y".to_string())),
                    ],
                )),
            ),
        ))),
        Statement::Definition(Definition::from(DefinitionBody(
            "apply-op".to_string(),
            simple_procedure(
                ParameterFormals(
                    vec!["op".to_string(), "x".to_string(), "y".to_string()],
                    None,
                ),
                Expression::from(ExpressionBody::ProcedureCall(
                    Box::new(Expression::from(ExpressionBody::Identifier(
                        "op".to_string(),
                    ))),
                    vec![
                        Expression::from(ExpressionBody::Identifier("x".to_string())),
                        Expression::from(ExpressionBody::Identifier("y".to_string())),
                    ],
                )),
            ),
        ))),
        Statement::Expression(Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
                "apply-op".to_string(),
            ))),
            vec![
                Expression::from(ExpressionBody::Identifier("add".to_string())),
                ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                ExpressionBody::Primitive(Primitive::Integer(2)).into(),
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
fn eval_tail_expression() -> Result<()> {
    let expect_result = convert_located(vec![
        ExpressionBody::Primitive(Primitive::Integer(2)),
        ExpressionBody::Primitive(Primitive::Integer(5)),
    ]);
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new_with_stdlib();

    {
        let expression = ExpressionBody::Primitive(Primitive::Integer(3)).into();
        assert_eq!(
            Interpreter::eval_tail_expression(&expression, interpreter.env.clone())?,
            TailExpressionResult::Value(Value::Number(Number::Integer(3)))
        );
    }
    {
        let expression = Expression::from(ExpressionBody::ProcedureCall(
            Box::new(Expression::from(ExpressionBody::Identifier(
                "+".to_string(),
            ))),
            expect_result.clone(),
        ));
        assert_eq!(
            Interpreter::eval_tail_expression(&expression, interpreter.env.clone())?,
            TailExpressionResult::TailCall(TailCall::Ref(
                &Expression::from(ExpressionBody::Identifier("+".to_string())),
                &expect_result,
                interpreter.env.clone()
            ))
        );
    }
    {
        let expression = Expression::from(ExpressionBody::Conditional(Box::new((
            ExpressionBody::Primitive(Primitive::Boolean(true)).into(),
            Expression::from(ExpressionBody::ProcedureCall(
                Box::new(Expression::from(ExpressionBody::Identifier(
                    "+".to_string(),
                ))),
                expect_result.clone(),
            )),
            None,
        ))));
        assert_eq!(
            Interpreter::eval_tail_expression(&expression, interpreter.env.clone())?,
            TailExpressionResult::TailCall(TailCall::Ref(
                &Expression::from(ExpressionBody::Identifier("+".to_string())),
                &expect_result,
                interpreter.env.clone()
            ))
        );
    }
    {
        let expression = Expression::from(ExpressionBody::Conditional(Box::new((
            ExpressionBody::Primitive(Primitive::Boolean(false)).into(),
            Expression::from(ExpressionBody::ProcedureCall(
                Box::new(Expression::from(ExpressionBody::Identifier(
                    "+".to_string(),
                ))),
                expect_result.clone(),
            )),
            Some(ExpressionBody::Primitive(Primitive::Integer(4)).into()),
        ))));
        assert_eq!(
            Interpreter::eval_tail_expression(&expression, interpreter.env.clone())?,
            TailExpressionResult::Value(Value::Number(Number::Integer(4)))
        );
    }
    {
        let expression = Expression::from(ExpressionBody::Conditional(Box::new((
            ExpressionBody::Primitive(Primitive::Boolean(false)).into(),
            ExpressionBody::Primitive(Primitive::Integer(4)).into(),
            Some(Expression::from(ExpressionBody::ProcedureCall(
                Box::new(Expression::from(ExpressionBody::Identifier(
                    "+".to_string(),
                ))),
                expect_result.clone(),
            ))),
        ))));
        assert_eq!(
            Interpreter::eval_tail_expression(&expression, interpreter.env.clone())?,
            TailExpressionResult::TailCall(TailCall::Ref(
                &Expression::from(ExpressionBody::Identifier("+".to_string())),
                &expect_result,
                interpreter.env.clone()
            ))
        );
    }
    Ok(())
}

#[test]
fn datum_literal() -> Result<()> {
    let interpreter = Interpreter::<f32, StandardEnv<f32>>::new();

    assert_eq!(
        Interpreter::eval_expression(
            &ExpressionBody::Quote(Box::new(
                ExpressionBody::Primitive(Primitive::Integer(1,)).into()
            ))
            .into(),
            &interpreter.env,
        )?,
        Value::Number(Number::Integer(1))
    );
    assert_eq!(
        Interpreter::eval_expression(
            &ExpressionBody::Quote(Box::new(ExpressionBody::Identifier("a".to_string()).into()))
                .into(),
            &interpreter.env,
        )?,
        Value::Symbol("a".to_string())
    );
    assert_eq!(
        Interpreter::eval_expression(
            &ExpressionBody::Quote(
                Box::<Expression>::new(
                    ExpressionBody::List(vec![
                        ExpressionBody::Primitive(Primitive::Integer(1)).into()
                    ])
                    .into()
                )
                .into()
            )
            .into(),
            &interpreter.env,
        )?,
        Value::Pair(Box::new(Pair {
            car: Value::Number(Number::Integer(1)),
            cdr: Value::EmptyList
        }))
    );
    assert_eq!(
        Interpreter::eval_expression(
            &ExpressionBody::Quote(
                Box::<Expression>::new(
                    ExpressionBody::Vector(
                        vec![ExpressionBody::Identifier("a".to_string()).into()]
                    )
                    .into()
                )
                .into()
            )
            .into(),
            &interpreter.env,
        )?,
        Value::Vector(ValueReference::new_immutable(vec![Value::Symbol(
            "a".to_string()
        )]))
    );
    Ok(())
}

#[test]
fn search_library() -> Result<()> {
    let mut interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    {
        let library = interpreter.get_library(library_name!("scheme", "base").into())?;
        assert!(library
            .iter_definitions()
            .find(|(name, _)| name.as_str() == "+")
            .is_some());
    }
    {
        // not exist
        let library = interpreter.get_library(library_name!("lib", "not", "exist").into());
        assert_eq!(
            library,
            error!(LogicError::LibraryNotFound(library_name!(
                "lib", "not", "exist"
            )))
        );
    }
    let lib_not_exist_factory =
        LibraryFactory::Native(library_name!("lib", "not", "exist"), || {
            vec![("a".to_string(), Value::Boolean(false))]
        });
    // exist now
    interpreter.register_library_factory(lib_not_exist_factory);
    {
        let library = interpreter.get_library(library_name!("lib", "not", "exist").into())?;
        assert_eq!(
            library
                .iter_definitions()
                .find(|(name, _)| name.as_str() == "a"),
            Some((&"a".to_string(), &Value::Boolean(false)))
        );
    }
    Ok(())
}

#[test]
fn import_set() -> Result<()> {
    let mut interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    interpreter.register_library_factory(LibraryFactory::Native(
        library_name!("foo", "bar").into(),
        || {
            vec![
                ("a".to_string(), Value::String("father".to_string())),
                ("b".to_string(), Value::String("bob".to_string())),
            ]
        },
    ));
    let direct = ImportSetBody::Direct(library_name!("foo", "bar").into());
    {
        let definitions = interpreter.eval_import_set(&direct.clone().into())?;
        assert!(definitions.len() == 2);
        assert!(definitions.contains(&("a".to_string(), Value::String("father".to_string()))));
        assert!(definitions.contains(&("b".to_string(), Value::String("bob".to_string()))));
    }
    {
        let only = ImportSetBody::Only(Box::new(direct.clone().into()), vec!["b".to_string()]);
        let definitions = interpreter.eval_import_set(&only.into())?;
        assert!(definitions.len() == 1);
        assert!(definitions.contains(&("b".to_string(), Value::String("bob".to_string()))));
    }
    let prefix = ImportSetBody::Prefix(Box::new(direct.clone().into()), "god-".to_string());
    {
        let definitions = interpreter.eval_import_set(&prefix.clone().into())?;
        assert!(definitions.len() == 2);
        assert!(definitions.contains(&("god-a".to_string(), Value::String("father".to_string()))));
        assert!(definitions.contains(&("god-b".to_string(), Value::String("bob".to_string()))));
    }
    {
        let except =
            ImportSetBody::Except(Box::new(prefix.clone().into()), vec!["god-b".to_string()]);
        let definitions = interpreter.eval_import_set(&except.into())?;
        assert!(definitions.len() == 1);
        assert!(definitions.contains(&("god-a".to_string(), Value::String("father".to_string()))));
    }
    {
        let rename = ImportSetBody::Rename(
            Box::new(prefix.clone().into()),
            vec![("god-b".to_string(), "human-a".to_string())],
        );
        let definitions = interpreter.eval_import_set(&rename.into())?;
        assert!(definitions.len() == 2);
        assert!(definitions.contains(&("god-a".to_string(), Value::String("father".to_string()))));
        assert!(definitions.contains(&("human-a".to_string(), Value::String("bob".to_string()))));
    }
    Ok(())
}
#[test]
fn import() -> Result<()> {
    let mut interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    interpreter.register_library_factory(LibraryFactory::Native(
        library_name!("foo", "bar").into(),
        || {
            vec![
                ("a".to_string(), Value::String("father".to_string())),
                ("b".to_string(), Value::String("bob".to_string())),
            ]
        },
    ));
    let import_declaration = ImportDeclaration(vec![
        ImportSetBody::Only(
            Box::new(ImportSetBody::Direct(library_name!("foo", "bar").into()).into()),
            vec!["b".to_string()],
        )
        .into(),
        ImportSetBody::Rename(
            Box::new(ImportSetBody::Direct(library_name!("foo", "bar").into()).into()),
            vec![("a".to_string(), "c".to_string())],
        )
        .into(),
    ]);
    use std::ops::Deref;
    interpreter.eval_import(&import_declaration, interpreter.env.clone())?;
    {
        let value = interpreter.env.get("b").unwrap();
        assert_eq!(value.deref(), &Value::String("bob".to_string()));
    }
    {
        let value = interpreter.env.get("c").unwrap();
        assert_eq!(value.deref(), &Value::String("father".to_string()));
    }
    Ok(())
}

#[test]
fn library_definition() -> Result<()> {
    let mut interpreter = Interpreter::<f32, StandardEnv<f32>>::new();
    interpreter.register_library_factory(LibraryFactory::Native(
        library_name!("foo", "bar").into(),
        || {
            vec![
                ("a".to_string(), Value::String("father".to_string())),
                ("b".to_string(), Value::String("bob".to_string())),
            ]
        },
    ));
    {
        // empty library
        let library_definition = LibraryDefinition(library_name!("foo", "foo-bar").into(), vec![]);
        let library = interpreter.eval_library_definition(&library_definition)?;
        assert_eq!(
            library,
            Library::new(library_name!("foo", "foo-bar").into(), vec![])
        );
    }
    {
        // export direct
        let library_definition = LibraryDefinition(
            library_name!("foo", "foo-bar").into(),
            vec![
                LibraryDeclaration::ImportDeclaration(
                    ImportDeclaration(vec![ImportSetBody::Direct(
                        library_name!("foo", "bar").into(),
                    )
                    .into()])
                    .no_locate(),
                )
                .into(),
                LibraryDeclaration::Export(vec![ExportSpec::Direct("a".to_string()).into()]).into(),
            ],
        );
        let library = interpreter.eval_library_definition(&library_definition)?;
        assert_eq!(
            library,
            Library::new(
                library_name!("foo", "foo-bar").into(),
                vec![("a".to_string(), Value::String("father".to_string()))]
            )
        );
    }
    {
        // export direct and rename
        let library_definition = LibraryDefinition(
            library_name!("foo", "foo-bar").into(),
            vec![
                LibraryDeclaration::Export(vec![ExportSpec::Rename(
                    "b".to_string(),
                    "c".to_string(),
                )
                .into()])
                .into(),
                LibraryDeclaration::ImportDeclaration(
                    ImportDeclaration(vec![ImportSetBody::Direct(
                        library_name!("foo", "bar").into(),
                    )
                    .into()])
                    .no_locate(),
                )
                .into(),
                LibraryDeclaration::Export(vec![ExportSpec::Direct("a".to_string()).into()]).into(),
            ],
        );
        let library = interpreter.eval_library_definition(&library_definition)?;
        assert_eq!(
            library,
            Library::new(
                library_name!("foo", "foo-bar").into(),
                vec![
                    ("a".to_string(), Value::String("father".to_string())),
                    ("c".to_string(), Value::String("bob".to_string()))
                ]
            )
        );
    }
    {
        // export local define
        let library_definition = LibraryDefinition(
            library_name!("foo", "foo-bar").into(),
            vec![
                // define need (scheme base)
                LibraryDeclaration::ImportDeclaration(
                    ImportDeclaration(vec![ImportSetBody::Direct(
                        library_name!("scheme", "base").into(),
                    )
                    .into()])
                    .no_locate(),
                )
                .into(),
                LibraryDeclaration::Begin(vec![Statement::Definition(
                    DefinitionBody(
                        "a".to_string(),
                        ExpressionBody::Primitive(Primitive::Integer(5)).into(),
                    )
                    .into(),
                )])
                .into(),
                LibraryDeclaration::Export(vec![ExportSpec::Direct("a".to_string()).into()]).into(),
            ],
        );
        let library = interpreter.eval_library_definition(&library_definition)?;
        assert_eq!(
            library,
            Library::new(
                library_name!("foo", "foo-bar").into(),
                vec![("a".to_string(), Value::Number(Number::Integer(5))),]
            )
        );
    }

    Ok(())
}

#[test]
fn import_cyclic() -> Result<()> {
    let mut it = Interpreter::<f32, StandardEnv<f32>>::new();
    it.register_library_factory(LibraryFactory::from_char_stream(
        &library_name!("foo"),
        "(define-library (foo) (import (foo)))".chars(),
    )?);
    let result = it.get_library(library_name!("foo").into());
    assert_eq!(
        result,
        error!(LogicError::LibraryImportCyclic(library_name!("foo")))
    );
    Ok(())
}

#[test]
fn transformer() -> Result<()> {
    let it = Interpreter::<f32, StandardEnv<f32>>::new_with_stdlib();
    // transform to expression
    it.env.define(
        "foo".to_string(),
        Value::Transformer(Transformer::Native(|expressions| {
            Ok(vec![Statement::Expression(
                ExpressionBody::ProcedureCall(
                    Box::new(ExpressionBody::Identifier("+".to_string()).into()),
                    expressions.into_iter().collect(),
                )
                .into(),
            )])
        })),
    );
    let env = it.env.clone();
    assert_eq!(
        Interpreter::eval_expression(
            &ExpressionBody::ProcedureCall(
                Box::new(ExpressionBody::Identifier("foo".to_string()).into()),
                vec![
                    ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                    ExpressionBody::Primitive(Primitive::Integer(2)).into(),
                    ExpressionBody::Primitive(Primitive::Integer(3)).into(),
                ],
            )
            .into(),
            &env,
        ),
        Ok(Value::Number(Number::Integer(6)))
    );
    assert_eq!(
        Interpreter::eval_tail_expression(
            &ExpressionBody::ProcedureCall(
                Box::new(ExpressionBody::Identifier("foo".to_string()).into()),
                vec![
                    ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                    ExpressionBody::Primitive(Primitive::Integer(2)).into(),
                    ExpressionBody::Primitive(Primitive::Integer(3)).into(),
                ],
            )
            .into(),
            env.clone(),
        ),
        Ok(TailExpressionResult::TailCall(TailCall::Owned(
            ExpressionBody::Identifier("+".to_string()).into(),
            vec![
                ExpressionBody::Primitive(Primitive::Integer(1)).into(),
                ExpressionBody::Primitive(Primitive::Integer(2)).into(),
                ExpressionBody::Primitive(Primitive::Integer(3)).into(),
            ],
            env,
            PhantomData
        )))
    );

    Ok(())
}
