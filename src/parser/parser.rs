#![allow(dead_code)]
use super::{
    lexer::Lexer,
    pair::GenericPair,
    pair::{PairIterItem, Pairable},
    Datum, DatumBody, DatumList, Result, SyntaxTemplateElement, Transformer,
};
use crate::error::ToLocated;
use crate::{environment::LexicalScope, error::*, parser::lexer::Token};
use crate::{interpreter::error::LogicError, parser::lexer::TokenData};
use fmt::Display;
use itertools::Itertools;
use std::{collections::HashSet, fmt, mem, rc::Rc};
use std::{
    iter::{repeat, FromIterator, Iterator, Peekable},
    path::PathBuf,
};

use either::Either;

use super::{
    error::SyntaxError, Primitive, SyntaxPattern, SyntaxPatternBody, SyntaxTemplate,
    SyntaxTemplateBody, UserDefinedTransformer,
};

pub type ParseResult = Result<Option<(Statement, Option<[u32; 2]>)>>;

#[derive(PartialEq, Debug, Clone)]
pub struct LibraryDefinition(pub LibraryName, pub Vec<Located<LibraryDeclaration>>);
impl ToLocated for LibraryDefinition {}
#[derive(PartialEq, Debug, Clone)]
pub enum LibraryDeclaration {
    ImportDeclaration(Located<ImportDeclaration>),
    Export(Vec<Located<ExportSpec>>),
    Begin(Vec<Statement>),
}
impl ToLocated for LibraryDeclaration {}

// r7rs:
// ⟨library name⟩ is a list whose members are identifiers and
// exact non-negative integers.
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum LibraryNameElement {
    Identifier(String),
    Integer(u32),
}

impl ToLocated for LibraryNameElement {}

impl Display for LibraryNameElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LibraryNameElement::Identifier(s) => write!(f, "{}", s),
            LibraryNameElement::Integer(i) => write!(f, "{}", i),
        }
    }
}

impl From<&str> for LibraryNameElement {
    fn from(s: &str) -> Self {
        Self::Identifier(s.to_string())
    }
}

impl From<u32> for LibraryNameElement {
    fn from(i: u32) -> Self {
        Self::Integer(i)
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct LibraryName(pub Vec<LibraryNameElement>);

#[macro_export]
macro_rules! library_name {
    ($($e:expr),+) => {
        LibraryName(vec![$($e.into()),+])
    };
}

impl ToLocated for LibraryName {}

impl FromIterator<LibraryNameElement> for LibraryName {
    fn from_iter<T: IntoIterator<Item = LibraryNameElement>>(iter: T) -> Self {
        Self(iter.into_iter().collect::<Vec<_>>())
    }
}

impl From<Vec<LibraryNameElement>> for LibraryName {
    fn from(value: Vec<LibraryNameElement>) -> Self {
        Self(value)
    }
}
impl From<LibraryNameElement> for LibraryName {
    fn from(value: LibraryNameElement) -> Self {
        Self(vec![value])
    }
}

impl Display for LibraryName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({})", self.0.iter().join(" "))
    }
}
#[test]
fn library_name_display() {
    assert_eq!(format!("{}", library_name!("foo", 0, "bar")), "(foo 0 bar)");
    assert_eq!(format!("{}", library_name!(1, 0, 2)), "(1 0 2)");
}
impl LibraryName {
    pub fn join(self, other: impl Into<Self>) -> Self {
        Self(
            self.0
                .into_iter()
                .chain(other.into().0.into_iter())
                .collect(),
        )
    }
}
#[test]
fn library_name_join() {
    assert_eq!(
        library_name!("a", "b").join(library_name!("c")),
        library_name!("a", "b", "c")
    );
    assert_eq!(
        library_name!("a", "b").join(LibraryNameElement::Integer(0)),
        library_name!("a", "b", 0)
    );
}

impl LibraryName {
    pub fn path(&self) -> PathBuf {
        self.0
            .iter()
            .map(|element| PathBuf::from(format!("{}", element)))
            .collect()
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum ExportSpec {
    Direct(String),
    Rename(String, String),
}
impl ToLocated for ExportSpec {}
#[derive(PartialEq, Debug, Clone)]
pub struct ImportDeclaration(pub Vec<ImportSet>);

impl ToLocated for ImportDeclaration {}

#[derive(PartialEq, Debug, Clone)]
pub enum Statement {
    ImportDeclaration(Located<ImportDeclaration>),
    Definition(Definition),
    SyntaxDefinition(SyntaxDef),
    Expression(Expression),
    LibraryDefinition(Located<LibraryDefinition>),
}

impl Statement {
    pub fn location(&self) -> Option<[u32; 2]> {
        match self {
            Statement::ImportDeclaration(located) => located.location,
            Statement::Definition(located) => located.location,
            Statement::SyntaxDefinition(located) => located.location,
            Statement::Expression(located) => located.location,
            Statement::LibraryDefinition(located) => located.location,
        }
    }
    pub fn expect_expression(self) -> Result<Expression> {
        let location = self.location();
        match self {
            Self::Expression(expression) => Ok(expression),
            _ => located_error!(
                SyntaxError::ExpectSomething("expression".to_string(), "other".to_string()),
                location
            ),
        }
    }
}

impl Into<Statement> for Expression {
    fn into(self) -> Statement {
        Statement::Expression(self)
    }
}

impl Into<Statement> for SyntaxDef {
    fn into(self) -> Statement {
        Statement::SyntaxDefinition(self)
    }
}

impl Into<Statement> for Definition {
    fn into(self) -> Statement {
        Statement::Definition(self)
    }
}

impl Into<Statement> for Located<ImportDeclaration> {
    fn into(self) -> Statement {
        Statement::ImportDeclaration(self)
    }
}

impl Into<Statement> for Located<LibraryDefinition> {
    fn into(self) -> Statement {
        Statement::LibraryDefinition(self)
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct DefinitionBody(pub String, pub Expression);

impl ToLocated for DefinitionBody {}

#[derive(PartialEq, Debug, Clone)]
pub struct SyntaxDefBody(pub String, pub UserDefinedTransformer);

impl ToLocated for SyntaxDefBody {}

pub type Definition = Located<DefinitionBody>;
pub type SyntaxDef = Located<SyntaxDefBody>;
pub type ImportSet = Located<ImportSetBody>;

#[derive(PartialEq, Debug, Clone)]
pub enum ImportSetBody {
    Direct(Located<LibraryName>),
    Only(Box<ImportSet>, Vec<String>),
    Except(Box<ImportSet>, Vec<String>),
    Prefix(Box<ImportSet>, String),
    Rename(Box<ImportSet>, Vec<(String, String)>),
}

impl ToLocated for ImportSetBody {}

pub type Expression = Located<ExpressionBody>;
#[derive(PartialEq, Debug, Clone)]
pub enum ExpressionBody {
    Symbol(String),
    Primitive(Primitive),
    Period,
    Assignment(String, Box<Expression>),
    Procedure(SchemeProcedure),
    ProcedureCall(Box<Expression>, Vec<Expression>),
    Conditional(Box<(Expression, Expression, Option<Expression>)>),
    Quote(Box<Datum>),
    Datum(Datum),
}

impl From<i32> for ExpressionBody {
    fn from(integer: i32) -> Self {
        ExpressionBody::Datum(DatumBody::Primitive(Primitive::Integer(integer)).into())
    }
}

impl ToLocated for ExpressionBody {}

impl From<Primitive> for ExpressionBody {
    fn from(p: Primitive) -> Self {
        ExpressionBody::Primitive(p)
    }
}

impl From<Primitive> for Expression {
    fn from(p: Primitive) -> Self {
        ExpressionBody::Primitive(p).into()
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum ParameterFormalsBody {
    Name(String),                             // (lambda x ...) or (define (f . x) ...)
    Pair(Box<GenericPair<ParameterFormals>>), // (lambda (...) ...) or (define (f ...) ...)
}

pub type ParameterFormals = Located<ParameterFormalsBody>;

impl ToLocated for ParameterFormalsBody {}

impl ParameterFormals {
    pub fn new_non_located(parameters: impl Iterator<Item = String>, last: Option<String>) -> Self {
        let fixed = parameters
            .map(|s| ParameterFormalsBody::Name(s).no_locate())
            .collect();
        match last {
            Some(last) => match fixed {
                GenericPair::Empty => ParameterFormalsBody::Name(last).no_locate(),
                list => ParameterFormalsBody::Pair(Box::new(GenericPair::cons(
                    ParameterFormalsBody::Pair(Box::new(list)).no_locate(),
                    ParameterFormalsBody::Name(last).no_locate(),
                )))
                .no_locate(),
            },
            None => ParameterFormalsBody::Pair(Box::new(fixed)).no_locate(),
        }
    }

    pub fn split(self) -> Result<(Vec<String>, Option<String>)> {
        Ok(match self.data {
            ParameterFormalsBody::Name(name) => (Vec::new(), Some(name)),
            ParameterFormalsBody::Pair(pair) => {
                let mut proper_list = Vec::new();
                let mut cdr = None;
                for item in pair.into_pair_iter() {
                    match item {
                        PairIterItem::Proper(ParameterFormals {
                            data: ParameterFormalsBody::Name(fixed),
                            ..
                        }) => proper_list.push(fixed),
                        PairIterItem::Improper(ParameterFormals {
                            data: ParameterFormalsBody::Name(last),
                            ..
                        }) => cdr = Some(last),
                        other => {
                            return located_error!(
                                SyntaxError::IllegalParameter(other.get_inside()),
                                other.get_inside().location
                            );
                        }
                    }
                }
                (proper_list, cdr)
            }
        })
    }

    pub fn len(&self) -> (usize, bool) {
        let mut fixed = 0;
        let variadic = self.iter_to_last(|_| fixed += 1).is_some();
        (fixed, variadic)
    }

    pub fn as_name(&self) -> String {
        match &self.data {
            ParameterFormalsBody::Name(name) => name.clone(),
            ParameterFormalsBody::Pair(_) => {
                unreachable!("parameter name can only be a identifier")
            }
        }
    }

    pub fn iter_to_last(
        &self,
        mut visitor: impl FnMut(&ParameterFormals) -> (),
    ) -> Option<&ParameterFormals> {
        let mut next: Option<&ParameterFormals> = Some(self);
        loop {
            match next.take().map(|p| p.either_pair_ref()) {
                Some(Either::Left(GenericPair::Some(car, cdr))) => {
                    visitor(&car);
                    next = Some(&cdr);
                }
                Some(Either::Right(improper)) => return Some(&improper),
                None | Some(Either::Left(GenericPair::Empty)) => return None,
            }
        }
    }

    pub fn append(&mut self, mut x: ParameterFormals) -> Result<()> {
        let mut next = self as *mut ParameterFormals;
        loop {
            match unsafe { &mut *next }.either_pair_mut() {
                Either::Left(GenericPair::Some(_, cdr)) => {
                    next = cdr as *mut ParameterFormals;
                }
                Either::Right(improper) => {
                    return error!(LogicError::InproperList(improper.to_string()))
                }
                _empty => {
                    mem::swap(&mut x, unsafe { &mut *next });
                    break Ok(());
                }
            }
        }
    }
}

impl Pairable for ParameterFormals {
    impl_located_pairable!(ParameterFormalsBody);
}

#[macro_export]
macro_rules! param_fixed {
    ($($x:expr),+) => {{
        use $crate::error::ToLocated;
        ParameterFormals::from(list![$(ParameterFormalsBody::Name($x.to_string()).no_locate()),*])
}};
    () => {
        ParameterFormals::from(list![])
    }
}

#[macro_export]
macro_rules! append_variadic_param {
    ($fixed:expr, $append:expr) => {{
        let mut pair = $fixed;
        use $crate::error::ToLocated;
        pair.append(ParameterFormalsBody::Name($append.to_string()).no_locate())?;
        pair
    }};
}

impl From<GenericPair<ParameterFormals>> for ParameterFormals {
    fn from(pair: GenericPair<ParameterFormals>) -> Self {
        ParameterFormalsBody::Pair(Box::new(pair)).no_locate()
    }
}

impl Display for ParameterFormalsBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParameterFormalsBody::Name(s) => write!(f, "{}", s),
            ParameterFormalsBody::Pair(pair) => write!(f, "{}", pair),
        }
    }
}

fn test_parameter_formals() -> Result<()> {
    let test_cases = vec![
        (
            vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
            Some("x".to_string()),
        ),
        (
            vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
            None,
        ),
        (vec![], Some("x".to_string())),
        (vec![], None),
    ];

    for (fixed, variadic) in test_cases.into_iter() {
        let parameters =
            ParameterFormals::new_non_located(fixed.clone().into_iter(), variadic.clone());
        assert_eq!(parameters.split()?, (fixed.clone(), variadic.clone()));
    }

    Ok(())
}

#[derive(PartialEq, Debug, Clone)]
pub struct SchemeProcedure(
    pub ParameterFormals,
    pub Vec<Definition>,
    pub Vec<Expression>,
);

impl SchemeProcedure {
    pub fn get_body_location(&self) -> Option<[u32; 2]> {
        let SchemeProcedure(_, defs, exprs) = self;
        defs.first()
            .and_then(|d| d.location)
            .or(exprs.first().and_then(|e| e.location))
    }
}

impl fmt::Display for SchemeProcedure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let SchemeProcedure(formals, ..) = self;
        write!(f, "(lambda {})", formals,)
    }
}

pub struct Parser<TokenIter: Iterator<Item = Result<Token>>> {
    pub current: Option<Token>,
    pub lexer: Peekable<TokenIter>,
    pub syntax_env: Rc<LexicalScope<Transformer>>,
    location: Option<[u32; 2]>,
}

impl<TokenIter: Iterator<Item = Result<Token>>> Iterator for Parser<TokenIter> {
    type Item = Result<Statement>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.parse(self.syntax_env.clone()) {
            Ok(Some(statement)) => Some(Ok(statement)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

fn create_syntax_binding() -> Rc<LexicalScope<Transformer>> {
    thread_local! {static BINDINGS: Rc<LexicalScope<Transformer>> = {
            let mut parser = Parser::from_lexer_primary_syntax(Lexer::from_char_stream(
                include_str!("grammar.sld").chars(),
            ));
            while parser.next().is_some() {} // force consume the parser iterator to bind all syntax.
            parser.syntax_env
        };
    }
    BINDINGS.with(|bindings| bindings.clone())
}

impl<TokenIter: Iterator<Item = Result<Token>>> Parser<TokenIter> {
    fn from_lexer_primary_syntax(lexer: TokenIter) -> Parser<TokenIter> {
        Self {
            current: None,
            lexer: lexer.peekable(),
            syntax_env: Rc::new(LexicalScope::new()),
            location: None,
        }
    }

    pub fn from_lexer(lexer: TokenIter) -> Parser<TokenIter> {
        Self {
            current: None,
            lexer: lexer.peekable(),
            syntax_env: create_syntax_binding(),
            location: None,
        }
    }

    pub fn parse_current(
        &mut self,
        syntax_env: &Rc<LexicalScope<Transformer>>,
    ) -> Result<Option<Statement>> {
        Ok(match self.current_datum()? {
            Some(datum) => Some(Self::transform_to_statement(datum, syntax_env)?),
            None => None,
        })
    }

    pub fn transform_to_statement(
        datum: Datum,
        syntax_env: &Rc<LexicalScope<Transformer>>,
    ) -> Result<Statement> {
        let location = datum.location;
        Ok(match datum.data {
            DatumBody::Primitive(p) => ExpressionBody::Primitive(p).locate(location).into(),
            DatumBody::Symbol(s) => ExpressionBody::Symbol(s).locate(location).into(),
            DatumBody::Pair(mut pair) => {
                // let mut iter = list.into_iter();
                let first = pair.pop();
                match first {
                    None => return error!(SyntaxError::EmptyCall),
                    Some(first) => {
                        let first = first.get_inside();

                        match &first.data {
                            DatumBody::Symbol(keyword) => match keyword.as_str() {
                                "define" => {
                                    Self::transform_definition(pair.into_iter(), syntax_env)?
                                        .locate(datum.location)
                                        .into()
                                }
                                "define-library" => {
                                    Self::transform_library(pair.into_iter(), syntax_env)?
                                        .locate(datum.location)
                                        .into()
                                }
                                "lambda" => Self::transform_lambda(pair.into_iter(), syntax_env)?
                                    .locate(datum.location)
                                    .into(),
                                "if" => Self::transform_condition(pair.into_iter(), syntax_env)?
                                    .locate(datum.location)
                                    .into(),
                                "import" => Self::transform_import_decl(pair.into_iter())?
                                    .locate(datum.location)
                                    .into(),
                                "quote" => Self::transform_quote(pair.into_iter())?
                                    .locate(datum.location)
                                    .into(),
                                "set!" => Self::transform_assignment(pair.into_iter(), syntax_env)?
                                    .locate(datum.location)
                                    .into(),
                                "define-syntax" => {
                                    Self::transform_syntax_definition(pair.into_iter(), syntax_env)?
                                        .locate(datum.location)
                                        .into()
                                }
                                keyword => {
                                    if let Some(transformer) =
                                        syntax_env.get(&first.expect_symbol()?)
                                    {
                                        let remained = DatumBody::Pair(pair).locate(location);
                                        let expanded_datum =
                                            transformer.transform(keyword, remained)?;
                                        Self::transform_to_statement(expanded_datum, syntax_env)?
                                    } else {
                                        Self::transform_procedure_call(
                                            first,
                                            pair.into_iter(),
                                            syntax_env,
                                        )?
                                        .locate(datum.location)
                                        .into()
                                    }
                                }
                            },
                            // Lambda expression
                            _ => {
                                Self::transform_procedure_call(first, pair.into_iter(), syntax_env)?
                                    .locate(datum.location)
                                    .into()
                            }
                        }
                    }
                }
            }
            other => ExpressionBody::Datum(Datum {
                data: other,
                location,
            })
            .locate(location)
            .into(),
        })
    }

    pub fn transform_to_expression(
        datum: Datum,
        syntax_env: &Rc<LexicalScope<Transformer>>,
    ) -> Result<Expression> {
        match Self::transform_to_statement(datum, syntax_env)? {
            Statement::Expression(expression) => Ok(expression),
            _ => error!(SyntaxError::ExpectSomething(
                "expression".to_string(),
                "other statement".to_string(),
            )),
        }
    }

    pub fn current_datum(&mut self) -> Result<Option<Datum>> {
        match self.current.take() {
            None => Ok(None),
            Some(current) => match current {
                Token { data, location } => Ok(match data {
                    TokenData::Primitive(p) => Datum {
                        data: DatumBody::Primitive(p),
                        location,
                    }
                    .into(),
                    TokenData::Identifier(a) => Datum {
                        data: DatumBody::Symbol(a),
                        location,
                    }
                    .into(),
                    TokenData::LeftParen => Some(self.current_list_or_pair()?),
                    TokenData::RightParen => {
                        return located_error!(SyntaxError::UnmatchedParentheses, location)
                    }
                    TokenData::VecConsIntro => self.vector()?.into(),
                    TokenData::Quote => {
                        self.advance(1)?;
                        self.parse_quoted()?
                    }
                    .into(),
                    other => return located_error!(SyntaxError::UnexpectedToken(other), location),
                }),
            },
        }
    }

    pub fn unwrap_non_end<T>(op: Option<T>) -> Result<T> {
        op.ok_or(ErrorData::from(SyntaxError::UnexpectedEnd).no_locate())
    }

    pub fn current_list_or_pair(&mut self) -> Result<Datum> {
        let mut head = Box::new(DatumList::Empty);
        let mut tail = head.as_mut();
        let list_location = self.location;
        let mut encounter_period = false;
        loop {
            match self.advance_unwrap(1)? {
                Token { data, location } => match data {
                    TokenData::Period => {
                        if encounter_period {
                            return located_error!(
                                SyntaxError::UnexpectedToken(TokenData::Period),
                                location.clone()
                            );
                        }
                        encounter_period = true;
                        continue;
                    }
                    TokenData::RightParen => break,
                    _ => {
                        let element = Self::unwrap_non_end(self.current_datum()?)?;
                        match tail {
                            DatumList::Empty => {
                                head = Box::new(DatumList::Some(
                                    element,
                                    Datum::from(DatumList::Empty),
                                ));
                                tail = head.as_mut();
                            }
                            DatumList::Some(_, cdr) => {
                                if encounter_period {
                                    *cdr = element;
                                    self.expect_next_nth(1, TokenData::RightParen)?;
                                    break;
                                }
                                assert_eq!(*cdr, Datum::from(DatumList::Empty));
                                let new_tail =
                                    DatumList::Some(element, Datum::from(DatumList::Empty));
                                *cdr = Datum::from(new_tail);
                                tail = cdr.either_pair_mut().left().unwrap();
                            }
                        }
                    }
                },
            }
        }
        Ok(DatumBody::Pair(head).locate(list_location))
    }

    pub fn parse_root(&mut self) -> Result<Option<Statement>> {
        self.parse(self.syntax_env.clone())
    }

    fn statement(&mut self, syntax_env: &Rc<LexicalScope<Transformer>>) -> Result<Statement> {
        match self.parse_current(syntax_env)? {
            Some(statement) => Ok(statement),
            None => located_error!(SyntaxError::UnexpectedEnd, self.location),
        }
    }
    pub fn parse(
        &mut self,
        syntax_env: Rc<LexicalScope<Transformer>>,
    ) -> Result<Option<Statement>> {
        self.advance(1)?;
        self.parse_current(&syntax_env)
    }

    fn transform_library(
        mut datums: impl Iterator<Item = Datum>,
        syntax_env: &Rc<LexicalScope<Transformer>>,
    ) -> Result<LibraryDefinition> {
        let library_name = Self::transform_library_name(
            Self::unwrap_non_end(datums.next())?
                .expect_list()?
                .into_iter(),
        )?;
        let library_declarations = datums
            .map(|datum| Self::transform_library_declaration(datum, syntax_env))
            .collect::<Result<_>>()?;
        Ok(LibraryDefinition(library_name, library_declarations))
    }

    fn transform_library_declaration(
        datum: Datum,
        syntax_env: &Rc<LexicalScope<Transformer>>,
    ) -> Result<Located<LibraryDeclaration>> {
        let location = datum.location;
        let mut iter = datum.expect_list()?.into_iter().peekable();
        Ok(match &Self::unwrap_non_end(iter.peek())?.data {
            DatumBody::Symbol(first) if first == "export" => LibraryDeclaration::Export(
                iter.skip(1)
                    .map(Self::transform_export_spec)
                    .collect::<Result<_>>()?,
            ),
            DatumBody::Symbol(first) if first == "begin" => LibraryDeclaration::Begin(
                iter.skip(1)
                    .map(|datum| Self::transform_to_statement(datum, syntax_env))
                    .collect::<Result<_>>()?,
            ),
            _ => LibraryDeclaration::ImportDeclaration(
                Self::transform_import_decl(iter.skip(1))?.locate(location),
            ),
        }
        .locate(location))
    }

    fn transform_export_spec(datum: Datum) -> Result<Located<ExportSpec>> {
        Ok(match datum.data {
            DatumBody::Symbol(ident) => ExportSpec::Direct(ident),
            DatumBody::Pair(list) => {
                let mut iter = list.into_iter();
                match Self::unwrap_non_end(iter.next())? {
                    Datum {
                        data: DatumBody::Symbol(ident),
                        ..
                    } if ident == "rename" => (),
                    o => return error!(SyntaxError::UnexpectedDatum(o)),
                };
                ExportSpec::Rename(
                    Self::transform_identifier(Self::unwrap_non_end(iter.next())?)?,
                    Self::transform_identifier(Self::unwrap_non_end(iter.next())?)?,
                )
            }
            _ => return error!(SyntaxError::UnexpectedDatum(datum)),
        }
        .locate(datum.location))
    }

    fn transform_identifier(datum: Datum) -> Result<String> {
        match datum.data {
            DatumBody::Symbol(ident) => Ok(ident.clone()),
            other => located_error!(
                SyntaxError::ExpectSomething("identifier".to_string(), other.to_string()),
                datum.location
            ),
        }
    }

    fn transform_identifier_pair(datum: Datum) -> Result<(String, String)> {
        let mut iter = datum.expect_list()?.into_iter();
        let car = Self::transform_identifier(Self::unwrap_non_end(iter.next())?)?;
        let cdr = Self::transform_identifier(Self::unwrap_non_end(iter.next())?)?;
        Ok((car, cdr))
    }

    fn expect_next_nth(&mut self, n: usize, tobe: TokenData) -> Result<()> {
        match self.advance_unwrap(n)? {
            Token { data, .. } if data == &tobe => Ok(()),
            Token { data, .. } => located_error!(
                SyntaxError::TokenMisMatch(tobe, Some(data.clone())),
                self.location
            ),
        }
    }

    // a lazy parser combinator to repeat the given parser
    fn repeat<'a, T>(
        &'a mut self,
        get_element: fn(&mut Self) -> Result<T>,
    ) -> impl Iterator<Item = Result<T>> + 'a
    where
        T: std::fmt::Debug + 'a,
    {
        repeat(())
            .map(move |_| match self.peek_next_token()?.map(|t| &t.data) {
                Some(TokenData::RightParen) => {
                    self.advance(1)?;
                    Ok(None)
                }
                None => located_error!(SyntaxError::UnexpectedEnd, self.location),
                _ => {
                    self.advance(1)?;
                    Some(get_element(self)).transpose()
                }
            })
            .map(|e| e.transpose())
            .take_while(|e| e.is_some())
            .map(|e| e.unwrap())
    }

    fn vector(&mut self) -> Result<Datum> {
        let vec = self.repeat(Self::datum).collect::<Result<_>>()?;
        Ok(self.locate(DatumBody::Vector(vec)))
    }

    fn transform_formals(args: Datum) -> Result<ParameterFormals> {
        let location = args.location;
        Ok(match args {
            Datum {
                data: DatumBody::Pair(pair),
                ..
            } => ParameterFormalsBody::Pair(Box::new(pair.map_ok(&mut |datum| {
                let sub_location = datum.location;
                Ok(
                    ParameterFormalsBody::Name(Self::transform_identifier(datum)?)
                        .locate(sub_location),
                )
            })?))
            .locate(location),
            single => {
                ParameterFormalsBody::Name(Self::transform_identifier(single)?).locate(location)
            }
        })
    }

    fn parse_quoted(&mut self) -> Result<Datum> {
        let quote_location = self.location;
        let inner = self.datum()?;
        Ok(Datum {
            location: quote_location,
            data: DatumBody::Pair(Box::new(list![
                Datum {
                    data: DatumBody::Symbol("quote".to_string()),
                    location: quote_location,
                },
                inner
            ])),
        })
    }

    fn transform_quote(mut datums: impl Iterator<Item = Datum>) -> Result<ExpressionBody> {
        Ok(ExpressionBody::Quote(Box::new(Self::unwrap_non_end(
            datums.next(),
        )?)))
    }

    fn datum(&mut self) -> Result<Datum> {
        let location = self.location;
        Ok(match &self.advance_unwrap(0)?.data {
            TokenData::LeftParen => self.current_list_or_pair()?,
            TokenData::VecConsIntro => {
                DatumBody::Vector(self.repeat(Self::datum).collect::<Result<_>>()?).locate(location)
            }
            TokenData::Identifier(symbol) => DatumBody::Symbol(symbol.clone()).locate(location),
            TokenData::Primitive(p) => DatumBody::Primitive(p.clone()).locate(location),
            other => return located_error!(SyntaxError::UnexpectedToken(other.clone()), location),
        })
    }

    fn transform_lambda(
        mut datums: impl Iterator<Item = Datum>,
        syntax_env: &Rc<LexicalScope<Transformer>>,
    ) -> Result<ExpressionBody> {
        let formals = Self::transform_formals(Self::unwrap_non_end(datums.next())?)?;
        let lambda_syntax_env = Rc::new(LexicalScope::new_child(syntax_env.clone()));
        let (definitions, expressions) =
            Self::transform_procedure_body(datums, &lambda_syntax_env)?;
        Ok(ExpressionBody::Procedure(SchemeProcedure(
            formals,
            definitions,
            expressions,
        )))
    }

    fn transform_procedure_body(
        datums: impl Iterator<Item = Datum>,
        syntax_env: &Rc<LexicalScope<Transformer>>,
    ) -> Result<(Vec<Definition>, Vec<Expression>)> {
        let mut definitions = vec![];
        let mut expressions = vec![];
        for datum in datums {
            let location = datum.location;
            let statement = Self::transform_to_statement(datum, syntax_env)?;
            match statement {
                Statement::Definition(def) => {
                    if expressions.is_empty() {
                        definitions.push(def)
                    } else {
                        return located_error!(
                            SyntaxError::InvalidDefinitionContext(def.data),
                            def.location
                        );
                    }
                }
                Statement::Expression(expr) => expressions.push(expr),
                _ => {
                    return located_error!(
                        SyntaxError::ExpectSomething(
                            "expression or definition".to_string(),
                            "other statement".to_string(),
                        ),
                        location
                    )
                }
            }
        }
        if expressions.is_empty() {
            return error!(SyntaxError::LambdaBodyNoExpression);
        }
        Ok((definitions, expressions))
    }

    fn transform_import_decl(datums: impl Iterator<Item = Datum>) -> Result<ImportDeclaration> {
        Ok(ImportDeclaration(
            datums
                .map(|import_set| Self::transform_import_set(import_set))
                .collect::<Result<_>>()?,
        ))
    }

    fn transform_condition(
        mut asts: impl Iterator<Item = Datum>,
        syntax_env: &Rc<LexicalScope<Transformer>>,
    ) -> Result<ExpressionBody> {
        let test = Self::transform_to_expression(Self::unwrap_non_end(asts.next())?, syntax_env)?;
        let consequent =
            Self::transform_to_expression(Self::unwrap_non_end(asts.next())?, syntax_env)?;
        let alternative = asts
            .next()
            .map(|datum| Self::transform_to_expression(datum, syntax_env))
            .transpose()?;
        Ok(ExpressionBody::Conditional(Box::new((
            test,
            consequent,
            alternative,
        ))))
    }

    fn transform_library_name_part(datum: Datum) -> Result<LibraryNameElement> {
        let location = datum.location;
        match datum.data {
            DatumBody::Symbol(identifier) => Ok(LibraryNameElement::Identifier(identifier)),
            DatumBody::Primitive(Primitive::Integer(i)) if i >= 0 => {
                Ok(LibraryNameElement::Integer(i as u32))
            }
            o => located_error!(SyntaxError::UnexpectedDatum(o.locate(location)), location),
        }
    }

    fn transform_library_name(datums: impl Iterator<Item = Datum>) -> Result<LibraryName> {
        Ok(LibraryName(
            datums
                .map(Self::transform_library_name_part)
                .collect::<Result<_>>()?,
        ))
    }

    fn transform_import_set(datum: Datum) -> Result<ImportSet> {
        let mut iter = datum.expect_list()?.into_iter().peekable();
        let first = Self::unwrap_non_end(iter.peek())?;
        let location = first.location;
        Ok(match Self::transform_identifier(first.clone())? {
            spec if spec == "only" => {
                iter.next();
                let sub_import = Self::transform_import_set(Self::unwrap_non_end(iter.next())?)?;
                let idents = iter
                    .map(Self::transform_identifier)
                    .collect::<Result<_>>()?;
                ImportSetBody::Only(Box::new(sub_import), idents)
            }
            spec if spec == "except" => {
                iter.next();
                let sub_import = Self::transform_import_set(Self::unwrap_non_end(iter.next())?)?;
                let idents = iter
                    .map(Self::transform_identifier)
                    .collect::<Result<_>>()?;
                ImportSetBody::Except(Box::new(sub_import), idents)
            }
            spec if spec == "prefix" => {
                iter.next();
                let sub_import = Self::transform_import_set(Self::unwrap_non_end(iter.next())?)?;
                let ident = Self::transform_identifier(Self::unwrap_non_end(iter.next())?)?;
                ImportSetBody::Prefix(Box::new(sub_import), ident)
            }
            spec if spec == "rename" => {
                iter.next();
                let sub_import = Self::transform_import_set(Self::unwrap_non_end(iter.next())?)?;
                let renaming = iter
                    .map(Self::transform_identifier_pair)
                    .collect::<Result<_>>()?;
                ImportSetBody::Rename(Box::new(sub_import), renaming)
            }
            _ => ImportSetBody::Direct(Self::transform_library_name(iter)?.locate(location)),
        }
        .locate(location))
    }

    fn transform_definition(
        mut datums: impl Iterator<Item = Datum>,
        syntax_env: &Rc<LexicalScope<Transformer>>,
    ) -> Result<DefinitionBody> {
        let first = Self::unwrap_non_end(datums.next())?;
        let location = first.location;
        match first.data {
            DatumBody::Symbol(symbol) => {
                let body = Self::transform_to_expression(
                    Self::unwrap_non_end(datums.next())?,
                    syntax_env,
                )?;
                Ok(DefinitionBody(symbol, body))
            }
            DatumBody::Pair(pair) => match *pair {
                GenericPair::Some(name, formals) => {
                    let location = name.location;
                    let name = Self::transform_identifier(name)?;
                    let formals = Self::transform_formals(formals)?;
                    let (defs, exprs) = Self::transform_procedure_body(datums, syntax_env)?;
                    let procedure =
                        ExpressionBody::Procedure(SchemeProcedure(formals, defs, exprs))
                            .locate(location);
                    Ok(DefinitionBody(name, procedure))
                }
                other => {
                    return located_error!(
                        SyntaxError::InvalidDefinition(Datum::from(other)),
                        location
                    )
                }
            },
            other => {
                return located_error!(SyntaxError::DefineNonSymbol(other.no_locate()), location)
            }
        }
    }

    fn transform_transformer(keyword: &String, datum: Datum) -> Result<UserDefinedTransformer> {
        // Skipping symbol 'syntax-rules'
        let mut iter = datum.expect_list()?.into_iter().skip(1);

        let first = Self::unwrap_non_end(iter.next())?;
        let location = first.location;

        let (ellipsis, pattern_literals) = match first.data {
            DatumBody::Symbol(ellipsis) => (
                Some(ellipsis),
                Self::unwrap_non_end(iter.next())?
                    .expect_list()?
                    .into_iter()
                    .map(Self::transform_identifier)
                    .collect::<Result<HashSet<_>>>()?,
            ),
            DatumBody::Pair(list) => (
                None,
                list.into_iter()
                    .map(Self::transform_identifier)
                    .collect::<Result<HashSet<_>>>()?,
            ),
            _ => return located_error!(SyntaxError::UnexpectedDatum(first), location),
        };

        let rules = iter
            .map(|datum| Self::transform_syntax_rule(keyword, datum))
            .collect::<Result<_>>()?;

        Ok(UserDefinedTransformer {
            ellipsis,
            literals: pattern_literals,
            rules,
        })
    }

    fn transform_syntax_definition(
        mut datums: impl Iterator<Item = Datum>,
        syntax_env: &Rc<LexicalScope<Transformer>>,
    ) -> Result<SyntaxDefBody> {
        let keyword = Self::transform_identifier(Self::unwrap_non_end(datums.next())?)?;
        let syntax_body =
            Self::transform_transformer(&keyword, Self::unwrap_non_end(datums.next())?)?;
        syntax_env.define(keyword.clone(), Transformer::Scheme(syntax_body.clone()));
        Ok(SyntaxDefBody(keyword, syntax_body))
    }

    fn transform_syntax_rule(
        keyword: &String,
        datum: Datum,
    ) -> Result<(SyntaxPattern, SyntaxTemplate)> {
        let mut iter = datum.expect_list()?.into_iter();
        let pattern = Self::transform_pattern_root(
            keyword,
            Self::unwrap_non_end(iter.next())?.expect_list()?,
        )?;
        let template = Self::transform_template(Self::unwrap_non_end(iter.next())?)?;
        Ok((pattern, template))
    }

    fn transform_pattern_root(
        keyword: &String,
        mut datum_list: DatumList,
    ) -> Result<SyntaxPattern> {
        let first = Self::unwrap_non_end(datum_list.pop())?.get_inside();
        let location = first.location;
        let providing_keyword = first.expect_symbol()?;
        if keyword != &providing_keyword {
            return located_error!(
                SyntaxError::MacroKeywordMissMatch(keyword.clone(), providing_keyword),
                location
            );
        }
        Ok(
            SyntaxPatternBody::Pair(Box::new(datum_list.map_ok(&mut Self::transform_pattern)?))
                .locate(location),
        )
    }

    fn transform_pattern(datum: Datum) -> Result<SyntaxPattern> {
        let location = datum.location;
        let data = match datum.data {
            DatumBody::Symbol(ident) if ident == "_" => SyntaxPatternBody::Underscore,
            DatumBody::Symbol(ident) if ident == "..." => SyntaxPatternBody::Ellipsis,
            DatumBody::Symbol(ident) => SyntaxPatternBody::Identifier(ident),
            DatumBody::Primitive(p) => SyntaxPatternBody::Primitive(p),
            DatumBody::Pair(list) => {
                SyntaxPatternBody::Pair(Box::new(list.map_ok(&mut Self::transform_pattern)?))
            }
            DatumBody::Vector(v) => SyntaxPatternBody::Vector(
                v.into_iter()
                    .map(Self::transform_pattern)
                    .collect::<Result<Vec<_>>>()?,
            ),
        };
        Ok(SyntaxPattern {
            data,
            location: location,
        })
    }

    fn collect_template_elements(
        mut datums: impl Iterator<Item = Datum>,
    ) -> Result<impl IntoIterator<Item = SyntaxTemplateElement>> {
        let mut last_template = None;
        let mut elements = vec![];
        while let Some(datum) = datums.next() {
            let location = datum.location;
            match datum.data {
                DatumBody::Symbol(symbol) if symbol == "..." => match last_template {
                    Some(SyntaxTemplateElement(template, false)) => {
                        elements.push(SyntaxTemplateElement(template, true));
                        last_template = None;
                    }
                    _ => {
                        return located_error!(
                            SyntaxError::UnexpectedDatum(
                                DatumBody::Symbol(symbol).locate(location)
                            ),
                            location
                        );
                    }
                },
                other => {
                    if let Some(last_template) = last_template {
                        elements.push(last_template)
                    };
                    last_template = Some(SyntaxTemplateElement(
                        Self::transform_template(Datum {
                            data: other,
                            location,
                        })?,
                        false,
                    ))
                }
            }
        }
        if let Some(last_template) = last_template {
            elements.push(last_template)
        };
        Ok(elements)
    }

    fn transform_template(datum: Datum) -> Result<SyntaxTemplate> {
        // TODO: replace with ellipsis tests
        // let element_transformer = |datum| {
        //     Ok(SyntaxTemplateElement(
        //         Self::transform_template(datum)?,
        //         false,
        //     ))
        // };
        let data = match datum.data {
            DatumBody::Symbol(ident) => SyntaxTemplateBody::Identifier(ident),
            DatumBody::Primitive(p) => SyntaxTemplateBody::Primitive(p),
            DatumBody::Pair(list) => {
                let elements = Self::collect_template_elements(list.into_iter())?
                    .into_iter()
                    .collect::<GenericPair<_>>();
                SyntaxTemplateBody::Pair(Box::new(elements))
            }
            DatumBody::Vector(vec) => SyntaxTemplateBody::Vector(
                Self::collect_template_elements(vec.into_iter())?
                    .into_iter()
                    .collect::<Vec<_>>(),
            ),
        };
        Ok(SyntaxTemplate {
            data,
            location: datum.location,
        })
    }

    fn transform_assignment(
        mut datums: impl Iterator<Item = Datum>,
        syntax_env: &Rc<LexicalScope<Transformer>>,
    ) -> Result<ExpressionBody> {
        let symbol = match Self::unwrap_non_end(datums.next())? {
            Datum {
                data: DatumBody::Symbol(symbol),
                ..
            } => symbol,
            other => return error!(SyntaxError::DefineNonSymbol(other)),
        };
        let body = Self::transform_to_expression(Self::unwrap_non_end(datums.next())?, syntax_env)?;
        Ok(ExpressionBody::Assignment(symbol, Box::new(body)))
    }

    fn transform_procedure_call(
        first: Datum,
        datum: impl Iterator<Item = Datum>,
        syntax_env: &Rc<LexicalScope<Transformer>>,
    ) -> Result<ExpressionBody> {
        Ok(ExpressionBody::ProcedureCall(
            Box::new(Self::transform_to_expression(first, syntax_env)?),
            datum
                .map(|datum| Self::transform_to_expression(datum, syntax_env))
                .collect::<Result<Vec<_>>>()?,
        ))
    }

    fn advance(&mut self, count: usize) -> Result<&mut Option<Token>> {
        for _ in 1..count {
            self.lexer.next();
        }
        if count > 0 {
            self.current = self.lexer.next().transpose()?;
            self.location = self.current.as_ref().and_then(|t| t.location);
        }
        Ok(&mut self.current)
    }

    fn advance_unwrap<'a>(&'a mut self, count: usize) -> Result<&'a mut Token> {
        let location = self.location;
        let token = self.advance(count)?;
        match token {
            Some(tok) => Ok(tok),
            None => located_error!(SyntaxError::UnexpectedEnd, location),
        }
    }

    fn advance_unwrap_take(&mut self, count: usize) -> Result<Token> {
        let token = self.advance(count)?.take();
        match token {
            Some(tok) => Ok(tok),
            None => located_error!(SyntaxError::UnexpectedEnd, self.location),
        }
    }

    fn peek_next_token(&mut self) -> Result<Option<&Token>> {
        match self.lexer.peek() {
            Some(ret) => match ret {
                Ok(t) => Ok(Some(t)),
                Err(e) => Err((*e).clone()),
            },
            None => Ok(None),
        }
    }

    fn locate<T: PartialEq>(&self, data: T) -> Located<T> {
        Located {
            data,
            location: self.location,
        }
    }
}

// macro_rules! match_expect_syntax {
//     ($value:expr, $type:pat => $inner: expr, $type_name:expr) => {
//         match $value {
//             $type => Ok($inner),
//             v => Err(SchemeError {
//                 location: None,
//                 category: ErrorType::Syntax,
//                 message: format!("expect a {}, got {}", $type_name, v),
//             }),
//         }
//     };
// }

#[cfg(test)]
pub fn simple_procedure(formals: ParameterFormals, expression: Expression) -> Expression {
    ExpressionBody::Procedure(SchemeProcedure(formals, vec![], vec![expression])).into()
}
#[test]
fn empty() -> Result<()> {
    let tokens = Vec::new();
    let mut parser = token_stream_to_parser(tokens.into_iter());
    assert_eq!(parser.parse_root(), Ok(None));
    Ok(())
}

fn expr_body_to_statement(t: ExpressionBody) -> Option<Statement> {
    Some(Located::from(t).into())
}

fn def_body_to_statement(t: DefinitionBody) -> Option<Statement> {
    Some(Located::from(t).into())
}

#[cfg(test)]
pub fn token_stream_to_parser(
    token_stream: impl Iterator<Item = Token>,
) -> Parser<impl Iterator<Item = Result<Token>>> {
    let mapped = token_stream.map(|t| -> Result<Token> { Ok(t) });
    Parser {
        current: None,
        lexer: mapped.peekable(),
        syntax_env: Rc::new(LexicalScope::new()),
        location: None,
    }
}

#[test]
fn integer() -> Result<()> {
    let tokens = convert_located(vec![TokenData::Primitive(Primitive::Integer(1))]);
    let mut parser = token_stream_to_parser(tokens.into_iter());
    let ast = parser.parse_root()?;
    assert_eq!(ast, expr_body_to_statement(Primitive::Integer(1).into()));
    Ok(())
}

#[test]
fn real_number() -> Result<()> {
    let tokens = convert_located(vec![TokenData::Primitive(Primitive::Real(
        "1.2".to_string(),
    ))]);
    let mut parser = token_stream_to_parser(tokens.into_iter());
    let ast = parser.parse_root()?;
    assert_eq!(
        ast,
        expr_body_to_statement(Primitive::Real("1.2".to_string()).into())
    );
    Ok(())
}

#[test]
fn rational() -> Result<()> {
    let tokens = convert_located(vec![TokenData::Primitive(Primitive::Rational(1, 2))]);
    let mut parser = token_stream_to_parser(tokens.into_iter());
    let ast = parser.parse_root()?;
    assert_eq!(
        ast,
        expr_body_to_statement(Primitive::Rational(1, 2).into())
    );
    Ok(())
}

#[test]
fn identifier() -> Result<()> {
    let tokens = convert_located(vec![TokenData::Identifier("test".to_string())]);
    let mut parser = token_stream_to_parser(tokens.into_iter());
    let ast = parser.parse_root()?;
    assert_eq!(
        ast,
        expr_body_to_statement(ExpressionBody::Symbol("test".to_string()))
    );
    Ok(())
}

#[test]
fn vector() -> Result<()> {
    let tokens = convert_located(vec![
        TokenData::VecConsIntro,
        TokenData::Primitive(Primitive::Integer(1)),
        TokenData::Primitive(Primitive::Boolean(false)),
        TokenData::RightParen,
    ]);
    let mut parser = token_stream_to_parser(tokens.into_iter());
    let ast = parser.parse_root()?;
    assert_eq!(
        ast,
        expr_body_to_statement(ExpressionBody::Datum(
            DatumBody::Vector(convert_located(vec![
                DatumBody::Primitive(Primitive::Integer(1)),
                DatumBody::Primitive(Primitive::Boolean(false))
            ]))
            .into()
        ))
    );
    Ok(())
}

#[test]
fn string() -> Result<()> {
    let tokens = convert_located(vec![TokenData::Primitive(Primitive::String(
        "hello world".to_string(),
    ))]);
    let mut parser = token_stream_to_parser(tokens.into_iter());
    let ast = parser.parse_root()?;
    assert_eq!(
        ast,
        expr_body_to_statement(Primitive::String("hello world".to_string()).into())
    );
    Ok(())
}

#[test]
fn procedure_call() -> Result<()> {
    let tokens = convert_located(vec![
        TokenData::LeftParen,
        TokenData::Identifier("+".to_string()),
        TokenData::Primitive(Primitive::Integer(1)),
        TokenData::Primitive(Primitive::Integer(2)),
        TokenData::Primitive(Primitive::Integer(3)),
        TokenData::RightParen,
    ]);
    let mut parser = token_stream_to_parser(tokens.into_iter());
    let ast = parser.parse_root()?;
    assert_eq!(
        ast,
        expr_body_to_statement(ExpressionBody::ProcedureCall(
            Box::new(ExpressionBody::Symbol("+".to_string()).into()),
            vec![
                Primitive::Integer(1).into(),
                Primitive::Integer(2).into(),
                Primitive::Integer(3).into(),
            ]
        ))
    );
    Ok(())
}

#[test]
fn unmatched_parantheses() {
    let tokens = convert_located(vec![
        TokenData::LeftParen,
        TokenData::Identifier("+".to_string()),
        TokenData::Primitive(Primitive::Integer(1)),
        TokenData::Primitive(Primitive::Integer(2)),
        TokenData::Primitive(Primitive::Integer(3)),
    ]);
    let mut parser = token_stream_to_parser(tokens.into_iter());
    assert_eq!(
        parser.parse_root(),
        located_error!(SyntaxError::UnexpectedEnd, None)
    );
}

#[test]
fn definition() -> Result<()> {
    {
        {
            let tokens = convert_located(vec![
                TokenData::LeftParen,
                TokenData::Identifier("define".to_string()),
                TokenData::Identifier("a".to_string()),
                TokenData::Primitive(Primitive::Integer(1)),
                TokenData::RightParen,
            ]);
            let mut parser = token_stream_to_parser(tokens.into_iter());
            let ast = parser.parse_root()?;
            assert_eq!(
                ast,
                def_body_to_statement(DefinitionBody(
                    "a".to_string(),
                    Primitive::Integer(1).into()
                ))
            );
        }
        {
            let tokens = convert_located(vec![
                TokenData::LeftParen,
                TokenData::Identifier("define".to_string()),
                TokenData::LeftParen,
                TokenData::Identifier("add".to_string()),
                TokenData::Identifier("x".to_string()),
                TokenData::Identifier("y".to_string()),
                TokenData::RightParen,
                TokenData::LeftParen,
                TokenData::Identifier("+".to_string()),
                TokenData::Identifier("x".to_string()),
                TokenData::Identifier("y".to_string()),
                TokenData::RightParen,
                TokenData::RightParen,
            ]);
            let mut parser = token_stream_to_parser(tokens.into_iter());
            let ast = parser.parse_root()?;
            assert_eq!(
                ast,
                def_body_to_statement(DefinitionBody(
                    "add".to_string(),
                    simple_procedure(
                        param_fixed!["x", "y"],
                        ExpressionBody::ProcedureCall(
                            Box::new(ExpressionBody::Symbol("+".to_string()).into()),
                            vec![
                                ExpressionBody::Symbol("x".to_string()).into(),
                                ExpressionBody::Symbol("y".to_string()).into(),
                            ]
                        )
                        .into()
                    )
                ))
            )
        }
        {
            let tokens = convert_located(vec![
                TokenData::LeftParen,
                TokenData::Identifier("define".to_string()),
                TokenData::LeftParen,
                TokenData::Identifier("add".to_string()),
                TokenData::Period,
                TokenData::Identifier("x".to_string()),
                TokenData::RightParen,
                TokenData::Identifier("x".to_string()),
                TokenData::RightParen,
            ]);
            let mut parser = token_stream_to_parser(tokens.into_iter());
            let ast = parser.parse_root()?;
            assert_eq!(
                ast,
                def_body_to_statement(DefinitionBody(
                    "add".to_string(),
                    simple_procedure(
                        append_variadic_param!(param_fixed![], "x"),
                        ExpressionBody::Symbol("x".to_string()).into()
                    )
                ))
            )
        }
        Ok(())
    }
}

#[test]
fn nested_procedure_call() -> Result<()> {
    let tokens = convert_located(vec![
        TokenData::LeftParen,
        TokenData::Identifier("+".to_string()),
        TokenData::Primitive(Primitive::Integer(1)),
        TokenData::LeftParen,
        TokenData::Identifier("-".to_string()),
        TokenData::Primitive(Primitive::Integer(2)),
        TokenData::Primitive(Primitive::Integer(3)),
        TokenData::RightParen,
        TokenData::RightParen,
    ]);
    let mut parser = token_stream_to_parser(tokens.into_iter());
    let ast = parser.parse_root()?;
    assert_eq!(
        ast,
        expr_body_to_statement(ExpressionBody::ProcedureCall(
            Box::new(ExpressionBody::Symbol("+".to_string()).into()),
            vec![
                Primitive::Integer(1).into(),
                ExpressionBody::ProcedureCall(
                    Box::new(ExpressionBody::Symbol("-".to_string()).into()),
                    vec![Primitive::Integer(2).into(), Primitive::Integer(3).into()]
                )
                .into(),
            ]
        ))
    );
    Ok(())
}

#[test]
fn lambda() -> Result<()> {
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("lambda".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("x".to_string()),
            TokenData::Identifier("y".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("+".to_string()),
            TokenData::Identifier("x".to_string()),
            TokenData::Identifier("y".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let ast = parser.parse_root()?;
        assert_eq!(
            ast,
            Some(Statement::Expression(simple_procedure(
                param_fixed!["x", "y"],
                ExpressionBody::ProcedureCall(
                    Box::new(ExpressionBody::Symbol("+".to_string()).into()),
                    vec![
                        ExpressionBody::Symbol("x".to_string()).into(),
                        ExpressionBody::Symbol("y".to_string()).into()
                    ]
                )
                .into()
            )))
        );
    }

    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("lambda".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("x".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("define".to_string()),
            TokenData::Identifier("y".to_string()),
            TokenData::Primitive(Primitive::Integer(1)),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("+".to_string()),
            TokenData::Identifier("x".to_string()),
            TokenData::Identifier("y".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let ast = parser.parse_root()?;
        assert_eq!(
            ast,
            Some(Statement::Expression(
                ExpressionBody::Procedure(SchemeProcedure(
                    param_fixed!["x".to_string()],
                    vec![Definition::from(DefinitionBody(
                        "y".to_string(),
                        Primitive::Integer(1).into()
                    ))],
                    vec![ExpressionBody::ProcedureCall(
                        Box::new(ExpressionBody::Symbol("+".to_string()).into()),
                        vec![
                            ExpressionBody::Symbol("x".to_string()).into(),
                            ExpressionBody::Symbol("y".to_string()).into()
                        ]
                    )
                    .into()]
                ))
                .into()
            ))
        );
    }

    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("lambda".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("x".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("define".to_string()),
            TokenData::Identifier("y".to_string()),
            TokenData::Primitive(Primitive::Integer(1)),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let err = parser.parse_root();
        assert_eq!(
            err,
            located_error!(SyntaxError::LambdaBodyNoExpression, None)
        );
    }

    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("lambda".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("x".to_string()),
            TokenData::Period,
            TokenData::Identifier("y".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("+".to_string()),
            TokenData::Identifier("x".to_string()),
            TokenData::Identifier("y".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let ast = parser.parse_root()?;
        assert_eq!(
            ast,
            Some(Statement::Expression(
                ExpressionBody::Procedure(SchemeProcedure(
                    append_variadic_param!(param_fixed!["x"], "y"),
                    vec![],
                    vec![ExpressionBody::ProcedureCall(
                        Box::new(ExpressionBody::Symbol("+".to_string()).into()),
                        vec![
                            ExpressionBody::Symbol("x".to_string()).into(),
                            ExpressionBody::Symbol("y".to_string()).into()
                        ]
                    )
                    .into()]
                ))
                .into()
            ))
        );
    }

    Ok(())
}

#[test]
fn conditional() -> Result<()> {
    let tokens = convert_located(vec![
        TokenData::LeftParen,
        TokenData::Identifier("if".to_string()),
        TokenData::Primitive(Primitive::Boolean(true)),
        TokenData::Primitive(Primitive::Integer(1)),
        TokenData::Primitive(Primitive::Integer(2)),
        TokenData::RightParen,
    ]);
    let mut parser = token_stream_to_parser(tokens.into_iter());
    assert_eq!(
        parser.parse_root()?,
        Some(Statement::Expression(
            ExpressionBody::Conditional(Box::new((
                Primitive::Boolean(true).into(),
                Primitive::Integer(1).into(),
                Some(Primitive::Integer(2).into())
            )))
            .into()
        ))
    );
    assert_eq!(parser.parse_root()?, None);
    Ok(())
}

#[test]
fn import_set() -> Result<()> {
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("import".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("foo".to_string()),
            TokenData::Primitive(Primitive::Integer(5)),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let import_set = parser.parse_root()?;
        assert_eq!(
            import_set,
            Some(Statement::ImportDeclaration(
                ImportDeclaration(vec![
                    ImportSetBody::Direct(library_name!("foo", 5).into()).into()
                ])
                .into()
            ))
        );
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("import".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("only".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("foo".to_string()),
            TokenData::Primitive(Primitive::Integer(5)),
            TokenData::RightParen,
            TokenData::Identifier("a".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let import_set = parser.parse_root()?;
        assert_eq!(
            import_set,
            Some(Statement::ImportDeclaration(
                ImportDeclaration(vec![ImportSetBody::Only(
                    Box::new(
                        ImportSetBody::Direct(library_name!("foo", 5).no_locate()).no_locate()
                    ),
                    vec!["a".to_string()]
                )
                .into()])
                .into()
            ))
        );
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("import".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("except".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("foo".to_string()),
            TokenData::Primitive(Primitive::Integer(5)),
            TokenData::RightParen,
            TokenData::Identifier("a".to_string()),
            TokenData::Identifier("b".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let import_set = parser.parse_root()?;
        assert_eq!(
            import_set,
            Some(Statement::ImportDeclaration(
                ImportDeclaration(vec![ImportSetBody::Except(
                    Box::new(
                        ImportSetBody::Direct(library_name!("foo", 5).no_locate()).no_locate()
                    ),
                    vec!["a".to_string(), "b".to_string()]
                )
                .into()])
                .into()
            ))
        );
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("import".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("prefix".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("foo".to_string()),
            TokenData::Primitive(Primitive::Integer(5)),
            TokenData::RightParen,
            TokenData::Identifier("a-".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let import_set = parser.parse_root()?;
        assert_eq!(
            import_set,
            Some(Statement::ImportDeclaration(
                ImportDeclaration(vec![ImportSetBody::Prefix(
                    Box::new(
                        ImportSetBody::Direct(library_name!("foo", 5).no_locate()).no_locate()
                    ),
                    "a-".to_string()
                )
                .into()])
                .into()
            ))
        );
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("import".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("rename".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("foo".to_string()),
            TokenData::Primitive(Primitive::Integer(5)),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("a".to_string()),
            TokenData::Identifier("b".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("c".to_string()),
            TokenData::Identifier("d".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let import_set = parser.parse_root()?;
        assert_eq!(
            import_set,
            Some(Statement::ImportDeclaration(
                ImportDeclaration(vec![ImportSetBody::Rename(
                    Box::new(
                        ImportSetBody::Direct(library_name!("foo", 5).no_locate()).no_locate()
                    ),
                    vec![
                        ("a".to_string(), "b".to_string()),
                        ("c".to_string(), "d".to_string()),
                    ]
                )
                .into()])
                .into()
            ))
        );
    }
    Ok(())
}
/* (import
(only (example-lib) a b)
(rename (example-lib) (old new))
) */

#[test]
fn import_declaration() -> Result<()> {
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("import".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("only".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("example-lib".to_string()),
            TokenData::RightParen,
            TokenData::Identifier("a".to_string()),
            TokenData::Identifier("b".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("rename".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("example-lib".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("old".to_string()),
            TokenData::Identifier("new".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
            TokenData::RightParen,
        ]);

        let mut parser = token_stream_to_parser(tokens.into_iter());
        let ast = parser.parse_root()?;
        assert_eq!(
            ast,
            Some(
                ImportDeclaration(convert_located(vec![
                    ImportSetBody::Only(
                        Box::new(
                            ImportSetBody::Direct(
                                LibraryName(vec![LibraryNameElement::Identifier(
                                    "example-lib".to_string()
                                )])
                                .into()
                            )
                            .into()
                        ),
                        vec!["a".to_string(), "b".to_string()]
                    ),
                    ImportSetBody::Rename(
                        Box::new(
                            ImportSetBody::Direct(
                                LibraryName(vec![LibraryNameElement::Identifier(
                                    "example-lib".to_string()
                                )])
                                .into()
                            )
                            .into()
                        ),
                        vec![("old".to_string(), "new".to_string())]
                    )
                ]))
                .no_locate()
                .into()
            )
        );
    }
    Ok(())
}

#[test]
fn literals() -> Result<()> {
    // symbol + list
    {
        let tokens = convert_located(vec![
            TokenData::Quote,
            TokenData::Primitive(Primitive::Integer(1)),
            TokenData::Quote,
            TokenData::Identifier("a".to_string()),
            TokenData::Quote,
            TokenData::LeftParen,
            TokenData::Primitive(Primitive::Integer(1)),
            TokenData::RightParen,
            TokenData::VecConsIntro,
            TokenData::Primitive(Primitive::Integer(1)),
            TokenData::RightParen,
            TokenData::Quote,
            TokenData::VecConsIntro,
            TokenData::Primitive(Primitive::Integer(1)),
            TokenData::RightParen,
        ]);
        let parser = token_stream_to_parser(tokens.into_iter());
        let asts = parser.collect::<Result<Vec<_>>>()?;
        assert_eq!(
            asts,
            vec![
                Statement::Expression(
                    ExpressionBody::Quote(Box::new(
                        DatumBody::Primitive(Primitive::Integer(1)).into()
                    ))
                    .into()
                ),
                Statement::Expression(
                    ExpressionBody::Quote(Box::new(DatumBody::Symbol("a".to_string()).into(),))
                        .into()
                ),
                Statement::Expression(
                    ExpressionBody::Quote(Box::new(
                        DatumBody::Pair(Box::new(list!(DatumBody::Primitive(Primitive::Integer(
                            1
                        ))
                        .no_locate())))
                        .into()
                    ))
                    .into()
                ),
                Statement::Expression(
                    ExpressionBody::Datum(
                        DatumBody::Vector(vec![
                            DatumBody::Primitive(Primitive::Integer(1)).no_locate()
                        ])
                        .into()
                    )
                    .into()
                ),
                Statement::Expression(
                    ExpressionBody::Quote(Box::new(
                        DatumBody::Vector(vec![
                            DatumBody::Primitive(Primitive::Integer(1)).no_locate()
                        ])
                        .into()
                    ))
                    .into()
                ),
            ]
        );
    }
    Ok(())
}

#[test]
fn macros() -> Result<()> {
    let tokens = convert_located(vec![
        TokenData::LeftParen,
        TokenData::Identifier("define-syntax".to_string()),
        TokenData::Identifier("begin".to_string()),
        TokenData::LeftParen,
        TokenData::Identifier("syntax-rules".to_string()),
        TokenData::LeftParen,
        TokenData::RightParen,
        TokenData::LeftParen,
        TokenData::LeftParen,
        TokenData::Identifier("begin".to_string()),
        TokenData::Identifier("exp".to_string()),
        TokenData::Identifier("...".to_string()),
        TokenData::RightParen,
        TokenData::LeftParen,
        TokenData::LeftParen,
        TokenData::Identifier("lambda".to_string()),
        TokenData::LeftParen,
        TokenData::RightParen,
        TokenData::Identifier("exp".to_string()),
        TokenData::Identifier("...".to_string()),
        TokenData::RightParen,
        TokenData::RightParen,
        TokenData::RightParen,
        TokenData::RightParen,
        TokenData::RightParen,
    ]);

    let parser = token_stream_to_parser(tokens.into_iter());
    let asts = parser.collect::<Result<Vec<_>>>()?;
    assert_eq!(
        asts,
        vec![Statement::SyntaxDefinition(
            SyntaxDefBody(
                "begin".to_owned(),
                UserDefinedTransformer {
                    ellipsis: None,
                    literals: HashSet::new(),
                    rules: vec![(
                        SyntaxPatternBody::Pair(Box::new(list![
                            SyntaxPatternBody::Identifier("exp".to_string()).into(),
                            SyntaxPatternBody::Ellipsis.into()
                        ]))
                        .into(),
                        SyntaxTemplateBody::Pair(Box::new(list![SyntaxTemplateElement(
                            SyntaxTemplateBody::Pair(Box::new(list![
                                SyntaxTemplateElement(
                                    SyntaxTemplateBody::Identifier("lambda".to_string()).into(),
                                    false
                                ),
                                SyntaxTemplateElement(
                                    SyntaxTemplateBody::Pair(Box::new(list![])).into(),
                                    false
                                ),
                                SyntaxTemplateElement(
                                    SyntaxTemplateBody::Identifier("exp".to_string()).into(),
                                    true
                                )
                            ]))
                            .into(),
                            false
                        )]))
                        .into()
                    )],
                }
            )
            .into()
        )]
    );

    Ok(())
}

#[test]
fn library_name() -> Result<()> {
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("import".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("f".to_string()),
            TokenData::Primitive(Primitive::Integer(5)),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let library_name = parser.parse_root()?;
        assert_eq!(
            library_name,
            Some(Statement::ImportDeclaration(
                ImportDeclaration(vec![
                    ImportSetBody::Direct(library_name!("f", 5).into()).into()
                ])
                .into()
            ))
        );
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("import".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("f".to_string()),
            TokenData::Primitive(Primitive::Integer(-5)),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let library_name = parser.parse_root();
        assert!(library_name.is_err());
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("import".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("f".to_string()),
            TokenData::Primitive(Primitive::String("haha".to_string())),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let library_name = parser.parse_root();
        assert!(library_name.is_err());
    }
    Ok(())
}
#[test]
fn export_spec() -> Result<()> {
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("define-library".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("foo".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("export".to_string()),
            TokenData::Identifier("a".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let export_spec = parser.parse_root()?;

        assert_eq!(
            export_spec,
            Some(Statement::LibraryDefinition(
                LibraryDefinition(
                    library_name!("foo").into(),
                    vec![LibraryDeclaration::Export(vec![
                        ExportSpec::Direct("a".to_string()).into()
                    ])
                    .into()]
                )
                .into()
            ))
        );
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("define-library".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("foo".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("export".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("rename".to_string()),
            TokenData::Identifier("a".to_string()),
            TokenData::Identifier("b".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let export_spec = parser.parse_root()?;
        assert_eq!(
            export_spec,
            Some(Statement::LibraryDefinition(
                LibraryDefinition(
                    library_name!("foo").into(),
                    vec![LibraryDeclaration::Export(vec![ExportSpec::Rename(
                        "a".to_string(),
                        "b".to_string()
                    )
                    .into()])
                    .into()]
                )
                .into()
            ))
        );
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("define-library".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("foo".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("export".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("c".to_string()),
            TokenData::Identifier("a".to_string()),
            TokenData::Identifier("b".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        assert!(parser.parse_root().is_err());
    }
    Ok(())
}
#[test]
fn library_declaration() -> Result<()> {
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("define-library".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("foo".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("import".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("a".to_string()),
            TokenData::Identifier("b".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());

        let library_declaration = parser.parse_root()?;
        assert_eq!(
            library_declaration,
            Some(Statement::LibraryDefinition(
                LibraryDefinition(
                    library_name!("foo").into(),
                    vec![LibraryDeclaration::ImportDeclaration(
                        ImportDeclaration(vec![ImportSetBody::Direct(
                            library_name!("a", "b").no_locate()
                        )
                        .no_locate()])
                        .no_locate()
                    )
                    .into()]
                )
                .into()
            ))
        );
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("define-library".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("foo".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("export".to_string()),
            TokenData::Identifier("a".to_string()),
            TokenData::Identifier("b".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let library_declaration = parser.parse_root()?;
        assert_eq!(
            library_declaration,
            Some(Statement::LibraryDefinition(
                LibraryDefinition(
                    library_name!("foo").into(),
                    vec![LibraryDeclaration::Export(vec![
                        ExportSpec::Direct("a".to_string()).no_locate(),
                        ExportSpec::Direct("b".to_string()).no_locate()
                    ])
                    .into()]
                )
                .into()
            ))
        );
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("define-library".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("foo".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("begin".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("define".to_string()),
            TokenData::Identifier("s".to_string()),
            TokenData::Primitive(Primitive::String("a".to_string())),
            TokenData::RightParen,
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let library_declaration = parser.parse_root()?;
        assert_eq!(
            library_declaration,
            Some(Statement::LibraryDefinition(
                LibraryDefinition(
                    library_name!("foo").into(),
                    vec![LibraryDeclaration::Begin(vec![Statement::Definition(
                        DefinitionBody(
                            "s".to_string(),
                            ExpressionBody::Primitive(Primitive::String("a".to_string()))
                                .no_locate()
                        )
                        .no_locate()
                    )])
                    .into()]
                )
                .into()
            ))
        );
    }
    Ok(())
}
/*
(define-library (lib-a 0 base)
    (import (lib-b))
    (begin
        (define c 0)
        (define d 1)
    )
    (export c (rename d e) f)
    (begin
        (define f 2)
    )
)
*/
#[test]
fn library() {
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("define-library".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("lib-a".to_string()),
            TokenData::Primitive(Primitive::Integer(0)),
            TokenData::Identifier("base".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("import".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("lib-b".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("begin".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("define".to_string()),
            TokenData::Identifier("c".to_string()),
            TokenData::Primitive(Primitive::Integer(0)),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("define".to_string()),
            TokenData::Identifier("d".to_string()),
            TokenData::Primitive(Primitive::Integer(1)),
            TokenData::RightParen,
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("export".to_string()),
            TokenData::Identifier("c".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("rename".to_string()),
            TokenData::Identifier("d".to_string()),
            TokenData::Identifier("e".to_string()),
            TokenData::RightParen,
            TokenData::Identifier("f".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("begin".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("define".to_string()),
            TokenData::Identifier("f".to_string()),
            TokenData::Primitive(Primitive::Integer(2)),
            TokenData::RightParen,
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        let ast = parser.parse_root();

        assert_eq!(
            ast,
            Ok(Some(Statement::LibraryDefinition(
                LibraryDefinition(
                    library_name!("lib-a", 0, "base").into(),
                    vec![
                        LibraryDeclaration::ImportDeclaration(
                            ImportDeclaration(vec![ImportSetBody::Direct(
                                LibraryName(vec![LibraryNameElement::Identifier(
                                    "lib-b".to_string()
                                )])
                                .into()
                            )
                            .into()])
                            .no_locate()
                        )
                        .into(),
                        LibraryDeclaration::Begin(vec![
                            Statement::Definition(
                                DefinitionBody(
                                    "c".to_string(),
                                    ExpressionBody::Primitive(Primitive::Integer(0)).into()
                                )
                                .into()
                            ),
                            Statement::Definition(
                                DefinitionBody(
                                    "d".to_string(),
                                    ExpressionBody::Primitive(Primitive::Integer(1)).into()
                                )
                                .into()
                            )
                        ])
                        .into(),
                        LibraryDeclaration::Export(vec![
                            ExportSpec::Direct("c".to_string()).into(),
                            ExportSpec::Rename("d".to_string(), "e".to_string()).into(),
                            ExportSpec::Direct("f".to_string()).into(),
                        ])
                        .into(),
                        LibraryDeclaration::Begin(vec![Statement::Definition(
                            DefinitionBody(
                                "f".to_string(),
                                ExpressionBody::Primitive(Primitive::Integer(2)).into()
                            )
                            .into()
                        ),])
                        .into()
                    ]
                )
                .no_locate()
            )))
        )
    }
}
