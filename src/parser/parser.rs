#![allow(dead_code)]
use super::Result;
use crate::interpreter::library::LibraryName;
use crate::parser::lexer::TokenData;
use crate::{error::*, interpreter::library::LibraryNameElement, parser::lexer::Token};
use fmt::Display;
use itertools::join;
use std::iter::{repeat, Iterator, Peekable};
use std::{fmt, ops::Deref};

use super::{
    error::SyntaxError, Primitive, SyntaxPattern, SyntaxPatternBody, SyntaxTemplate,
    SyntaxTemplateBody, Transformer,
};

pub type ParseResult = Result<Option<(Statement, Option<[u32; 2]>)>>;

pub(crate) fn join_displayable(iter: impl IntoIterator<Item = impl fmt::Display>) -> String {
    join(iter.into_iter().map(|d| format!("{}", d)), " ")
}

#[derive(PartialEq, Debug, Clone)]
pub struct LibraryDefinition(
    pub Located<LibraryName>,
    pub Vec<Located<LibraryDeclaration>>,
);
impl ToLocated for LibraryDefinition {}
#[derive(PartialEq, Debug, Clone)]
pub enum LibraryDeclaration {
    ImportDeclaration(Located<ImportDeclaration>),
    Export(Vec<Located<ExportSpec>>),
    Begin(Vec<Statement>),
}
impl ToLocated for LibraryDeclaration {}
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
    fn location(&self) -> Option<[u32; 2]> {
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
                SyntaxError::ExpectSomething("expression".to_string()),
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
pub struct SyntaxDefBody(pub String, pub Transformer);

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
    Identifier(String),
    Primitive(Primitive),
    Period,
    List(Vec<Expression>),
    Vector(Vec<Expression>),
    Assignment(String, Box<Expression>),
    Procedure(SchemeProcedure),
    ProcedureCall(Box<Expression>, Vec<Expression>),
    Conditional(Box<(Expression, Expression, Option<Expression>)>),
    Quote(Box<Expression>),
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

#[derive(Clone, Debug, PartialEq)]
pub struct ParameterFormals(pub Vec<String>, pub Option<String>);

impl ParameterFormals {
    pub fn new() -> ParameterFormals {
        Self(Vec::new(), None)
    }
}

impl Display for ParameterFormals {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.0.len() {
            0 => match &self.1 {
                None => write!(f, "()"),
                Some(variadic) => write!(f, "{}", variadic),
            },
            _ => match &self.1 {
                Some(last) => write!(f, "({} . {})", self.0.join(" "), last),
                None => write!(f, "({})", self.0.join(" ")),
            },
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct SchemeProcedure(
    pub ParameterFormals,
    pub Vec<Definition>,
    pub Vec<Expression>,
);

impl fmt::Display for SchemeProcedure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let SchemeProcedure(formals, ..) = self;
        write!(f, "(lambda {})", formals,)
    }
}

pub struct Parser<TokenIter: Iterator<Item = Result<Token>>> {
    pub current: Option<Token>,
    pub lexer: Peekable<TokenIter>,
    location: Option<[u32; 2]>,
}

impl<TokenIter: Iterator<Item = Result<Token>>> Iterator for Parser<TokenIter> {
    type Item = Result<Statement>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.parse() {
            Ok(Some(statement)) => Some(Ok(statement)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

impl<TokenIter: Iterator<Item = Result<Token>>> Parser<TokenIter> {
    pub fn from_lexer(lexer: TokenIter) -> Parser<TokenIter> {
        Self {
            current: None,
            lexer: lexer.peekable(),
            location: None,
        }
    }
    pub fn parse_current(&mut self) -> Result<Option<Statement>> {
        match self.current.take() {
            Some(Token { data, location }) => Ok(Some(match data {
                TokenData::Primitive(p) => Expression {
                    data: ExpressionBody::Primitive(p),
                    location,
                }
                .into(),
                TokenData::Identifier(a) => Expression {
                    data: ExpressionBody::Identifier(a),
                    location,
                }
                .into(),
                TokenData::LeftParen => match self.peek_next_token()? {
                    Some(Token {
                        data: TokenData::Identifier(ident),
                        ..
                    }) => match ident.as_str() {
                        "lambda" => self.lambda()?.into(),
                        "quote" => {
                            self.advance(2)?;
                            let quoted = self.quote()?;
                            self.expect_next_nth(1, TokenData::RightParen)?;
                            quoted.into()
                        }
                        "define" => self.definition()?.into(),
                        "define-syntax" => self.def_syntax()?.into(),
                        "define-library" => self.library()?.into(),
                        "set!" => self.assginment()?.into(),
                        "import" => self.import_declaration()?.into(),
                        "if" => self.condition()?.into(),
                        _ => self.procedure_call()?.into(),
                    },
                    Some(Token {
                        data: TokenData::RightParen,
                        location,
                    }) => return located_error!(SyntaxError::EmptyCall, *location),
                    _ => self.procedure_call()?.into(),
                },
                TokenData::RightParen => {
                    return located_error!(SyntaxError::UnmatchedParentheses, location)
                }
                TokenData::VecConsIntro => self.vector()?.into(),
                TokenData::Quote => {
                    self.advance(1)?;
                    self.quote()?.into()
                }
                TokenData::Period => Expression {
                    data: ExpressionBody::Period,
                    location,
                }
                .into(),
                _ => panic!("unsupported grammar"),
            })),
            None => Ok(None),
        }
    }

    pub fn current_expression(&mut self) -> Result<Expression> {
        match self.parse_current()? {
            Some(Statement::Expression(expr)) => Ok(expr),
            _ => located_error!(
                SyntaxError::ExpectSomething("expression".to_owned()),
                self.location
            ),
        }
    }

    pub fn parse_root(&mut self) -> Result<Option<(Statement, Option<[u32; 2]>)>> {
        Ok(self
            .parse()?
            .and_then(|statement| Some((statement, self.location))))
    }

    fn statement(&mut self) -> Result<Statement> {
        match self.parse_current()? {
            Some(statement) => Ok(statement),
            None => located_error!(SyntaxError::UnexpectedEnd, self.location),
        }
    }
    pub fn parse(&mut self) -> Result<Option<Statement>> {
        self.advance(1)?;
        self.parse_current()
    }
    fn library(&mut self) -> Result<Located<LibraryDefinition>> {
        let location = self.location;
        self.advance(2)?;
        match self.current.as_ref().map(Deref::deref) {
            Some(TokenData::LeftParen) => {
                let library_name = self.library_name()?;
                let library_declarations: Result<Vec<_>> =
                    self.repeat(Self::library_declaration).collect();
                Ok(LibraryDefinition(library_name, library_declarations?).locate(location))
            }
            other => located_error!(
                SyntaxError::TokenMisMatch(TokenData::LeftParen, other.map(Clone::clone)),
                self.location
            ),
        }
    }

    fn library_declaration(&mut self) -> Result<Located<LibraryDeclaration>> {
        let location = self.location;
        match self.current.take().map(Located::extract_data) {
            Some(TokenData::LeftParen) => {
                let token = self.peek_next_token()?;
                match token.map(Deref::deref) {
                    Some(TokenData::Identifier(identifier)) => match identifier.as_str() {
                        "import" => Ok(LibraryDeclaration::ImportDeclaration(
                            self.import_declaration()?,
                        )
                        .locate(location)),
                        "export" => {
                            self.advance(1)?;
                            Ok(Located::from(LibraryDeclaration::Export(
                                self.repeat(Self::export_spec).collect::<Result<Vec<_>>>()?,
                            )))
                        }
                        "begin" => {
                            self.advance(1)?;
                            Ok(Located::from(LibraryDeclaration::Begin(
                                self.repeat(Self::statement).collect::<Result<Vec<_>>>()?,
                            )))
                        }
                        _ => located_error!(
                            SyntaxError::ExpectSomething("import/export/begin".to_string()),
                            token.unwrap().location
                        ),
                    },
                    Some(_) => located_error!(
                        SyntaxError::ExpectSomething("import/export/begin".to_string()),
                        token.unwrap().location
                    ),
                    None => located_error!(SyntaxError::UnexpectedEnd, location),
                }
            }
            other => located_error!(
                SyntaxError::TokenMisMatch(TokenData::LeftParen, other),
                location
            ),
        }
    }

    fn export_spec(&mut self) -> Result<Located<ExportSpec>> {
        let location = self.location;
        match self.current.take().map(Located::extract_data) {
            Some(TokenData::LeftParen) => {
                self.advance(1)?;
                self.expect_next_nth(0, TokenData::Identifier("rename".to_string()))?;
                let from = self.next_identifier()?;
                let to = self.next_identifier()?;
                self.expect_next_nth(1, TokenData::RightParen)?;
                Ok(Located::from(ExportSpec::Rename(from, to)))
            }
            Some(TokenData::Identifier(identifier)) => {
                Ok(Located::from(ExportSpec::Direct(identifier)))
            }
            Some(other) => located_error!(SyntaxError::UnexpectedToken(other), location),
            None => located_error!(SyntaxError::UnexpectedEnd, location),
        }
    }

    fn expression(&mut self) -> Result<Expression> {
        self.advance(1)?;
        self.current_expression()
    }

    fn next_identifier(&mut self) -> Result<String> {
        self.advance(1)?;
        self.current_identifier()
    }

    fn current_identifier(&mut self) -> Result<String> {
        match self.current.as_ref().map(|t| &t.data) {
            Some(TokenData::Identifier(ident)) => Ok(ident.clone()),
            _ => located_error!(
                SyntaxError::ExpectSomething("identifier".to_string()),
                self.location
            ),
        }
    }

    fn current_identifier_pair(&mut self) -> Result<(String, String)> {
        self.expect_next_nth(0, TokenData::LeftParen)?;
        let car = self.next_identifier()?;
        let cdr = self.next_identifier()?;
        self.expect_next_nth(1, TokenData::RightParen)?;
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

    fn vector(&mut self) -> Result<Expression> {
        let vec = self.repeat(Self::datum).collect::<Result<_>>()?;
        Ok(self.locate(ExpressionBody::Vector(vec)))
    }

    fn procedure_formals(&mut self) -> Result<ParameterFormals> {
        let mut formals = ParameterFormals::new();
        loop {
            match self.peek_next_token()?.map(|t| &t.data) {
                Some(TokenData::RightParen) => {
                    self.advance(1)?;
                    break Ok(formals);
                }
                Some(TokenData::Period) => {
                    self.advance(2)?;
                    formals.1 = Some(self.current_identifier()?);
                }
                None => return located_error!(SyntaxError::UnexpectedEnd, self.location),
                _ => {
                    self.advance(1)?;
                    let parameter = self.current_identifier()?;
                    formals.0.push(parameter);
                }
            }
        }
    }

    fn quote(&mut self) -> Result<Expression> {
        let inner = self.datum()?;
        Ok(Expression {
            data: ExpressionBody::Quote(Box::new(inner)),
            location: self.location,
        })
    }

    fn datum(&mut self) -> Result<Expression> {
        Ok(match self.current.as_ref().map(|t| &t.data) {
            Some(TokenData::LeftParen) => {
                let seq = self.repeat(Self::datum).collect::<Result<_>>()?;
                Expression {
                    data: ExpressionBody::List(seq),
                    location: self.location,
                }
            }
            Some(TokenData::VecConsIntro) => {
                let seq = self.repeat(Self::datum).collect::<Result<_>>()?;
                Expression {
                    data: ExpressionBody::Vector(seq),
                    location: self.location,
                }
            }
            None => return located_error!(SyntaxError::UnexpectedEnd, self.location),
            _ => self.current_expression()?,
        })
    }

    fn lambda(&mut self) -> Result<Expression> {
        let mut formals = ParameterFormals::new();
        match self.advance(2)?.take().map(|t| t.data) {
            Some(TokenData::Identifier(ident)) => formals.1 = Some(ident),
            Some(TokenData::LeftParen) => {
                formals = self.procedure_formals()?;
            }
            _ => {
                return located_error!(
                    SyntaxError::ExpectSomething("lambda formals".to_string()),
                    self.location
                );
            }
        }
        self.procedure_body(formals)
    }

    fn procedure_body(&mut self, formals: ParameterFormals) -> Result<Expression> {
        let body_location = self.location;
        let statements = self.repeat(Self::parse_current);
        let mut definitions = vec![];
        let mut expressions = vec![];
        for statement in statements {
            match statement? {
                Some(Statement::Definition(def)) => {
                    if expressions.is_empty() {
                        definitions.push(def)
                    } else {
                        return located_error!(
                            SyntaxError::InvalidDefinitionContext(def.data),
                            def.location
                        );
                    }
                }
                Some(Statement::Expression(expr)) => expressions.push(expr),
                None => return located_error!(SyntaxError::LambdaBodyNoExpression, body_location),
                _ => {
                    return located_error!(
                        SyntaxError::ExpectSomething("expression or definition".to_string()),
                        body_location
                    )
                }
            }
        }
        if expressions.is_empty() {
            return located_error!(SyntaxError::LambdaBodyNoExpression, self.location);
        }
        Ok(self.locate(ExpressionBody::Procedure(SchemeProcedure(
            formals,
            definitions,
            expressions,
        ))))
    }

    fn import_declaration(&mut self) -> Result<Located<ImportDeclaration>> {
        let location = self.location;
        self.expect_next_nth(1, TokenData::Identifier("import".to_string()))?;
        Ok(
            ImportDeclaration(self.repeat(Self::import_set).collect::<Result<_>>()?)
                .locate(location),
        )
    }

    fn condition(&mut self) -> Result<Expression> {
        self.advance(1)?;
        let test = self.expression()?;
        let consequent = self.expression()?;
        match self.advance_unwrap(1)?.data {
            TokenData::RightParen => Ok(Expression {
                data: ExpressionBody::Conditional(Box::new((test, consequent, None))),
                location: self.location,
            }),
            _ => {
                let alternative = self.current_expression()?;
                self.expect_next_nth(1, TokenData::RightParen)?;
                Ok(Expression {
                    data: ExpressionBody::Conditional(Box::new((
                        test,
                        consequent,
                        Some(alternative),
                    ))),
                    location: self.location,
                })
            }
        }
    }

    fn library_name_element(&mut self) -> Result<Located<LibraryNameElement>> {
        let location = self.location;
        match self.current.take() {
            Some(token) => match token.data {
                TokenData::Identifier(identifier) => {
                    Ok(LibraryNameElement::Identifier(identifier).locate(self.location))
                }
                TokenData::Primitive(Primitive::Integer(i)) => {
                    if i >= 0 {
                        Ok(LibraryNameElement::Integer(i as u32).locate(self.location))
                    } else {
                        located_error!(
                            SyntaxError::ExpectSomething("non-negative integer".to_string()),
                            location
                        )
                    }
                }
                token => located_error!(SyntaxError::UnexpectedToken(token), self.location),
            },
            None => located_error!(SyntaxError::UnexpectedEnd, self.location),
        }
    }
    fn library_name(&mut self) -> Result<Located<LibraryName>> {
        let location = self.location;
        match self.current.take().map(Located::extract_data) {
            Some(TokenData::LeftParen) => Ok(LibraryName(
                self.repeat(Self::library_name_element)
                    .collect::<Result<Vec<Located<LibraryNameElement>>>>()?
                    .into_iter()
                    .map(Located::extract_data)
                    .collect(),
            )
            .locate(location)),
            other => located_error!(
                SyntaxError::TokenMisMatch(TokenData::LeftParen, other),
                location
            )?,
        }
    }

    fn import_set(&mut self) -> Result<ImportSet> {
        let location = self.location;
        Ok(match self.current.as_ref().map(Deref::deref) {
            Some(TokenData::LeftParen) => match self.peek_next_token()?.map(|token| &token.data) {
                Some(TokenData::Identifier(ident)) => match ident.as_str() {
                    "only" => {
                        self.advance(2)?;
                        ImportSet {
                            data: ImportSetBody::Only(
                                Box::new(self.import_set()?),
                                self.repeat(Self::current_identifier)
                                    .collect::<Result<_>>()?,
                            ),
                            location,
                        }
                    }
                    "except" => {
                        self.advance(2)?;
                        ImportSet {
                            data: ImportSetBody::Except(
                                Box::new(self.import_set()?),
                                self.repeat(Self::current_identifier)
                                    .collect::<Result<_>>()?,
                            ),
                            location,
                        }
                    }
                    "prefix" => {
                        self.advance(2)?;
                        ImportSet {
                            data: ImportSetBody::Prefix(
                                Box::new(self.import_set()?),
                                self.next_identifier()?,
                            ),
                            location,
                        }
                    }
                    "rename" => {
                        self.advance(2)?;
                        ImportSet {
                            data: ImportSetBody::Rename(
                                Box::new(self.import_set()?),
                                self.repeat(Self::current_identifier_pair)
                                    .collect::<Result<_>>()?,
                            ),
                            location,
                        }
                    }
                    _ => ImportSetBody::Direct(self.library_name()?).locate(location),
                },
                _ => ImportSetBody::Direct(self.library_name()?).locate(location),
            },
            other => located_error!(
                SyntaxError::TokenMisMatch(TokenData::LeftParen, other.map(Clone::clone)),
                location
            )?,
        })
    }

    fn definition(&mut self) -> Result<Definition> {
        let location = self.location;
        let current = self.advance(2)?.take().map(|t| t.data);
        match current {
            Some(TokenData::Identifier(identifier)) => {
                let expr = self.expression()?;
                self.expect_next_nth(1, TokenData::RightParen)?;
                Ok(Definition {
                    data: DefinitionBody(identifier, expr),
                    location,
                })
            }
            Some(TokenData::LeftParen) => {
                let identifier = self.next_identifier()?;
                let mut formals = ParameterFormals::new();
                match self.peek_next_token()?.map(|t| &t.data) {
                    Some(TokenData::Period) => {
                        self.advance(2)?;
                        formals.1 = Some(self.current_identifier()?);
                        self.advance(1)?;
                    }
                    _ => formals = self.procedure_formals()?,
                }
                let body = self.procedure_body(formals)?;
                Ok(Definition::from(DefinitionBody(identifier, body)))
            }
            _ => located_error!(SyntaxError::IllegalDefinition, self.location),
        }
    }

    fn def_syntax(&mut self) -> Result<SyntaxDef> {
        let location = self.location;
        self.advance(2)?;
        let keyword = self.current_identifier()?;
        self.expect_next_nth(1, TokenData::LeftParen)?;
        self.expect_next_nth(1, TokenData::Identifier("syntax-rules".to_string()))?;
        let (ellipsis, literals) = match self.advance(1)? {
            Some(Token {
                data: TokenData::LeftParen,
                ..
            }) => (None, {
                self.repeat(Self::current_identifier)
                    .collect::<Result<_>>()?
            }),
            Some(Token {
                data: TokenData::Identifier(ellipsis),
                ..
            }) => (Some(ellipsis.clone()), {
                self.expect_next_nth(1, TokenData::LeftParen)?;
                self.repeat(Self::current_identifier)
                    .collect::<Result<_>>()?
            }),
            Some(other) => {
                return located_error!(
                    SyntaxError::UnexpectedToken(other.data.clone()),
                    other.location
                )
            }
            None => return located_error!(SyntaxError::UnexpectedEnd, self.location),
        };
        let rules = self.repeat(Self::syntax_rule).collect::<Result<_>>()?;
        let syntax = SyntaxDef {
            data: SyntaxDefBody(
                keyword,
                Transformer {
                    ellipsis,
                    literals,
                    rules,
                },
            ),
            location,
        };
        self.expect_next_nth(1, TokenData::RightParen)?;
        Ok(syntax)
    }

    fn syntax_rule(&mut self) -> Result<(SyntaxPattern, SyntaxTemplate)> {
        self.expect_next_nth(1, TokenData::LeftParen)?;
        let pattern = self.pattern()?;
        self.advance(1)?;
        let template = self.template()?;
        self.expect_next_nth(1, TokenData::RightParen)?;
        Ok((pattern, template))
    }

    fn pattern(&mut self) -> Result<SyntaxPattern> {
        let token = self.advance_unwrap_take(0)?;
        let data = match token.data {
            TokenData::Identifier(ident) if ident == "_" => SyntaxPatternBody::Underscore,
            TokenData::Identifier(ident) if ident == "." => SyntaxPatternBody::Period,
            TokenData::Identifier(ident) if ident == "..." => SyntaxPatternBody::Ellipsis,
            TokenData::Identifier(ident) => SyntaxPatternBody::Identifier(ident),
            TokenData::Primitive(p) => SyntaxPatternBody::Primitive(p),
            TokenData::LeftParen => {
                let iter = self.repeat(Self::pattern);
                let (mut prefix, mut suffix, mut cdr) = (vec![], vec![], None);
                let (mut has_suffix, mut has_cdr) = (false, false);
                for element in iter {
                    let element = element?;
                    match element.data {
                        SyntaxPatternBody::Period => {
                            if has_cdr {
                                return located_error!(
                                    SyntaxError::UnexpectedToken(TokenData::Period),
                                    element.location
                                );
                            }
                            has_cdr = true;
                        }
                        SyntaxPatternBody::Ellipsis => {
                            if has_suffix {
                                return located_error!(
                                    SyntaxError::UnexpectedToken(TokenData::Period),
                                    element.location
                                );
                            } else {
                                has_suffix = true;
                            }
                        }
                        _ => match (has_suffix, has_cdr) {
                            (false, false) => prefix.push(element),
                            (true, false) => suffix.push(element),
                            (_, true) => {
                                if cdr.is_some() {
                                    return located_error!(
                                        SyntaxError::IllegalPattern,
                                        element.location
                                    );
                                }
                                cdr = Some(Box::new(element))
                            }
                        },
                    }
                }
                SyntaxPatternBody::List(prefix, suffix, cdr)
            }
            TokenData::VecConsIntro => {
                let iter = self.repeat(Self::pattern);
                let mut vec = [vec![], vec![]];
                let mut has_suffix = 0;
                for element in iter {
                    let element = element?;
                    match element.data {
                        SyntaxPatternBody::Period => {
                            return located_error!(
                                SyntaxError::UnexpectedToken(TokenData::Period),
                                element.location
                            )
                        }
                        SyntaxPatternBody::Ellipsis => {
                            has_suffix = 1;
                        }
                        _ => vec[has_suffix].push(element),
                    }
                }
                let [prefix, suffix] = vec;
                SyntaxPatternBody::Vector(prefix, suffix)
            }
            o => return located_error!(SyntaxError::UnexpectedToken(o), token.location),
        };
        Ok(SyntaxPattern {
            data,
            location: token.location,
        })
    }

    fn template_element(&mut self) -> Result<(SyntaxTemplate, /* with ellipsis */ bool)> {
        let current = self.template()?;
        let with_ellipsis = match (&current.data, self.peek_next_token()?) {
            (SyntaxTemplateBody::Ellipsis, _) => todo!(),
            (SyntaxTemplateBody::Period, _) => false,
            (
                _,
                Some(Token {
                    data: TokenData::Identifier(ident),
                    ..
                }),
            ) if ident == "..." => {
                self.advance(1)?;
                true
            }
            _ => false,
        };
        Ok((current, with_ellipsis))
    }

    fn template(&mut self) -> Result<SyntaxTemplate> {
        let token = self.advance_unwrap_take(0)?;

        let data = match token.data {
            TokenData::Identifier(ident) if ident == "..." => SyntaxTemplateBody::Ellipsis,
            TokenData::Identifier(ident) if ident == "." => SyntaxTemplateBody::Period,
            TokenData::Identifier(ident) => SyntaxTemplateBody::Identifier(ident),
            TokenData::Primitive(p) => SyntaxTemplateBody::Primitive(p),
            TokenData::LeftParen => {
                let iter = self.repeat(Self::template_element);
                let mut car = vec![];
                let mut cdr = None;
                let mut has_tail = false;
                for element in iter {
                    let (element, with_ellipsis) = element?;
                    match element.data {
                        SyntaxTemplateBody::Period => {
                            if has_tail {
                                return located_error!(
                                    SyntaxError::UnexpectedToken(TokenData::Period),
                                    element.location
                                );
                            } else {
                                has_tail = true;
                            }
                        }
                        _ => match (&cdr, has_tail, with_ellipsis) {
                            (_, false, _) => car.push((element, with_ellipsis)),
                            (None, true, false) => cdr = Some(Box::new(element)),
                            _ => {
                                return located_error!(
                                    SyntaxError::IllegalPattern,
                                    element.location
                                )
                            }
                        },
                    }
                }
                SyntaxTemplateBody::List(car, cdr)
            }
            TokenData::VecConsIntro => {
                let iter = self.repeat(Self::template_element);
                let mut vec = vec![];
                for element in iter {
                    let element = element?;
                    match element.0.data {
                        SyntaxTemplateBody::Period => {
                            return located_error!(
                                SyntaxError::UnexpectedToken(TokenData::Period),
                                element.0.location
                            )
                        }
                        _ => vec.push(element),
                    }
                }
                SyntaxTemplateBody::Vector(vec)
            }
            o => return located_error!(SyntaxError::UnexpectedToken(o), self.location),
        };
        Ok(SyntaxTemplate {
            data,
            location: self.location,
        })
    }

    fn assginment(&mut self) -> Result<Expression> {
        let location = self.location;
        self.advance(2)?;
        let symbol = self.current_identifier()?;
        let value = self.expression()?;
        self.expect_next_nth(1, TokenData::RightParen)?;
        Ok(ExpressionBody::Assignment(symbol, Box::new(value)).locate(location))
    }

    fn procedure_call(&mut self) -> Result<Expression> {
        let location = self.location;
        let operator = self.expression()?;
        let arguments = self
            .repeat(Self::current_expression)
            .collect::<Result<Vec<Expression>>>()?;
        Ok(ExpressionBody::ProcedureCall(Box::new(operator), arguments).locate(location))
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

    // fn advance_unwrap(&mut self, count: usize) -> Result<&mut Token> {
    //     let token = self.advance(count)?;
    //     match token {
    //         Some(tok) => Ok(tok),
    //         None => located_error!(SyntaxError::UnexpectedEnd, self.location),
    //     }
    // }

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
    assert_eq!(parser.parse(), Ok(None));
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
        location: None,
    }
}

#[test]
fn integer() -> Result<()> {
    let tokens = convert_located(vec![TokenData::Primitive(Primitive::Integer(1))]);
    let mut parser = token_stream_to_parser(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(ast, expr_body_to_statement(Primitive::Integer(1).into()));
    Ok(())
}

#[test]
fn real_number() -> Result<()> {
    let tokens = convert_located(vec![TokenData::Primitive(Primitive::Real(
        "1.2".to_string(),
    ))]);
    let mut parser = token_stream_to_parser(tokens.into_iter());
    let ast = parser.parse()?;
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
    let ast = parser.parse()?;
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
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        expr_body_to_statement(ExpressionBody::Identifier("test".to_string()))
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
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        expr_body_to_statement(ExpressionBody::Vector(convert_located(vec![
            Primitive::Integer(1).into(),
            Primitive::Boolean(false).into()
        ])))
    );
    Ok(())
}

#[test]
fn string() -> Result<()> {
    let tokens = convert_located(vec![TokenData::Primitive(Primitive::String(
        "hello world".to_string(),
    ))]);
    let mut parser = token_stream_to_parser(tokens.into_iter());
    let ast = parser.parse()?;
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
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        expr_body_to_statement(ExpressionBody::ProcedureCall(
            Box::new(ExpressionBody::Identifier("+".to_string()).into()),
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
        parser.parse(),
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
            let ast = parser.parse()?;
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
            let ast = parser.parse()?;
            assert_eq!(
                ast,
                def_body_to_statement(DefinitionBody(
                    "add".to_string(),
                    simple_procedure(
                        ParameterFormals(vec!["x".to_string(), "y".to_string()], None),
                        ExpressionBody::ProcedureCall(
                            Box::new(ExpressionBody::Identifier("+".to_string()).into()),
                            vec![
                                ExpressionBody::Identifier("x".to_string()).into(),
                                ExpressionBody::Identifier("y".to_string()).into(),
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
            let ast = parser.parse()?;
            assert_eq!(
                ast,
                def_body_to_statement(DefinitionBody(
                    "add".to_string(),
                    simple_procedure(
                        ParameterFormals(vec![], Some("x".to_string())),
                        ExpressionBody::Identifier("x".to_string()).into()
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
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        expr_body_to_statement(ExpressionBody::ProcedureCall(
            Box::new(ExpressionBody::Identifier("+".to_string()).into()),
            vec![
                Primitive::Integer(1).into(),
                ExpressionBody::ProcedureCall(
                    Box::new(ExpressionBody::Identifier("-".to_string()).into()),
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
        let ast = parser.parse()?;
        assert_eq!(
            ast,
            Some(Statement::Expression(simple_procedure(
                ParameterFormals(vec!["x".to_string(), "y".to_string()], None),
                ExpressionBody::ProcedureCall(
                    Box::new(ExpressionBody::Identifier("+".to_string()).into()),
                    vec![
                        ExpressionBody::Identifier("x".to_string()).into(),
                        ExpressionBody::Identifier("y".to_string()).into()
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
        let ast = parser.parse()?;
        assert_eq!(
            ast,
            Some(Statement::Expression(
                ExpressionBody::Procedure(SchemeProcedure(
                    ParameterFormals(vec!["x".to_string()], None),
                    vec![Definition::from(DefinitionBody(
                        "y".to_string(),
                        Primitive::Integer(1).into()
                    ))],
                    vec![ExpressionBody::ProcedureCall(
                        Box::new(ExpressionBody::Identifier("+".to_string()).into()),
                        vec![
                            ExpressionBody::Identifier("x".to_string()).into(),
                            ExpressionBody::Identifier("y".to_string()).into()
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
        let err = parser.parse();
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
        let ast = parser.parse()?;
        assert_eq!(
            ast,
            Some(Statement::Expression(
                ExpressionBody::Procedure(SchemeProcedure(
                    ParameterFormals(vec!["x".to_string()], Some("y".to_string())),
                    vec![],
                    vec![ExpressionBody::ProcedureCall(
                        Box::new(ExpressionBody::Identifier("+".to_string()).into()),
                        vec![
                            ExpressionBody::Identifier("x".to_string()).into(),
                            ExpressionBody::Identifier("y".to_string()).into()
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
        parser.parse()?,
        Some(Statement::Expression(
            ExpressionBody::Conditional(Box::new((
                Primitive::Boolean(true).into(),
                Primitive::Integer(1).into(),
                Some(Primitive::Integer(2).into())
            )))
            .into()
        ))
    );
    assert_eq!(parser.parse()?, None);
    Ok(())
}

#[test]
fn import_set() -> Result<()> {
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("foo".to_string()),
            TokenData::Primitive(Primitive::Integer(5)),
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        let import_set = parser.import_set()?;
        assert_eq!(
            import_set.data,
            ImportSetBody::Direct(library_name!("foo", 5).no_locate())
        );
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("only".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("foo".to_string()),
            TokenData::Primitive(Primitive::Integer(5)),
            TokenData::RightParen,
            TokenData::Identifier("a".to_string()),
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        let import_set = parser.import_set()?;
        assert_eq!(
            import_set.data,
            ImportSetBody::Only(
                Box::new(ImportSetBody::Direct(library_name!("foo", 5).no_locate()).no_locate()),
                vec!["a".to_string()]
            )
        );
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("except".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("foo".to_string()),
            TokenData::Primitive(Primitive::Integer(5)),
            TokenData::RightParen,
            TokenData::Identifier("a".to_string()),
            TokenData::Identifier("b".to_string()),
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        let import_set = parser.import_set()?;
        assert_eq!(
            import_set.data,
            ImportSetBody::Except(
                Box::new(ImportSetBody::Direct(library_name!("foo", 5).no_locate()).no_locate()),
                vec!["a".to_string(), "b".to_string()]
            )
        );
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("prefix".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("foo".to_string()),
            TokenData::Primitive(Primitive::Integer(5)),
            TokenData::RightParen,
            TokenData::Identifier("a-".to_string()),
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        let import_set = parser.import_set()?;
        assert_eq!(
            import_set.data,
            ImportSetBody::Prefix(
                Box::new(ImportSetBody::Direct(library_name!("foo", 5).no_locate()).no_locate()),
                "a-".to_string()
            )
        );
    }
    {
        let tokens = convert_located(vec![
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
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        let import_set = parser.import_set()?;
        assert_eq!(
            import_set.data,
            ImportSetBody::Rename(
                Box::new(ImportSetBody::Direct(library_name!("foo", 5).no_locate()).no_locate()),
                vec![
                    ("a".to_string(), "b".to_string()),
                    ("c".to_string(), "d".to_string()),
                ]
            )
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
        let ast = parser.parse()?;
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
                    ExpressionBody::Quote(Box::new(Primitive::Integer(1).into())).into()
                ),
                Statement::Expression(
                    ExpressionBody::Quote(Box::new(
                        ExpressionBody::Identifier("a".to_string()).into(),
                    ))
                    .into()
                ),
                Statement::Expression(
                    ExpressionBody::Quote(Box::new(
                        ExpressionBody::List(vec![Primitive::Integer(1).into()]).into()
                    ))
                    .into()
                ),
                Statement::Expression(
                    ExpressionBody::Vector(vec![Primitive::Integer(1).into()]).into()
                ),
                Statement::Expression(
                    ExpressionBody::Quote(Box::new(
                        ExpressionBody::Vector(vec![Primitive::Integer(1).into()]).into()
                    ))
                    .into()
                ),
            ]
        );
    }
    Ok(())
}

#[test]
fn syntax() -> Result<()> {
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
                Transformer {
                    ellipsis: None,
                    literals: Vec::new(),
                    rules: vec![(
                        SyntaxPatternBody::List(
                            vec![
                                SyntaxPatternBody::Identifier("begin".to_string()).into(),
                                SyntaxPatternBody::Identifier("exp".to_string()).into(),
                            ],
                            vec![],
                            None
                        )
                        .into(),
                        SyntaxTemplateBody::List(
                            vec![(
                                SyntaxTemplateBody::List(
                                    vec![
                                        (
                                            SyntaxTemplateBody::Identifier("lambda".to_string())
                                                .into(),
                                            false
                                        ),
                                        (SyntaxTemplateBody::List(vec![], None).into(), false),
                                        (
                                            SyntaxTemplateBody::Identifier("exp".to_string())
                                                .into(),
                                            true
                                        ),
                                    ],
                                    None
                                )
                                .into(),
                                false
                            )],
                            None
                        )
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
            TokenData::Identifier("f".to_string()),
            TokenData::Primitive(Primitive::Integer(5)),
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        let library_name = parser.library_name()?;
        assert_eq!(library_name.data, library_name!("f", 5));
    }
    {
        let tokens = convert_located(vec![TokenData::Identifier("f".to_string())]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        assert!(parser.library_name().is_err());
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("f".to_string()),
            TokenData::Primitive(Primitive::Integer(-5)),
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        assert!(parser.library_name().is_err());
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("f".to_string()),
            TokenData::Primitive(Primitive::Integer(5)),
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        assert!(parser.library_name().is_err());
    }
    {
        let tokens = convert_located(vec![
            TokenData::Identifier("f".to_string()),
            TokenData::Primitive(Primitive::Integer(5)),
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        assert!(parser.library_name().is_err());
    }
    Ok(())
}
#[test]
fn export_spec() -> Result<()> {
    {
        let tokens = convert_located(vec![TokenData::Identifier("a".to_string())]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        let export_spec = parser.export_spec()?;
        assert_eq!(export_spec.data, ExportSpec::Direct("a".to_string()));
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("rename".to_string()),
            TokenData::Identifier("a".to_string()),
            TokenData::Identifier("b".to_string()),
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        let export_spec = parser.export_spec()?;
        assert_eq!(
            export_spec.data,
            ExportSpec::Rename("a".to_string(), "b".to_string())
        );
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("c".to_string()),
            TokenData::Identifier("a".to_string()),
            TokenData::Identifier("b".to_string()),
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        assert!(parser.export_spec().is_err());
    }
    Ok(())
}
#[test]
fn library_declaration() -> Result<()> {
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("import".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("a".to_string()),
            TokenData::Identifier("b".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        let library_declaration = parser.library_declaration()?;
        assert_eq!(
            library_declaration.data,
            LibraryDeclaration::ImportDeclaration(
                ImportDeclaration(vec![ImportSetBody::Direct(
                    library_name!("a", "b").no_locate()
                )
                .no_locate()])
                .no_locate()
            )
        );
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("export".to_string()),
            TokenData::Identifier("a".to_string()),
            TokenData::Identifier("b".to_string()),
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        let library_declaration = parser.library_declaration()?;
        assert_eq!(
            library_declaration.data,
            LibraryDeclaration::Export(vec![
                ExportSpec::Direct("a".to_string()).no_locate(),
                ExportSpec::Direct("b".to_string()).no_locate()
            ])
        );
    }
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("begin".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("define".to_string()),
            TokenData::Identifier("s".to_string()),
            TokenData::Primitive(Primitive::String("a".to_string())),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = token_stream_to_parser(tokens.into_iter());
        parser.advance(1)?;
        let library_declaration = parser.library_declaration()?;
        assert_eq!(
            library_declaration.data,
            LibraryDeclaration::Begin(vec![Statement::Definition(
                DefinitionBody(
                    "s".to_string(),
                    ExpressionBody::Primitive(Primitive::String("a".to_string())).no_locate()
                )
                .no_locate()
            )])
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
        let ast = parser.parse();

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
