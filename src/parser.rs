#![allow(dead_code)]
use crate::lexer::TokenData;
use crate::{error::*, lexer::Token};
use fmt::Display;
use itertools::join;
use std::fmt;
use std::iter::{FromIterator, Iterator, Peekable};

type Result<T> = std::result::Result<T, SchemeError>;
pub type ParseResult = Result<Option<(Statement, Option<[u32; 2]>)>>;

pub(crate) fn join_displayable(iter: impl IntoIterator<Item = impl fmt::Display>) -> String {
    join(iter.into_iter().map(|d| format!("{}", d)), " ")
}

#[derive(PartialEq, Debug, Clone)]
pub enum Statement {
    ImportDeclaration(Vec<ImportSet>),
    Definition(Definition),
    Expression(Expression),
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Statement::ImportDeclaration(imports) => write!(
                f,
                "(import {})",
                join(imports.iter().map(|i| format!("{}", i)), " ")
            ),
            Statement::Definition(def) => write!(f, "{}", def),
            Statement::Expression(expr) => write!(f, "{}", expr),
        }
    }
}

impl Into<Statement> for Expression {
    fn into(self) -> Statement {
        Statement::Expression(self)
    }
}

impl Into<Statement> for Definition {
    fn into(self) -> Statement {
        Statement::Definition(self)
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct DefinitionBody(pub String, pub Expression);

pub type Definition = Located<DefinitionBody>;

impl fmt::Display for DefinitionBody {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(define {} {})", self.0, self.1)
    }
}

pub type ImportSet = Located<ImportSetBody>;

#[derive(PartialEq, Debug, Clone)]
pub enum ImportSetBody {
    Direct(String),
    Only(Box<ImportSet>, Vec<String>),
    Except(Box<ImportSet>, Vec<String>),
    Prefix(Box<ImportSet>, String),
    Rename(Box<ImportSet>, Vec<(String, String)>),
}

impl fmt::Display for ImportSetBody {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Direct(s) => write!(f, "{}", s),
            Self::Only(lib, names) => write!(f, "(only {} {}", lib, names.join(" ")),
            Self::Except(lib, names) => write!(f, "(except {} {}", lib, names.join(" ")),
            Self::Prefix(lib, prefix) => write!(f, "(prefix {} {}", lib, prefix),
            Self::Rename(lib, rename) => write!(
                f,
                "({} {})",
                lib,
                join(rename.iter().map(|(a, b)| format!("{} {}", a, b)), " ")
            ),
        }
    }
}

pub type Expression = Located<ExpressionBody>;
#[derive(PartialEq, Debug, Clone)]
pub enum ExpressionBody {
    Identifier(String),
    Integer(i32),
    Boolean(bool),
    Real(String),
    Rational(i32, u32),
    Character(char),
    String(String),
    Vector(Vec<Expression>),
    Assignment(String, Box<Expression>),
    Procedure(SchemeProcedure),
    ProcedureCall(Box<Expression>, Vec<Expression>),
    Conditional(Box<(Expression, Expression, Option<Expression>)>),
    Datum(Box<Statement>),
}

// external representation, code as data
impl fmt::Display for ExpressionBody {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Identifier(s) => write!(f, "{}", s),
            Self::Integer(n) => write!(f, "{}", n),
            Self::Real(n) => write!(f, "{:?}", n),
            Self::Rational(a, b) => write!(f, "{}/{}", a, b),
            Self::Vector(vector) => write!(f, "({})", join_displayable(vector)),
            Self::Assignment(name, value) => write!(f, "(set! {} {})", name, value),
            Self::Procedure(p) => write!(f, "{}", p),
            Self::ProcedureCall(op, args) => write!(f, "({} {})", op, join_displayable(args)),
            Self::Conditional(cond) => {
                let (test, consequent, alternative) = &cond.as_ref();
                match alternative {
                    Some(alt) => write!(f, "({} {}{})", test, consequent, alt),
                    None => write!(f, "({} {})", test, consequent),
                }
            }
            Self::Character(c) => write!(f, "#\\{}", c),
            Self::String(ref s) => write!(f, "\"{}\"", s),
            Self::Datum(datum) => write!(f, "{}", datum),
            Self::Boolean(true) => write!(f, "#t"),
            Self::Boolean(false) => write!(f, "#f"),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct SchemeProcedure(pub Vec<String>, pub Vec<Definition>, pub Vec<Expression>);

impl fmt::Display for SchemeProcedure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let SchemeProcedure(formals, definitions, expressions) = self;
        write!(
            f,
            "(lambda ({}) {} {})",
            formals.join(" "),
            join_displayable(definitions),
            join_displayable(expressions)
        )
    }
}

pub struct Parser<TokenIter: Iterator<Item = Token>> {
    pub current: Option<Token>,
    pub lexer: Peekable<TokenIter>,
    location: Option<[u32; 2]>,
}

impl FromIterator<Token> for ParseResult {
    fn from_iter<I: IntoIterator<Item = Token>>(iter: I) -> Self {
        let mut p = Parser::from_token_stream(iter.into_iter());
        p.parse_root()
    }
}

impl<TokenIter: Iterator<Item = Token>> Iterator for Parser<TokenIter> {
    type Item = Result<Statement>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.parse() {
            Ok(Some(statement)) => Some(Ok(statement)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

impl<TokenIter: Iterator<Item = Token>> Parser<TokenIter> {
    pub fn from_token_stream(lexer: TokenIter) -> Parser<TokenIter> {
        Self {
            current: None,
            lexer: lexer.peekable(),
            location: None,
        }
    }

    pub fn parse_current(&mut self) -> Result<Option<Statement>> {
        match self.current.take() {
            Some(Token { data, location }) => Ok(Some(match data {
                TokenData::Boolean(b) => Expression {
                    data: ExpressionBody::Boolean(b),
                    location,
                }
                .into(),
                TokenData::Integer(a) => Expression {
                    data: ExpressionBody::Integer(a),
                    location,
                }
                .into(),
                TokenData::Real(a) => Expression {
                    data: ExpressionBody::Real(a),
                    location,
                }
                .into(),
                TokenData::Rational(a, b) => Expression {
                    data: ExpressionBody::Rational(a, b),
                    location,
                }
                .into(),
                TokenData::Identifier(a) => Expression {
                    data: ExpressionBody::Identifier(a),
                    location,
                }
                .into(),
                TokenData::LeftParen => match self.lexer.peek() {
                    Some(Token {
                        data: TokenData::Identifier(ident),
                        ..
                    }) => match ident.as_str() {
                        "lambda" => self.lambda()?.into(),
                        "define" => self.definition()?.into(),
                        "set!" => self.assginment()?.into(),
                        "import" => self.import_declaration()?.into(),
                        "if" => self.condition()?.into(),
                        _ => self.procedure_call()?.into(),
                    },
                    Some(Token {
                        data: TokenData::RightParen,
                        location,
                    }) => syntax_error!(*location, "empty procedure call"),
                    _ => self.procedure_call()?.into(),
                },
                TokenData::RightParen => syntax_error!(location, "Unmatched Parentheses!"),
                TokenData::VecConsIntro => self.vector()?.into(),
                TokenData::Character(c) => Expression {
                    data: ExpressionBody::Character(c),
                    location,
                }
                .into(),
                TokenData::String(s) => Expression {
                    data: ExpressionBody::String(s),
                    location,
                }
                .into(),
                TokenData::Quote => Expression {
                    data: ExpressionBody::Datum(Box::new(match self.parse()? {
                        Some(statement) => statement,
                        None => syntax_error!(location, "expect something to be quoted!"),
                    })),
                    location,
                }
                .into(),
                _ => syntax_error!(location, "unsupported grammar"),
            })),
            None => Ok(None),
        }
    }

    pub fn parse_current_expression(&mut self) -> Result<Expression> {
        match self.parse_current()? {
            Some(Statement::Expression(expr)) => Ok(expr),
            _ => syntax_error!(self.location, "expect a expression here"),
        }
    }

    pub fn parse_root(&mut self) -> Result<Option<(Statement, Option<[u32; 2]>)>> {
        Ok(self
            .parse()?
            .and_then(|statement| Some((statement, self.location))))
    }

    pub fn parse(&mut self) -> Result<Option<Statement>> {
        self.advance(1);
        self.parse_current()
    }

    // we know it will never be RightParen
    fn get_identifier(&mut self) -> Result<String> {
        match self.current.as_ref().map(|t| &t.data) {
            Some(TokenData::Identifier(ident)) => Ok(ident.clone()),
            _ => syntax_error!(self.location, "expect an identifier"),
        }
    }

    fn get_identifier_pair(&mut self) -> Result<(String, String)> {
        let location = self.location;
        let pairs = [
            self.current.take(),
            self.advance(1).take(),
            self.advance(1).take(),
            self.advance(1).take(),
        ];
        let datas = pairs
            .iter()
            .map(|o| o.as_ref().map(|t| &t.data))
            .collect::<Vec<_>>();
        match datas.as_slice() {
            [Some(TokenData::LeftParen), Some(TokenData::Identifier(ident1)), Some(TokenData::Identifier(ident2)), Some(TokenData::RightParen)] => {
                Ok((ident1.clone(), ident2.clone()))
            }
            other => syntax_error!(
                location,
                "expect an identifier pair: (ident1, ident2), got {:?}",
                other
            ),
        }
    }

    fn collect<T>(&mut self, get_element: fn(&mut Self) -> Result<T>) -> Result<Vec<T>>
    where
        T: std::fmt::Debug,
    {
        let mut collection = vec![];
        loop {
            match self.lexer.peek().map(|t| &t.data) {
                Some(TokenData::RightParen) => {
                    self.advance(1);
                    break Ok(collection);
                }
                None => syntax_error!(self.location, "unexpect end of input"),
                _ => {
                    self.advance(1);
                    let ele = get_element(self)?;
                    collection.push(ele);
                }
            }
        }
    }

    fn vector(&mut self) -> Result<Expression> {
        let collection = self.collect(Self::parse_current_expression)?;
        Ok(self.locate(ExpressionBody::Vector(collection)))
    }

    fn lambda(&mut self) -> Result<Expression> {
        let mut formals = vec![];
        let location = self.location;
        match self.advance(2).take().map(|t| t.data) {
            Some(TokenData::Identifier(ident)) => {
                formals.push(ident);
            }
            Some(TokenData::LeftParen) => {
                formals = self.collect(Self::get_identifier)?;
            }
            _ => syntax_error!(location, "expect formal identifiers"),
        }
        self.procedure_body(formals)
    }

    fn procedure_body(&mut self, formals: Vec<String>) -> Result<Expression> {
        let statements = self.collect(Self::parse_current)?;
        let mut definitions = vec![];
        let mut expressions = vec![];
        for statement in statements {
            match statement {
                Some(Statement::Definition(def)) => {
                    if expressions.is_empty() {
                        definitions.push(def)
                    } else {
                        syntax_error!(self.location, "unexpect definition af expression")
                    }
                }
                Some(Statement::Expression(expr)) => expressions.push(expr),
                None => syntax_error!(self.location, "lambda body empty"),
                _ => syntax_error!(
                    self.location,
                    "procedure body can only contains definition or expression"
                ),
            }
        }
        if expressions.is_empty() {
            syntax_error!(self.location, "no expression in procedure body")
        }
        Ok(self.locate(ExpressionBody::Procedure(SchemeProcedure(
            formals,
            definitions,
            expressions,
        ))))
    }

    fn import_declaration(&mut self) -> Result<Statement> {
        self.advance(1);
        Ok(Statement::ImportDeclaration(
            self.collect(Self::import_set)?,
        ))
    }

    fn condition(&mut self) -> Result<Expression> {
        self.advance(1);
        match (
            self.parse()?,
            self.parse()?,
            self.lexer.peek().map(|t| &t.data),
        ) {
            (
                Some(Statement::Expression(test)),
                Some(Statement::Expression(consequent)),
                Some(TokenData::RightParen),
            ) => {
                self.advance(1);
                Ok(Expression {
                    data: ExpressionBody::Conditional(Box::new((test, consequent, None))),
                    location: self.location,
                })
            }
            (
                Some(Statement::Expression(test)),
                Some(Statement::Expression(consequent)),
                Some(_),
            ) => match self.parse()? {
                Some(Statement::Expression(alternative)) => {
                    self.advance(1);
                    Ok(Expression {
                        data: ExpressionBody::Conditional(Box::new((
                            test,
                            consequent,
                            Some(alternative),
                        ))),
                        location: self.location,
                    })
                }
                other => syntax_error!(
                    self.location,
                    "expect condition alternatives, got {:?}",
                    other
                ),
            },
            _ => syntax_error!(self.location, "conditional syntax error"),
        }
    }

    fn import_set(&mut self) -> Result<ImportSet> {
        let import_declaration = self.location;
        Ok(match self.current.take() {
            Some(Token {
                data: TokenData::Identifier(libname),
                location,
            }) => Ok(ImportSet {
                data: ImportSetBody::Direct(libname),
                location,
            })?,
            Some(Token {
                data: TokenData::LeftParen,
                location,
            }) => match self.advance(1).take().map(|t| t.data) {
                Some(TokenData::Identifier(ident)) => match ident.as_str() {
                    "only" => {
                        self.advance(1);
                        ImportSet {
                            data: ImportSetBody::Only(
                                Box::new(self.import_set()?),
                                self.collect(Self::get_identifier)?,
                            ),
                            location,
                        }
                    }
                    "except" => {
                        self.advance(1);
                        ImportSet {
                            data: ImportSetBody::Except(
                                Box::new(self.import_set()?),
                                self.collect(Self::get_identifier)?,
                            ),
                            location,
                        }
                    }
                    "prefix" => match self.advance(2).take().map(|t| t.data) {
                        Some(TokenData::Identifier(identifier)) => ImportSet {
                            data: ImportSetBody::Prefix(Box::new(self.import_set()?), identifier),
                            location,
                        },
                        _ => syntax_error!(location, "expect a prefix name after import"),
                    },
                    "rename" => {
                        self.advance(1);
                        ImportSet {
                            data: ImportSetBody::Rename(
                                Box::new(self.import_set()?),
                                self.collect(Self::get_identifier_pair)?,
                            ),
                            location,
                        }
                    }
                    _ => syntax_error!(location, "import: expect sub import set"),
                },
                _ => syntax_error!(location, "import: expect library name or sub import sets"),
            },
            other => syntax_error!(import_declaration, "expect an import set, got {:?}", other),
        })
    }

    fn definition(&mut self) -> Result<Definition> {
        let location = self.location;
        let current = self.advance(2).take().map(|t| t.data);
        match current {
            Some(TokenData::Identifier(identifier)) => {
                match (self.parse()?, self.advance(1).take().map(|t| t.data)) {
                    (Some(Statement::Expression(expr)), Some(TokenData::RightParen)) => {
                        Ok(Definition::from_data(DefinitionBody(identifier, expr)))
                    }
                    _ => syntax_error!(location, "define: expect identifier and expression"),
                }
            }
            Some(TokenData::LeftParen) => match self.advance(1).take().map(|t| t.data) {
                Some(TokenData::Identifier(identifier)) => {
                    let formals = self.collect(Self::get_identifier)?;
                    let body = self.procedure_body(formals)?;
                    Ok(Definition::from_data(DefinitionBody(identifier, body)))
                }
                _ => syntax_error!(location, "define: expect identifier and expression"),
            },
            _ => syntax_error!(location, "define: expect identifier and expression"),
        }
    }

    fn assginment(&mut self) -> Result<Expression> {
        let location = self.location;
        let current = self.advance(2).take().map(|t| t.data);
        match current {
            Some(TokenData::Identifier(identifier)) => {
                match (self.parse()?, self.advance(1).take().map(|t| t.data)) {
                    (Some(Statement::Expression(expr)), Some(TokenData::RightParen)) => {
                        Ok(self.locate(ExpressionBody::Assignment(identifier, Box::new(expr))))
                    }
                    _ => syntax_error!(location, "define: expect identifier and expression"),
                }
            }
            Some(TokenData::LeftParen) => match self.advance(1).take().map(|t| t.data) {
                Some(TokenData::Identifier(identifier)) => {
                    let formals = self.collect(Self::get_identifier)?;
                    let body = Box::new(self.procedure_body(formals)?);
                    Ok(self.locate(ExpressionBody::Assignment(identifier, body)))
                }
                _ => syntax_error!(location, "set!: expect identifier and expression"),
            },
            _ => syntax_error!(location, "set!: expect identifier and expression"),
        }
    }

    fn procedure_call(&mut self) -> Result<Expression> {
        match self.parse()? {
            Some(Statement::Expression(operator)) => {
                let mut arguments: Vec<Expression> = vec![];
                loop {
                    match self.lexer.peek().map(|t| &t.data) {
                        Some(TokenData::RightParen) => {
                            self.advance(1);
                            return Ok(self.locate(ExpressionBody::ProcedureCall(
                                Box::new(operator),
                                arguments,
                            )));
                        }
                        None => syntax_error!(self.location, "Unmatched Parentheses!"),
                        _ => arguments.push(match self.parse()? {
                            Some(Statement::Expression(subexpr)) => subexpr,
                            _ => syntax_error!(self.location, "Unmatched Parentheses!"),
                        }),
                    }
                }
            }
            _ => syntax_error!(self.location, "operator should be an expression"),
        }
    }

    fn advance(&mut self, count: usize) -> &mut Option<Token> {
        for _ in 1..count {
            self.lexer.next();
        }
        self.current = self.lexer.next();
        self.location = self.current.as_ref().and_then(|t| t.location);
        &mut self.current
    }

    fn locate<T: PartialEq + Display>(&self, data: T) -> Located<T> {
        Located {
            data,
            location: self.location,
        }
    }
}

#[cfg(test)]
pub(crate) fn convert_located<T: PartialEq + Display>(datas: Vec<T>) -> Vec<Located<T>> {
    datas.into_iter().map(|d| Located::from_data(d)).collect()
}

#[cfg(test)]
pub fn simple_procedure(formals: Vec<String>, expression: Expression) -> Expression {
    Expression::from_data(ExpressionBody::Procedure(SchemeProcedure(
        formals,
        vec![],
        vec![expression],
    )))
}
#[test]
fn empty() -> Result<()> {
    let tokens = Vec::new();
    let mut parser = Parser::from_token_stream(tokens.into_iter());
    assert_eq!(parser.parse(), Ok(None));
    Ok(())
}

fn expr_body_to_statement(t: ExpressionBody) -> Option<Statement> {
    Some(Located::from_data(t).into())
}

fn def_body_to_statement(t: DefinitionBody) -> Option<Statement> {
    Some(Located::from_data(t).into())
}

#[test]
fn integer() -> Result<()> {
    let tokens = convert_located(vec![TokenData::Integer(1)]);
    let mut parser = Parser::from_token_stream(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(ast, expr_body_to_statement(ExpressionBody::Integer(1)));
    Ok(())
}

#[test]
fn real_number() -> Result<()> {
    let tokens = convert_located(vec![TokenData::Real("1.2".to_string())]);
    let mut parser = Parser::from_token_stream(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        expr_body_to_statement(ExpressionBody::Real("1.2".to_string()))
    );
    Ok(())
}

#[test]
fn rational() -> Result<()> {
    let tokens = convert_located(vec![TokenData::Rational(1, 2)]);
    let mut parser = Parser::from_token_stream(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(ast, expr_body_to_statement(ExpressionBody::Rational(1, 2)));
    Ok(())
}

#[test]
fn identifier() -> Result<()> {
    let tokens = convert_located(vec![TokenData::Identifier("test".to_string())]);
    let mut parser = Parser::from_token_stream(tokens.into_iter());
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
        TokenData::Integer(1),
        TokenData::Boolean(false),
        TokenData::RightParen,
    ]);
    let mut parser = Parser::from_token_stream(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        expr_body_to_statement(ExpressionBody::Vector(vec![
            Expression::from_data(ExpressionBody::Integer(1)),
            Expression::from_data(ExpressionBody::Boolean(false))
        ]))
    );
    Ok(())
}

#[test]
fn string() -> Result<()> {
    let tokens = convert_located(vec![TokenData::String("hello world".to_string())]);
    let mut parser = Parser::from_token_stream(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        expr_body_to_statement(ExpressionBody::String("hello world".to_string()))
    );
    Ok(())
}

#[test]
fn procedure_call() -> Result<()> {
    let tokens = convert_located(vec![
        TokenData::LeftParen,
        TokenData::Identifier("+".to_string()),
        TokenData::Integer(1),
        TokenData::Integer(2),
        TokenData::Integer(3),
        TokenData::RightParen,
    ]);
    let mut parser = Parser::from_token_stream(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        expr_body_to_statement(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "+".to_string()
            ))),
            vec![
                Expression::from_data(ExpressionBody::Integer(1)),
                Expression::from_data(ExpressionBody::Integer(2)),
                Expression::from_data(ExpressionBody::Integer(3)),
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
        TokenData::Integer(1),
        TokenData::Integer(2),
        TokenData::Integer(3),
    ]);
    let mut parser = Parser::from_token_stream(tokens.into_iter());
    assert_eq!(
        parser.parse(),
        Err(SchemeError {
            category: ErrorType::Syntax,
            message: "Unmatched Parentheses!".to_string(),
            location: None
        })
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
                TokenData::Integer(1),
                TokenData::RightParen,
            ]);
            let mut parser = Parser::from_token_stream(tokens.into_iter());
            let ast = parser.parse()?;
            assert_eq!(
                ast,
                def_body_to_statement(DefinitionBody(
                    "a".to_string(),
                    Expression::from_data(ExpressionBody::Integer(1))
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
            let mut parser = Parser::from_token_stream(tokens.into_iter());
            let ast = parser.parse()?;
            assert_eq!(
                ast,
                def_body_to_statement(DefinitionBody(
                    "add".to_string(),
                    simple_procedure(
                        vec!["x".to_string(), "y".to_string()],
                        Expression::from_data(ExpressionBody::ProcedureCall(
                            Box::new(Expression::from_data(ExpressionBody::Identifier(
                                "+".to_string()
                            ))),
                            vec![
                                Expression::from_data(ExpressionBody::Identifier("x".to_string())),
                                Expression::from_data(ExpressionBody::Identifier("y".to_string())),
                            ]
                        ))
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
        TokenData::Integer(1),
        TokenData::LeftParen,
        TokenData::Identifier("-".to_string()),
        TokenData::Integer(2),
        TokenData::Integer(3),
        TokenData::RightParen,
        TokenData::RightParen,
    ]);
    let mut parser = Parser::from_token_stream(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        expr_body_to_statement(ExpressionBody::ProcedureCall(
            Box::new(Expression::from_data(ExpressionBody::Identifier(
                "+".to_string()
            ))),
            vec![
                Expression::from_data(ExpressionBody::Integer(1)),
                Expression::from_data(ExpressionBody::ProcedureCall(
                    Box::new(Expression::from_data(ExpressionBody::Identifier(
                        "-".to_string()
                    ))),
                    vec![
                        Expression::from_data(ExpressionBody::Integer(2)),
                        Expression::from_data(ExpressionBody::Integer(3))
                    ]
                )),
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
        let mut parser = Parser::from_token_stream(tokens.into_iter());
        let ast = parser.parse()?;
        assert_eq!(
            ast,
            Some(Statement::Expression(simple_procedure(
                vec!["x".to_string(), "y".to_string()],
                Expression::from_data(ExpressionBody::ProcedureCall(
                    Box::new(Expression::from_data(ExpressionBody::Identifier(
                        "+".to_string()
                    ))),
                    vec![
                        Expression::from_data(ExpressionBody::Identifier("x".to_string())),
                        Expression::from_data(ExpressionBody::Identifier("y".to_string()))
                    ]
                ))
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
            TokenData::Integer(1),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("+".to_string()),
            TokenData::Identifier("x".to_string()),
            TokenData::Identifier("y".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = Parser::from_token_stream(tokens.into_iter());
        let ast = parser.parse()?;
        assert_eq!(
            ast,
            Some(Statement::Expression(Expression::from_data(
                ExpressionBody::Procedure(SchemeProcedure(
                    vec!["x".to_string()],
                    vec![Definition::from_data(DefinitionBody(
                        "y".to_string(),
                        Expression::from_data(ExpressionBody::Integer(1))
                    ))],
                    vec![Expression::from_data(ExpressionBody::ProcedureCall(
                        Box::new(Expression::from_data(ExpressionBody::Identifier(
                            "+".to_string()
                        ))),
                        vec![
                            Expression::from_data(ExpressionBody::Identifier("x".to_string())),
                            Expression::from_data(ExpressionBody::Identifier("y".to_string()))
                        ]
                    ))]
                ))
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
            TokenData::Integer(1),
            TokenData::RightParen,
            TokenData::RightParen,
        ]);
        let mut parser = Parser::from_token_stream(tokens.into_iter());
        let err = parser.parse();
        assert_eq!(
            err,
            Err(SchemeError {
                category: ErrorType::Syntax,
                message: "no expression in procedure body".to_string(),
                location: None
            })
        );
    }

    Ok(())
}

#[test]
fn conditional() -> Result<()> {
    let tokens = convert_located(vec![
        TokenData::LeftParen,
        TokenData::Identifier("if".to_string()),
        TokenData::Boolean(true),
        TokenData::Integer(1),
        TokenData::Integer(2),
        TokenData::RightParen,
    ]);
    let mut parser = Parser::from_token_stream(tokens.into_iter());
    assert_eq!(
        parser.parse()?,
        Some(Statement::Expression(Expression::from_data(
            ExpressionBody::Conditional(Box::new((
                Expression::from_data(ExpressionBody::Boolean(true)),
                Expression::from_data(ExpressionBody::Integer(1)),
                Some(Expression::from_data(ExpressionBody::Integer(2)))
            )))
        )))
    );
    assert_eq!(parser.parse()?, None);
    Ok(())
}

/* (import
(only example-lib a b)
(rename example-lib (old new))
) */
#[test]
fn import_declaration() -> Result<()> {
    {
        let tokens = convert_located(vec![
            TokenData::LeftParen,
            TokenData::Identifier("import".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("only".to_string()),
            TokenData::Identifier("example-lib".to_string()),
            TokenData::Identifier("a".to_string()),
            TokenData::Identifier("b".to_string()),
            TokenData::RightParen,
            TokenData::LeftParen,
            TokenData::Identifier("rename".to_string()),
            TokenData::Identifier("example-lib".to_string()),
            TokenData::LeftParen,
            TokenData::Identifier("old".to_string()),
            TokenData::Identifier("new".to_string()),
            TokenData::RightParen,
            TokenData::RightParen,
            TokenData::RightParen,
        ]);

        let mut parser = Parser::from_token_stream(tokens.into_iter());
        let ast = parser.parse()?;
        assert_eq!(
            ast,
            Some(Statement::ImportDeclaration(convert_located(vec![
                ImportSetBody::Only(
                    Box::new(ImportSet::from_data(ImportSetBody::Direct(
                        "example-lib".to_string()
                    ))),
                    vec!["a".to_string(), "b".to_string()]
                ),
                ImportSetBody::Rename(
                    Box::new(ImportSet::from_data(ImportSetBody::Direct(
                        "example-lib".to_string()
                    ))),
                    vec![("old".to_string(), "new".to_string())]
                )
            ])))
        );
    }
    Ok(())
}
