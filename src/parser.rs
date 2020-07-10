#![allow(dead_code)]
use crate::error::*;
use crate::lexer::Token;
use std::iter::{FromIterator, Iterator, Peekable};

type Result<T> = std::result::Result<T, Error>;
pub type ParseResult = Result<Statement>;

macro_rules! syntax_error {
    ($($arg:tt)*) => (
        return Err(Error {category: ErrorType::Syntax, message: format!($($arg)*) });
    )
}

macro_rules! expr_to_statement {
    ($expr:expr) => {
        Statement::Expression($expr)
    };
}

macro_rules! def_to_statement {
    ($definition:expr) => {
        Statement::Definition($definition)
    };
}

#[derive(PartialEq, Debug)]
pub enum Statement {
    ImportDeclaration(Vec<ImportSet>),
    Definition(Definition),
    Expression(Expression),
}

#[derive(PartialEq, Debug)]
pub struct Definition(pub String, pub Expression);

#[derive(PartialEq, Debug)]
pub enum ImportSet {
    Direct(String),
    Only(Box<ImportSet>, Vec<String>),
    Except(Box<ImportSet>, Vec<String>),
    Prefix(Box<ImportSet>, String),
    Rename(Box<ImportSet>, Vec<(String, String)>),
}

#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
    Identifier(String),
    Integer(i64),
    Real(String),
    Rational(i64, u64),
    Procedure(Vec<String>, Box<Expression>),
    ProcedureCall(Box<Expression>, Vec<Box<Expression>>),
}

pub struct Parser<TokenIter: Iterator<Item = Token>> {
    current: Option<Token>,
    lexer: Peekable<TokenIter>,
}

impl FromIterator<Token> for Result<Statement> {
    fn from_iter<I: IntoIterator<Item = Token>>(iter: I) -> Self {
        let mut p = Parser::new(iter.into_iter());
        p.parse()
    }
}

impl<TokenIter: Iterator<Item = Token>> Parser<TokenIter> {
    pub fn new(mut lexer: TokenIter) -> Parser<TokenIter> {
        Self {
            current: lexer.next(),
            lexer: lexer.peekable(),
        }
    }

    pub fn parse(&mut self) -> Result<Statement> {
        match self.current.take() {
            Some(token) => match token {
                Token::Integer(a) => self.generate(expr_to_statement!(Expression::Integer(a))),
                Token::Real(a) => self.generate(expr_to_statement!(Expression::Real(a))),
                Token::Rational(a, b) => {
                    self.generate(expr_to_statement!(Expression::Rational(a, b)))
                }
                Token::Identifier(a) => {
                    self.generate(expr_to_statement!(Expression::Identifier(a)))
                }
                Token::LeftParen => match self.lexer.peek() {
                    Some(Token::Identifier(ident)) => match ident.as_str() {
                        "lambda" => Ok(expr_to_statement!(self.lambda()?)),
                        "define" => Ok(def_to_statement!(self.definition()?)),
                        "import" => Ok(self.import_declaration()?),
                        _ => Ok(expr_to_statement!(self.procedure_call()?)),
                    },
                    Some(Token::RightParen) => syntax_error!("empty procedure call"),
                    _ => Ok(expr_to_statement!(self.procedure_call()?)),
                },
                Token::RightParen => syntax_error!("Unmatched Parentheses!"),
                _ => syntax_error!("unsupported grammar"),
            },
            None => syntax_error!("empty input or Unmatched Parentheses"),
        }
    }

    fn get_identifier(&mut self) -> Result<String> {
        match self.current.take() {
            Some(Token::Identifier(ident)) => self.generate(ident),
            _ => syntax_error!("expect an identifier"),
        }
    }

    fn get_identifier_pair(&mut self) -> Result<(String, String)> {
        match (
            self.current.take(),
            self.advance(1).take(),
            self.advance(1).take(),
            self.advance(1).take(),
        ) {
            (
                Some(Token::LeftParen),
                Some(Token::Identifier(ident1)),
                Some(Token::Identifier(ident2)),
                Some(Token::RightParen),
            ) => self.generate((ident1, ident2)),
            _ => syntax_error!("expect an identifier pair: (ident1, ident2)"),
        }
    }

    fn collect<T>(&mut self, get_element: fn(&mut Self) -> Result<T>) -> Result<Vec<T>> {
        let mut collection = vec![];
        loop {
            match self.current {
                Some(Token::RightParen) => {
                    break self.generate(collection);
                }
                Some(_) => collection.push(get_element(self)?),
                None => syntax_error!("unexpect end of input"),
            }
        }
    }

    fn lambda(&mut self) -> Result<Expression> {
        let mut formals = vec![];
        match self.advance(2).take() {
            Some(Token::Identifier(ident)) => {
                formals.push(ident);
                self.advance(1);
            }
            Some(Token::LeftParen) => {
                self.advance(1);
                formals = self.collect(Self::get_identifier)?;
            }
            _ => syntax_error!("expect identifiers"),
        }
        match (self.parse()?, self.current.take()) {
            (Statement::Expression(body), Some(Token::RightParen)) => {
                self.generate(Expression::Procedure(formals, Box::new(body)))
            }
            _ => syntax_error!("lambda body empty"),
        }
    }

    fn import_declaration(&mut self) -> Result<Statement> {
        self.advance(2);
        Ok(Statement::ImportDeclaration(
            self.collect(Self::import_set)?,
        ))
    }

    fn import_set(&mut self) -> Result<ImportSet> {
        let current = self.current.take();
        Ok(match current {
            Some(Token::Identifier(libname)) => self.generate(ImportSet::Direct(libname))?,
            Some(Token::LeftParen) => match self.advance(1).take() {
                Some(Token::Identifier(ident)) => match ident.as_str() {
                    "only" => {
                        self.advance(1);
                        ImportSet::Only(
                            Box::new(self.import_set()?),
                            self.collect(Self::get_identifier)?,
                        )
                    }
                    "except" => {
                        self.advance(1);
                        ImportSet::Except(
                            Box::new(self.import_set()?),
                            self.collect(Self::get_identifier)?,
                        )
                    }
                    "prefix" => match self.advance(1).take() {
                        Some(Token::Identifier(identifier)) => {
                            ImportSet::Prefix(Box::new(self.import_set()?), identifier)
                        }
                        _ => syntax_error!("expect a prefix name after import"),
                    },
                    "rename" => {
                        self.advance(1);
                        ImportSet::Rename(
                            Box::new(self.import_set()?),
                            self.collect(Self::get_identifier_pair)?,
                        )
                    }
                    _ => syntax_error!("import: expect sub import set"),
                },
                _ => syntax_error!("import: expect library name or sub import sets"),
            },
            _ => syntax_error!("expect a import set"),
        })
    }

    fn definition(&mut self) -> Result<Definition> {
        let current = self.advance(2).take();
        match current {
            Some(Token::Identifier(identifier)) => {
                self.advance(1);
                match (self.parse()?, self.current.take()) {
                    (Statement::Expression(expr), Some(Token::RightParen)) => {
                        self.generate(Definition(identifier, expr))
                    }
                    _ => syntax_error!("define: expect identifier and expression"),
                }
            }
            Some(Token::LeftParen) => match self.advance(1).take() {
                Some(Token::Identifier(identifier)) => {
                    let formals = self.collect(Self::get_identifier)?;
                    match self.parse()? {
                        Statement::Expression(body) => self.generate(Definition(
                            identifier,
                            Expression::Procedure(formals, Box::new(body)),
                        )),
                        _ => syntax_error!("expect procedure body"),
                    }
                }
                _ => syntax_error!("define: expect identifier and expression"),
            },
            _ => syntax_error!("define: expect identifier and expression"),
        }
    }

    fn procedure_call(&mut self) -> Result<Expression> {
        self.advance(1);
        match self.parse()? {
            Statement::Expression(operator) => {
                let mut params: Vec<Box<Expression>> = vec![];
                loop {
                    match &self.current {
                        Some(Token::RightParen) => {
                            return self
                                .generate(Expression::ProcedureCall(Box::new(operator), params));
                        }
                        None => syntax_error!("Unmatched Parentheses!"),
                        _ => params.push(match self.parse()? {
                            Statement::Expression(subexpr) => Box::new(subexpr),
                            _ => syntax_error!("Unmatched Parentheses!"),
                        }),
                    }
                }
            }
            _ => syntax_error!("operator should be an expression"),
        }
    }

    fn advance(&mut self, count: usize) -> &mut Option<Token> {
        for _ in 1..count {
            self.lexer.next();
        }
        self.current = self.lexer.next();
        &mut self.current
    }

    fn generate<T>(&mut self, ast: T) -> Result<T> {
        self.advance(1);
        Ok(ast)
    }
}

#[test]
fn empty() -> Result<()> {
    let tokens = Vec::new();
    let mut parser = Parser::new(tokens.into_iter());
    assert_eq!(
        parser.parse(),
        Err(Error {
            category: ErrorType::Syntax,
            message: "empty input or Unmatched Parentheses".to_string()
        })
    );
    Ok(())
}

#[test]
fn integer() -> Result<()> {
    let tokens = vec![Token::Integer(1)];
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(ast, expr_to_statement!(Expression::Integer(1)));
    Ok(())
}

#[test]
fn real_number() -> Result<()> {
    let tokens = vec![Token::Real("1.2".to_string())];
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(ast, expr_to_statement!(Expression::Real("1.2".to_string())));
    Ok(())
}

#[test]
fn rational() -> Result<()> {
    let tokens = vec![Token::Rational(1, 2)];
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(ast, expr_to_statement!(Expression::Rational(1, 2)));
    Ok(())
}

#[test]
fn identifier() -> Result<()> {
    let tokens = vec![Token::Identifier("test".to_string())];
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        expr_to_statement!(Expression::Identifier("test".to_string()))
    );
    Ok(())
}

#[test]
fn procedure_call() -> Result<()> {
    let tokens = vec![
        Token::LeftParen,
        Token::Identifier("+".to_string()),
        Token::Integer(1),
        Token::Integer(2),
        Token::Integer(3),
        Token::RightParen,
    ];
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        expr_to_statement!(Expression::ProcedureCall(
            Box::new(Expression::Identifier("+".to_string())),
            vec![
                Box::new(Expression::Integer(1)),
                Box::new(Expression::Integer(2)),
                Box::new(Expression::Integer(3)),
            ]
        ))
    );
    Ok(())
}

#[test]
fn unmatched_parantheses() {
    let tokens = vec![
        Token::LeftParen,
        Token::Identifier("+".to_string()),
        Token::Integer(1),
        Token::Integer(2),
        Token::Integer(3),
    ];
    let mut parser = Parser::new(tokens.into_iter());
    assert_eq!(
        parser.parse(),
        Err(Error {
            category: ErrorType::Syntax,
            message: "Unmatched Parentheses!".to_string()
        })
    );
}

#[test]
fn nested_procedure_call() -> Result<()> {
    let tokens = vec![
        Token::LeftParen,
        Token::Identifier("+".to_string()),
        Token::Integer(1),
        Token::LeftParen,
        Token::Identifier("-".to_string()),
        Token::Integer(2),
        Token::Integer(3),
        Token::RightParen,
        Token::RightParen,
    ];
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        expr_to_statement!(Expression::ProcedureCall(
            Box::new(Expression::Identifier("+".to_string())),
            vec![
                Box::new(Expression::Integer(1)),
                Box::new(Expression::ProcedureCall(
                    Box::new(Expression::Identifier("-".to_string())),
                    vec![
                        Box::new(Expression::Integer(2)),
                        Box::new(Expression::Integer(3))
                    ]
                )),
            ]
        ))
    );
    Ok(())
}

#[test]
fn lambda() -> Result<()> {
    let tokens = vec![
        Token::LeftParen,
        Token::Identifier("lambda".to_string()),
        Token::LeftParen,
        Token::Identifier("x".to_string()),
        Token::Identifier("y".to_string()),
        Token::RightParen,
        Token::LeftParen,
        Token::Identifier("+".to_string()),
        Token::Identifier("x".to_string()),
        Token::Identifier("y".to_string()),
        Token::RightParen,
        Token::RightParen,
    ];
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        Statement::Expression(Expression::Procedure(
            vec!["x".to_string(), "y".to_string()],
            Box::new(Expression::ProcedureCall(
                Box::new(Expression::Identifier("+".to_string())),
                vec![
                    Box::new(Expression::Identifier("x".to_string())),
                    Box::new(Expression::Identifier("y".to_string()))
                ]
            ))
        ),)
    );
    Ok(())
}

#[test]
fn import_declaration() -> Result<()> {
    {
        let tokens = vec![
            Token::LeftParen,
            Token::Identifier("import".to_string()),
            Token::LeftParen,
            Token::Identifier("only".to_string()),
            Token::Identifier("example-lib".to_string()),
            Token::Identifier("a".to_string()),
            Token::Identifier("b".to_string()),
            Token::RightParen,
            Token::LeftParen,
            Token::Identifier("rename".to_string()),
            Token::Identifier("example-lib".to_string()),
            Token::LeftParen,
            Token::Identifier("old".to_string()),
            Token::Identifier("new".to_string()),
            Token::RightParen,
            Token::RightParen,
            Token::RightParen,
        ];

        let mut parser = Parser::new(tokens.into_iter());
        let ast = parser.parse()?;
        assert_eq!(
            ast,
            Statement::ImportDeclaration(vec![
                ImportSet::Only(
                    Box::new(ImportSet::Direct("example-lib".to_string())),
                    vec!["a".to_string(), "b".to_string()]
                ),
                ImportSet::Rename(
                    Box::new(ImportSet::Direct("example-lib".to_string())),
                    vec![("old".to_string(), "new".to_string())]
                )
            ])
        );
    }
    Ok(())
}
