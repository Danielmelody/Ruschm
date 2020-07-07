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
    Definition(Definition),
    Expression(Expression),
}

#[derive(PartialEq, Debug)]
pub struct Definition(pub String, pub Expression);

#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
    Identifier(String),
    Interger(i64),
    Demicals(String),
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
                Token::Interger(a) => self.generate(expr_to_statement!(Expression::Interger(a))),
                Token::Demicals(a) => self.generate(expr_to_statement!(Expression::Demicals(a))),
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

    fn collect_formals(&mut self) -> Result<Vec<String>> {
        let mut formals = vec![];
        loop {
            self.advance();
            match &self.current {
                Some(Token::Identifier(ident)) => formals.push(ident.clone()),
                Some(Token::RightParen) => {
                    self.advance();
                    break Ok(formals);
                }
                _ => syntax_error!("lambda formals must be idenfiers"),
            }
        }
    }

    fn lambda(&mut self) -> Result<Expression> {
        self.advance();
        let mut formals = vec![];
        match self.advance().take() {
            Some(Token::Identifier(ident)) => {
                formals.push(ident);
                self.advance();
            }
            Some(Token::LeftParen) => formals = self.collect_formals()?,
            _ => syntax_error!("expect identifiers"),
        }
        match (self.parse()?, self.current.take()) {
            (Statement::Expression(body), Some(Token::RightParen)) => {
                self.generate(Expression::Procedure(formals, Box::new(body)))
            }
            _ => syntax_error!("lambda body empty"),
        }
    }

    fn definition(&mut self) -> Result<Definition> {
        self.advance();
        let current = self.advance().take();
        match current {
            Some(Token::Identifier(identifier)) => {
                self.advance();
                match (self.parse()?, self.current.take()) {
                    (Statement::Expression(expr), Some(Token::RightParen)) => {
                        self.generate(Definition(identifier, expr))
                    }
                    _ => syntax_error!("define: expect identifier and expression"),
                }
            }
            Some(Token::LeftParen) => match self.advance().take() {
                Some(Token::Identifier(identifier)) => {
                    let formals = self.collect_formals()?;
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
        self.advance();
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

    fn advance(&mut self) -> &mut Option<Token> {
        self.current = self.lexer.next();
        &mut self.current
    }

    fn generate<T>(&mut self, ast: T) -> Result<T> {
        self.advance();
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
fn interger() -> Result<()> {
    let tokens = vec![Token::Interger(1)];
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(ast, expr_to_statement!(Expression::Interger(1)));
    Ok(())
}

#[test]
fn demicals() -> Result<()> {
    let tokens = vec![Token::Demicals("1.2".to_string())];
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        expr_to_statement!(Expression::Demicals("1.2".to_string()))
    );
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
        Token::Interger(1),
        Token::Interger(2),
        Token::Interger(3),
        Token::RightParen,
    ];
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        expr_to_statement!(Expression::ProcedureCall(
            Box::new(Expression::Identifier("+".to_string())),
            vec![
                Box::new(Expression::Interger(1)),
                Box::new(Expression::Interger(2)),
                Box::new(Expression::Interger(3)),
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
        Token::Interger(1),
        Token::Interger(2),
        Token::Interger(3),
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
        Token::Interger(1),
        Token::LeftParen,
        Token::Identifier("-".to_string()),
        Token::Interger(2),
        Token::Interger(3),
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
                Box::new(Expression::Interger(1)),
                Box::new(Expression::ProcedureCall(
                    Box::new(Expression::Identifier("-".to_string())),
                    vec![
                        Box::new(Expression::Interger(2)),
                        Box::new(Expression::Interger(3))
                    ]
                )),
            ]
        ))
    );
    Ok(())
}
