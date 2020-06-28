#![allow(dead_code)]
use crate::error::*;
use crate::lexer::Token;
use std::iter::FromIterator;
use std::iter::Iterator;

type Result<T> = std::result::Result<T, Error>;
pub type ParseResult = Result<Option<Box<Expression>>>;

macro_rules! syntax_error {
    ($($arg:tt)*) => (
        return Err(Error {category: ErrorType::Syntax, message: format!($($arg)*) });
    )
}

#[derive(PartialEq, Debug)]
pub enum Expression {
    Identifier(String),
    Interger(i64),
    Demicals(String),
    Rational(i64, u64),
    ProcudureCall(Box<Expression>, Vec<Box<Expression>>),
}

pub struct Parser<TokenIter: Iterator<Item = Token>> {
    current: Option<Token>,
    lexer: TokenIter,
}

impl FromIterator<Token> for Result<Option<Box<Expression>>> {
    fn from_iter<I: IntoIterator<Item = Token>>(iter: I) -> Self {
        let mut p = Parser::new(iter.into_iter());
        p.parse()
    }
}

impl<TokenIter: Iterator<Item = Token>> Parser<TokenIter> {
    pub fn new(mut lexer: TokenIter) -> Parser<TokenIter> {
        Self {
            current: lexer.next(),
            lexer: lexer,
        }
    }

    pub fn parse(&mut self) -> Result<Option<Box<Expression>>> {
        match self.current.clone() {
            Some(token) => match token {
                Token::Interger(a) => self.generate(Box::new(Expression::Interger(a))),
                Token::Demicals(a) => self.generate(Box::new(Expression::Demicals(a))),
                Token::Rational(a, b) => self.generate(Box::new(Expression::Rational(a, b))),
                Token::Identifier(a) => self.generate(Box::new(Expression::Identifier(a))),
                Token::LeftParen => self.procedure_call(),
                Token::RightParen => syntax_error!("Unmatched Parentheses!"),
                _ => Ok(None),
            },
            None => Ok(None),
        }
    }

    fn procedure_call(&mut self) -> Result<Option<Box<Expression>>> {
        self.advance();
        match self.parse()? {
            None => Ok(None),
            Some(operator) => {
                let mut params: Vec<Box<Expression>> = vec![];
                loop {
                    match &self.current {
                        Some(Token::RightParen) => {
                            return self
                                .generate(Box::new(Expression::ProcudureCall(operator, params)));
                        }
                        None => syntax_error!("Unmatched Parentheses!"),
                        _ => params.push(match self.parse()? {
                            None => syntax_error!("Unmatched Parentheses!"),
                            Some(subexpr) => subexpr,
                        }),
                    }
                }
            }
        }
    }

    fn advance(&mut self) {
        self.current = self.lexer.next();
    }

    fn generate(&mut self, ast: Box<Expression>) -> Result<Option<Box<Expression>>> {
        self.advance();
        Ok(Some(ast))
    }
}

#[test]
fn empty() -> Result<()> {
    let tokens = Vec::new();
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(ast, None);
    Ok(())
}

#[test]
fn interger() -> Result<()> {
    let tokens = vec![Token::Interger(1)];
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(ast, Some(Box::new(Expression::Interger(1))));
    Ok(())
}

#[test]
fn demicals() -> Result<()> {
    let tokens = vec![Token::Demicals("1.2".to_string())];
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(ast, Some(Box::new(Expression::Demicals("1.2".to_string()))));
    Ok(())
}

#[test]
fn rational() -> Result<()> {
    let tokens = vec![Token::Rational(1, 2)];
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(ast, Some(Box::new(Expression::Rational(1, 2))));
    Ok(())
}

#[test]
fn identifier() -> Result<()> {
    let tokens = vec![Token::Identifier("test".to_string())];
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        Some(Box::new(Expression::Identifier("test".to_string())))
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
        Some(Box::new(Expression::ProcudureCall(
            Box::new(Expression::Identifier("+".to_string())),
            vec![
                Box::new(Expression::Interger(1)),
                Box::new(Expression::Interger(2)),
                Box::new(Expression::Interger(3)),
            ]
        )))
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
        Some(Box::new(Expression::ProcudureCall(
            Box::new(Expression::Identifier("+".to_string())),
            vec![
                Box::new(Expression::Interger(1)),
                Box::new(Expression::ProcudureCall(
                    Box::new(Expression::Identifier("-".to_string())),
                    vec![
                        Box::new(Expression::Interger(2)),
                        Box::new(Expression::Interger(3))
                    ]
                )),
            ]
        )))
    );
    Ok(())
}
