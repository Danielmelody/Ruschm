#![allow(dead_code)]
use crate::error::*;
use crate::lexer::Token;
use std::iter::{FromIterator, Iterator, Peekable};

type Result<T> = std::result::Result<T, Error>;
pub type ParseResult = Result<Option<Statement>>;

macro_rules! syntax_error {
    ($($arg:tt)*) => (
        return Err(Error {category: ErrorType::Syntax, message: format!($($arg)*) });
    )
}

macro_rules! expr_to_statement {
    ($expr:expr) => {
        Some(Statement::Expression($expr))
    };
}

macro_rules! def_to_statement {
    ($definition:expr) => {
        Some(Statement::Definition($definition))
    };
}

#[derive(PartialEq, Debug, Clone)]
pub enum Statement {
    ImportDeclaration(Vec<ImportSet>),
    Definition(Definition),
    Expression(Expression),
}

#[derive(PartialEq, Debug, Clone)]
pub struct Definition(pub String, pub Expression);

#[derive(PartialEq, Debug, Clone)]
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
    Boolean(bool),
    Real(String),
    Rational(i64, u64),
    Vector(Vec<Expression>),
    Procedure(Vec<String>, Vec<Definition>, Vec<Expression>),
    ProcedureCall(Box<Expression>, Vec<Expression>),
    Conditional(Box<(Expression, Expression, Option<Expression>)>),
}

pub struct Parser<TokenIter: Iterator<Item = Token>> {
    pub current: Option<Token>,
    pub lexer: Peekable<TokenIter>,
}

impl FromIterator<Token> for ParseResult {
    fn from_iter<I: IntoIterator<Item = Token>>(iter: I) -> Self {
        let mut p = Parser::new(iter.into_iter());
        p.parse()
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
    pub fn new(lexer: TokenIter) -> Parser<TokenIter> {
        Self {
            current: None,
            lexer: lexer.peekable(),
        }
    }

    pub fn parse_current(&mut self) -> Result<Option<Statement>> {
        match self.current.take() {
            Some(token) => match token {
                Token::Boolean(b) => Ok(Some(Statement::Expression(Expression::Boolean(b)))),
                Token::Integer(a) => Ok(expr_to_statement!(Expression::Integer(a))),
                Token::Real(a) => Ok(expr_to_statement!(Expression::Real(a))),
                Token::Rational(a, b) => Ok(expr_to_statement!(Expression::Rational(a, b))),
                Token::Identifier(a) => Ok(expr_to_statement!(Expression::Identifier(a))),
                Token::LeftParen => match self.lexer.peek() {
                    Some(Token::Identifier(ident)) => match ident.as_str() {
                        "lambda" => Ok(expr_to_statement!(self.lambda()?)),
                        "define" => Ok(def_to_statement!(self.definition()?)),
                        "import" => Ok(Some(self.import_declaration()?)),
                        "if" => Ok(expr_to_statement!(self.condition()?)),
                        _ => Ok(expr_to_statement!(self.procedure_call()?)),
                    },
                    Some(Token::RightParen) => syntax_error!("empty procedure call"),
                    _ => Ok(expr_to_statement!(self.procedure_call()?)),
                },
                Token::RightParen => syntax_error!("Unmatched Parentheses!"),
                Token::VecConsIntro => Ok(expr_to_statement!(self.vector()?)),
                _ => syntax_error!("unsupported grammar"),
            },
            None => Ok(None),
        }
    }

    pub fn parse_current_expression(&mut self) -> Result<Expression> {
        match self.parse_current()? {
            Some(Statement::Expression(expr)) => Ok(expr),
            _ => syntax_error!("expect a expression here"),
        }
    }

    pub fn parse(&mut self) -> Result<Option<Statement>> {
        self.advance(1);
        self.parse_current()
    }

    // we know it will never be RightParen
    fn get_identifier(&mut self) -> Result<String> {
        match self.current.take() {
            Some(Token::Identifier(ident)) => Ok(ident.clone()),
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
            ) => Ok((ident1, ident2)),
            other => syntax_error!(
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
            match self.lexer.peek() {
                Some(Token::RightParen) => {
                    self.advance(1);
                    break Ok(collection);
                }
                None => syntax_error!("unexpect end of input"),
                _ => {
                    self.advance(1);
                    let ele = get_element(self)?;
                    collection.push(ele);
                }
            }
        }
    }

    fn vector(&mut self) -> Result<Expression> {
        Ok(Expression::Vector(
            self.collect(Self::parse_current_expression)?,
        ))
    }

    fn lambda(&mut self) -> Result<Expression> {
        let mut formals = vec![];
        match self.advance(2).take() {
            Some(Token::Identifier(ident)) => {
                formals.push(ident);
            }
            Some(Token::LeftParen) => {
                formals = self.collect(Self::get_identifier)?;
            }
            _ => syntax_error!("expect identifiers"),
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
                        syntax_error!("unexpect definition af expression")
                    }
                }
                Some(Statement::Expression(expr)) => expressions.push(expr),
                None => syntax_error!("lambda body empty"),
                _ => syntax_error!("procedure body can only contains definition or expression"),
            }
        }
        if expressions.is_empty() {
            syntax_error!("no expression in procedure body")
        }
        Ok(Expression::Procedure(formals, definitions, expressions))
    }

    fn import_declaration(&mut self) -> Result<Statement> {
        self.advance(1);
        Ok(Statement::ImportDeclaration(
            self.collect(Self::import_set)?,
        ))
    }

    fn condition(&mut self) -> Result<Expression> {
        self.advance(1);
        match (self.parse()?, self.parse()?, self.lexer.peek()) {
            (
                Some(Statement::Expression(test)),
                Some(Statement::Expression(consequent)),
                Some(Token::RightParen),
            ) => {
                self.advance(1);
                Ok(Expression::Conditional(Box::new((test, consequent, None))))
            }
            (
                Some(Statement::Expression(test)),
                Some(Statement::Expression(consequent)),
                Some(_),
            ) => match self.parse()? {
                Some(Statement::Expression(alternative)) => {
                    self.advance(1);
                    Ok(Expression::Conditional(Box::new((
                        test,
                        consequent,
                        Some(alternative),
                    ))))
                }
                other => syntax_error!("expect condition alternatives, got {:?}", other),
            },
            _ => syntax_error!("conditional syntax error"),
        }
    }

    fn import_set(&mut self) -> Result<ImportSet> {
        Ok(match self.current.take() {
            Some(Token::Identifier(libname)) => Ok(ImportSet::Direct(libname))?,
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
                    "prefix" => match self.advance(2).take() {
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
            other => syntax_error!("expect an import set, got {:?}", other),
        })
    }

    fn definition(&mut self) -> Result<Definition> {
        let current = self.advance(2).take();
        match current {
            Some(Token::Identifier(identifier)) => match (self.parse()?, self.advance(1)) {
                (Some(Statement::Expression(expr)), Some(Token::RightParen)) => {
                    Ok(Definition(identifier, expr))
                }
                _ => syntax_error!("define: expect identifier and expression"),
            },
            Some(Token::LeftParen) => match self.advance(1).take() {
                Some(Token::Identifier(identifier)) => {
                    let formals = self.collect(Self::get_identifier)?;
                    let body = self.procedure_body(formals)?;
                    Ok(Definition(identifier, body))
                }
                _ => syntax_error!("define: expect identifier and expression"),
            },
            _ => syntax_error!("define: expect identifier and expression"),
        }
    }

    fn procedure_call(&mut self) -> Result<Expression> {
        match self.parse()? {
            Some(Statement::Expression(operator)) => {
                let mut arguments: Vec<Expression> = vec![];
                loop {
                    match self.lexer.peek() {
                        Some(Token::RightParen) => {
                            self.advance(1);
                            return Ok(Expression::ProcedureCall(Box::new(operator), arguments));
                        }
                        None => syntax_error!("Unmatched Parentheses!"),
                        _ => arguments.push(match self.parse()? {
                            Some(Statement::Expression(subexpr)) => subexpr,
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
}

pub fn simple_procedure(formals: Vec<String>, expression: Expression) -> Expression {
    Expression::Procedure(formals, vec![], vec![expression])
}
#[test]
fn empty() -> Result<()> {
    let tokens = Vec::new();
    let mut parser = Parser::new(tokens.into_iter());
    assert_eq!(parser.parse(), Ok(None));
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
fn vector() -> Result<()> {
    let tokens = vec![
        Token::VecConsIntro,
        Token::Integer(1),
        Token::Boolean(false),
    ];
    let mut parser = Parser::new(tokens.into_iter());
    let ast = parser.parse()?;
    assert_eq!(
        ast,
        expr_to_statement!(Expression::Vector(vec![
            Expression::Integer(1),
            Expression::Boolean(false)
        ]))
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
                Expression::Integer(1),
                Expression::Integer(2),
                Expression::Integer(3),
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
fn definition() -> Result<()> {
    {
        {
            let tokens = vec![
                Token::LeftParen,
                Token::Identifier("define".to_string()),
                Token::Identifier("a".to_string()),
                Token::Integer(1),
                Token::RightParen,
            ];
            let mut parser = Parser::new(tokens.into_iter());
            let ast = parser.parse()?;
            assert_eq!(
                ast,
                def_to_statement!(Definition("a".to_string(), Expression::Integer(1)))
            );
        }
        {
            let tokens = vec![
                Token::LeftParen,
                Token::Identifier("define".to_string()),
                Token::LeftParen,
                Token::Identifier("add".to_string()),
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
                def_to_statement!(Definition(
                    "add".to_string(),
                    simple_procedure(
                        vec!["x".to_string(), "y".to_string()],
                        Expression::ProcedureCall(
                            Box::new(Expression::Identifier("+".to_string())),
                            vec![
                                Expression::Identifier("x".to_string()),
                                Expression::Identifier("y".to_string()),
                            ]
                        )
                    )
                ))
            )
        }
        Ok(())
    }
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
                Expression::Integer(1),
                Expression::ProcedureCall(
                    Box::new(Expression::Identifier("-".to_string())),
                    vec![Expression::Integer(2), Expression::Integer(3)]
                ),
            ]
        ))
    );
    Ok(())
}

#[test]
fn lambda() -> Result<()> {
    {
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
            Some(Statement::Expression(simple_procedure(
                vec!["x".to_string(), "y".to_string()],
                Expression::ProcedureCall(
                    Box::new(Expression::Identifier("+".to_string())),
                    vec![
                        Expression::Identifier("x".to_string()),
                        Expression::Identifier("y".to_string())
                    ]
                )
            )))
        );
    }

    {
        let tokens = vec![
            Token::LeftParen,
            Token::Identifier("lambda".to_string()),
            Token::LeftParen,
            Token::Identifier("x".to_string()),
            Token::RightParen,
            Token::LeftParen,
            Token::Identifier("define".to_string()),
            Token::Identifier("y".to_string()),
            Token::Integer(1),
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
            Some(Statement::Expression(Expression::Procedure(
                vec!["x".to_string()],
                vec![Definition("y".to_string(), Expression::Integer(1))],
                vec![Expression::ProcedureCall(
                    Box::new(Expression::Identifier("+".to_string())),
                    vec![
                        Expression::Identifier("x".to_string()),
                        Expression::Identifier("y".to_string())
                    ]
                )]
            )))
        );
    }

    {
        let tokens = vec![
            Token::LeftParen,
            Token::Identifier("lambda".to_string()),
            Token::LeftParen,
            Token::Identifier("x".to_string()),
            Token::RightParen,
            Token::LeftParen,
            Token::Identifier("define".to_string()),
            Token::Identifier("y".to_string()),
            Token::Integer(1),
            Token::RightParen,
            Token::RightParen,
        ];
        let mut parser = Parser::new(tokens.into_iter());
        let err = parser.parse();
        assert_eq!(
            err,
            Err(Error {
                category: ErrorType::Syntax,
                message: "no expression in procedure body".to_string()
            })
        );
    }

    Ok(())
}

#[test]
fn conditional() -> Result<()> {
    let tokens = vec![
        Token::LeftParen,
        Token::Identifier("if".to_string()),
        Token::Boolean(true),
        Token::Integer(1),
        Token::Integer(2),
        Token::RightParen,
    ];
    let mut parser = Parser::new(tokens.into_iter());
    assert_eq!(
        parser.parse()?,
        Some(Statement::Expression(Expression::Conditional(Box::new((
            Expression::Boolean(true),
            Expression::Integer(1),
            Some(Expression::Integer(2))
        )))))
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
            Some(Statement::ImportDeclaration(vec![
                ImportSet::Only(
                    Box::new(ImportSet::Direct("example-lib".to_string())),
                    vec!["a".to_string(), "b".to_string()]
                ),
                ImportSet::Rename(
                    Box::new(ImportSet::Direct("example-lib".to_string())),
                    vec![("old".to_string(), "new".to_string())]
                )
            ]))
        );
    }
    Ok(())
}
