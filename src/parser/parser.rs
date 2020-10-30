#![allow(dead_code)]
use crate::parser::lexer::TokenData;
use crate::{error::*, parser::lexer::Token};
use fmt::Display;
use itertools::join;
use std::fmt;
use std::iter::{repeat, Iterator, Peekable};

use super::{
    Primitive, SyntaxPattern, SyntaxPatternBody, SyntaxTemplate, SyntaxTemplateBody, Transformer,
};

type Result<T> = std::result::Result<T, SchemeError>;
pub type ParseResult = Result<Option<(Statement, Option<[u32; 2]>)>>;

pub(crate) fn join_displayable(iter: impl IntoIterator<Item = impl fmt::Display>) -> String {
    join(iter.into_iter().map(|d| format!("{}", d)), " ")
}

#[derive(PartialEq, Debug, Clone)]
pub enum Statement {
    ImportDeclaration(Vec<ImportSet>),
    Definition(Definition),
    SyntaxDefinition(SyntaxDef),
    Expression(Expression),
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

#[derive(PartialEq, Debug, Clone)]
pub struct DefinitionBody(pub String, pub Expression);

#[derive(PartialEq, Debug, Clone)]
pub struct SyntaxDefBody(pub String, pub Transformer);

pub type Definition = Located<DefinitionBody>;
pub type SyntaxDef = Located<SyntaxDefBody>;
pub type ImportSet = Located<ImportSetBody>;

#[derive(PartialEq, Debug, Clone)]
pub enum ImportSetBody {
    Direct(String),
    Only(Box<ImportSet>, Vec<String>),
    Except(Box<ImportSet>, Vec<String>),
    Prefix(Box<ImportSet>, String),
    Rename(Box<ImportSet>, Vec<(String, String)>),
}

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
                            match self.advance(1)?.take().map(|t| t.data) {
                                Some(TokenData::RightParen) => (),
                                Some(o) => syntax_error!(self.location, "expect , got {}", o),
                                None => syntax_error!(self.location, "unclosed quotation!"),
                            }
                            quoted.into()
                        }
                        "define" => self.definition()?.into(),
                        "define-syntax" => self.def_syntax()?.into(),
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
                TokenData::Quote => {
                    self.advance(1)?;
                    self.quote()?.into()
                }
                TokenData::Period => Expression {
                    data: ExpressionBody::Period,
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
        self.advance(1)?;
        self.parse_current()
    }

    fn expression(&mut self) -> Result<Expression> {
        Ok(match self.parse()? {
            Some(Statement::Expression(expr)) => expr,
            _ => syntax_error!(self.location, "expect expression"),
        })
    }

    // we know it will never be RightParen
    fn get_identifier(&mut self) -> Result<String> {
        match self.current.as_ref().map(|t| &t.data) {
            Some(TokenData::Identifier(ident)) => Ok(ident.clone()),
            Some(o) => syntax_error!(self.location, "expect an identifier, got {}", o),
            None => syntax_error!(
                self.location,
                "expect an identifier while encountered end of input"
            ),
        }
    }

    fn get_identifier_pair(&mut self) -> Result<(String, String)> {
        let location = self.location;
        let pairs = [
            self.current.take(),
            self.advance(1)?.take(),
            self.advance(1)?.take(),
            self.advance(1)?.take(),
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

    fn expect_next_nth(&mut self, n: usize, tobe: TokenData) -> Result<()> {
        let test = self.advance(n)?;
        match test {
            Some(Token { data, .. }) if data == &tobe => Ok(()),
            Some(other) => syntax_error!(other.location, "expect {}, got {}", tobe, other),
            None => syntax_error!(self.location, "expect {}", tobe),
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
                None => syntax_error!(self.location, "unexpect end of input"),
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
                    if formals.0.len() == 0 {
                        syntax_error!(
                            self.location,
                            "must provide at least normal parameter before variadic parameter"
                        )
                    }
                    self.advance(2)?;
                    formals.1 = Some(self.get_identifier()?);
                }
                None => syntax_error!(self.location, "unexpect end of input"),
                _ => {
                    self.advance(1)?;
                    let parameter = self.get_identifier()?;
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
            None => syntax_error!(self.location, "expect a literal"),
            _ => self.parse_current_expression()?,
        })
    }

    fn lambda(&mut self) -> Result<Expression> {
        let location = self.location;
        let mut formals = ParameterFormals::new();
        match self.advance(2)?.take().map(|t| t.data) {
            Some(TokenData::Identifier(ident)) => formals.1 = Some(ident),
            Some(TokenData::LeftParen) => {
                formals = self.procedure_formals()?;
            }
            _ => syntax_error!(location, "expect formal identifiers"),
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
                        syntax_error!(def.location, "unexpect definition af expression")
                    }
                }
                Some(Statement::Expression(expr)) => expressions.push(expr),
                None => syntax_error!(body_location, "lambda body empty"),
                _ => syntax_error!(
                    body_location,
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
        self.expect_next_nth(1, TokenData::Identifier("import".to_string()))?;
        Ok(Statement::ImportDeclaration(
            self.repeat(Self::import_set).collect::<Result<_>>()?,
        ))
    }

    fn condition(&mut self) -> Result<Expression> {
        self.advance(1)?;
        match (
            self.parse()?,
            self.parse()?,
            self.peek_next_token()?.map(|t| &t.data),
        ) {
            (
                Some(Statement::Expression(test)),
                Some(Statement::Expression(consequent)),
                Some(TokenData::RightParen),
            ) => {
                self.advance(1)?;
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
                    self.advance(1)?;
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
            }) => match self.advance(1)?.take().map(|t| t.data) {
                Some(TokenData::Identifier(ident)) => match ident.as_str() {
                    "only" => {
                        self.advance(1)?;
                        ImportSet {
                            data: ImportSetBody::Only(
                                Box::new(self.import_set()?),
                                self.repeat(Self::get_identifier).collect::<Result<_>>()?,
                            ),
                            location,
                        }
                    }
                    "except" => {
                        self.advance(1)?;
                        ImportSet {
                            data: ImportSetBody::Except(
                                Box::new(self.import_set()?),
                                self.repeat(Self::get_identifier).collect::<Result<_>>()?,
                            ),
                            location,
                        }
                    }
                    "prefix" => match self.advance(2)?.take().map(|t| t.data) {
                        Some(TokenData::Identifier(identifier)) => ImportSet {
                            data: ImportSetBody::Prefix(Box::new(self.import_set()?), identifier),
                            location,
                        },
                        _ => syntax_error!(location, "expect a prefix name after import"),
                    },
                    "rename" => {
                        self.advance(1)?;
                        ImportSet {
                            data: ImportSetBody::Rename(
                                Box::new(self.import_set()?),
                                self.repeat(Self::get_identifier_pair)
                                    .collect::<Result<_>>()?,
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
        let current = self.advance(2)?.take().map(|t| t.data);
        match current {
            Some(TokenData::Identifier(identifier)) => {
                match (self.parse()?, self.advance(1)?.take().map(|t| t.data)) {
                    (Some(Statement::Expression(expr)), Some(TokenData::RightParen)) => {
                        Ok(Definition::from_data(DefinitionBody(identifier, expr)))
                    }
                    _ => syntax_error!(location, "define: expect identifier and expression"),
                }
            }
            Some(TokenData::LeftParen) => match self.advance(1)?.take().map(|t| t.data) {
                Some(TokenData::Identifier(identifier)) => {
                    let mut formals = ParameterFormals::new();
                    match self.peek_next_token()?.map(|t| &t.data) {
                        Some(TokenData::Period) => {
                            self.advance(2)?;
                            formals.1 = Some(self.get_identifier()?);
                            self.advance(1)?;
                        }
                        _ => formals = self.procedure_formals()?,
                    }
                    let body = self.procedure_body(formals)?;
                    Ok(Definition::from_data(DefinitionBody(identifier, body)))
                }
                _ => syntax_error!(location, "define: expect identifier and expression"),
            },
            _ => syntax_error!(location, "define: expect identifier and expression"),
        }
    }

    fn def_syntax(&mut self) -> Result<SyntaxDef> {
        let location = self.location;
        self.advance(2)?;
        let keyword = self.get_identifier()?;
        self.expect_next_nth(1, TokenData::LeftParen)?;
        self.expect_next_nth(1, TokenData::Identifier("syntax-rules".to_string()))?;
        let (ellipsis, literals) = match self.advance(1)? {
            Some(Token {
                data: TokenData::LeftParen,
                ..
            }) => (None, {
                self.repeat(Self::get_identifier).collect::<Result<_>>()?
            }),
            Some(Token {
                data: TokenData::Identifier(ellipsis),
                ..
            }) => (Some(ellipsis.clone()), {
                self.expect_next_nth(1, TokenData::LeftParen)?;
                self.repeat(Self::get_identifier).collect::<Result<_>>()?
            }),
            other => syntax_error!(location, "expect literal (identifier), get {:?}", other),
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
        let pattern = match self.current.take() {
            Some(token) => {
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
                                        syntax_error!(element.location, "unexpected period");
                                    } else {
                                        has_cdr = true;
                                    }
                                }
                                SyntaxPatternBody::Ellipsis => {
                                    if has_suffix {
                                        syntax_error!(element.location, "unexpected ellipsis");
                                    } else {
                                        has_suffix = true;
                                    }
                                }
                                _ => match (has_suffix, has_cdr) {
                                    (false, false) => prefix.push(element),
                                    (true, false) => suffix.push(element),
                                    (_, true) => {
                                        if cdr.is_some() {
                                            syntax_error!(element.location, "illegal pair pattern")
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
                                    syntax_error!(element.location, "unexpected period");
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
                    o => syntax_error!(self.location, "unrecognized pattern {}", o),
                };
                SyntaxPattern {
                    data,
                    location: token.location,
                }
            }
            _ => syntax_error!(self.location, "unexpected end of input"),
        };
        Ok(pattern)
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
        let tem = match self.current.take() {
            Some(token) => {
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
                                        syntax_error!(element.location, "unexpected period");
                                    } else {
                                        has_tail = true;
                                    }
                                }
                                _ => match (&cdr, has_tail, with_ellipsis) {
                                    (_, false, _) => car.push((element, with_ellipsis)),
                                    (None, true, false) => cdr = Some(Box::new(element)),
                                    _ => syntax_error!(element.location, "illegal pair pattern"),
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
                                    syntax_error!(element.0.location, "unexpected period")
                                }
                                _ => vec.push(element),
                            }
                        }
                        SyntaxTemplateBody::Vector(vec)
                    }
                    o => syntax_error!(self.location, "unrecognized template {}", o),
                };
                SyntaxTemplate {
                    data,
                    location: self.location,
                }
            }
            None => syntax_error!(self.location, "unexpect end of input"),
        };
        Ok(tem)
    }

    fn assginment(&mut self) -> Result<Expression> {
        let location = self.location;
        let current = self.advance(2)?.take().map(|t| t.data);
        match current {
            Some(TokenData::Identifier(identifier)) => {
                match (self.parse()?, self.advance(1)?.take().map(|t| t.data)) {
                    (Some(Statement::Expression(expr)), Some(TokenData::RightParen)) => {
                        Ok(self.locate(ExpressionBody::Assignment(identifier, Box::new(expr))))
                    }
                    _ => syntax_error!(location, "define: expect identifier and expression"),
                }
            }
            Some(TokenData::LeftParen) => match self.advance(1)?.take().map(|t| t.data) {
                Some(TokenData::Identifier(identifier)) => {
                    let formals = self.procedure_formals()?;
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
                    match self.peek_next_token()?.map(|t| &t.data) {
                        Some(TokenData::RightParen) => {
                            self.advance(1)?;
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

    fn peek_next_token(&mut self) -> Result<Option<&Token>> {
        match self.lexer.peek() {
            Some(ret) => match ret {
                Ok(t) => Ok(Some(t)),
                Err(e) => Err(e.clone()),
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
    Some(Located::from_data(t).into())
}

fn def_body_to_statement(t: DefinitionBody) -> Option<Statement> {
    Some(Located::from_data(t).into())
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
                    vec![Definition::from_data(DefinitionBody(
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
            Err(SchemeError {
                category: ErrorType::Syntax,
                message: "no expression in procedure body".to_string(),
                location: None
            })
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

        let mut parser = token_stream_to_parser(tokens.into_iter());
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
