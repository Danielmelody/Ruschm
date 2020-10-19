#![allow(dead_code)]
use crate::error::*;
use std::fmt;
use std::iter::Iterator;
use std::iter::Peekable;

type Result<T> = std::result::Result<T, SchemeError>;

pub type Token = Located<TokenData>;

#[derive(PartialEq, Debug, Clone)]
pub enum TokenData {
    Identifier(String),
    Boolean(bool),
    Real(String), // delay the conversion of real literal to internal represent for different virtual machines (for example, fixed-points).
    Integer(i32),
    Rational(i32, u32),
    Character(char),
    String(String),
    LeftParen,
    RightParen,
    VecConsIntro,     // #(...)
    ByteVecConsIntro, // #u8(...)
    Quote,            // '
    Quasiquote,       // BackQuote
    Unquote,          // ,
    UnquoteSplicing,  // ,@
    Period,
}

impl fmt::Display for TokenData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct Lexer<CharIter: Iterator<Item = char>> {
    pub current: Option<char>,
    pub peekable_char_stream: Peekable<CharIter>,
    location: [u32; 2],
}

impl<CharIter: Iterator<Item = char>> Iterator for Lexer<CharIter> {
    type Item = Result<Token>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.try_next() {
            Ok(None) => None,
            Ok(Some(data)) => Some(Ok(Token::from_data(data).locate(Some(self.location)))),
            Err(e) => Some(Err(e)),
        }
    }
}

fn is_identifier_initial(c: char) -> bool {
    match c {
        'a'..='z'
        | 'A'..='Z'
        | '!'
        | '$'
        | '%'
        | '&'
        | '*'
        | '/'
        | ':'
        | '<'
        | '='
        | '>'
        | '?'
        | '@'
        | '^'
        | '_'
        | '~' => true,
        _ => false,
    }
}

impl<CharIter: Iterator<Item = char>> Lexer<CharIter> {
    pub fn from_char_stream(char_stream: CharIter) -> Lexer<CharIter> {
        Self {
            current: None,
            peekable_char_stream: char_stream.peekable(),
            location: [1, 1],
        }
    }

    pub fn set_last_location(&mut self, location: [u32; 2]) {
        self.location = location;
    }

    fn try_next(&mut self) -> Result<Option<TokenData>> {
        match self.advance(1) {
            Some(c) => match c {
                ' ' | '\t' | '\n' | '\r' => self.atmosphere(),
                ';' => self.comment(),
                '(' => Ok(Some(TokenData::LeftParen)),
                ')' => Ok(Some(TokenData::RightParen)),
                '#' => match self.advance(1) {
                    Some(cn) => match cn {
                        '(' => Ok(Some(TokenData::VecConsIntro)),
                        't' => Ok(Some(TokenData::Boolean(true))),
                        'f' => Ok(Some(TokenData::Boolean(false))),
                        '\\' => match self.advance(1).take() {
                            Some(cnn) => Ok(Some(TokenData::Character(cnn))),
                            None => {
                                invalid_token!(Some(self.location), "expect character after #\\")
                            }
                        },
                        'u' => {
                            if Some('8') == self.advance(1).take()
                                && Some('(') == self.advance(1).take()
                            {
                                return Ok(Some(TokenData::ByteVecConsIntro));
                            } else {
                                invalid_token!(
                                    Some(self.location),
                                    "Imcomplete bytevector constant introducer"
                                );
                            }
                        }
                        _ => invalid_token!(Some(self.location), "expect '(' 't' or 'f' after #"),
                    },
                    None => invalid_token!(Some(self.location), "expect '(' 't' or 'f' after #"),
                },
                '\'' => Ok(Some(TokenData::Quote)),
                '`' => Ok(Some(TokenData::Quasiquote)),
                ',' => match self.peekable_char_stream.peek() {
                    Some(nc) => match nc {
                        '@' => {
                            self.advance(1).take();
                            Ok(Some(TokenData::UnquoteSplicing))
                        }
                        _ => Ok(Some(TokenData::Unquote)),
                    },
                    None => Ok(None),
                },
                '.' => match self.peekable_char_stream.peek() {
                    Some(c) => match c {
                        ' ' | '\t' | '\n' | '\r' | '(' | ')' | '"' | ';' | '|' => {
                            Ok(Some(TokenData::Period))
                        }
                        _ => self.percular_identifier(),
                    },
                    None => Ok(Some(TokenData::Period)),
                },
                '+' | '-' => match self.peekable_char_stream.peek() {
                    Some('0'..='9') => self.number(),
                    Some('.') => self.number(),
                    _ => self.percular_identifier(),
                },
                '"' => self.string(),
                '0'..='9' => self.number(),
                '|' => self.quote_identifier(),
                _ => self.normal_identifier(),
            },
            None => Ok(None),
        }
    }

    fn advance(&mut self, count: usize) -> &mut Option<char> {
        for _ in 0..count {
            self.current = self.peekable_char_stream.next();
            match self.current {
                Some('\n') => {
                    self.location[0] += 1;
                    self.location[1] = 1;
                }
                Some(_) => self.location[1] += 1,
                None => (),
            }
        }
        // println!("{:?} location {:?} ", self.current.unwrap(), self.location);
        &mut self.current
    }

    fn test_delimiter(location: Option<[u32; 2]>, c: char) -> Result<()> {
        match c {
            ' ' | '\t' | '\n' | '\r' | '(' | ')' | '"' | ';' | '|' => Ok(()),
            _ => invalid_token!(location, "Expect delimiter here instead of {}", c),
        }
    }

    fn atmosphere(&mut self) -> Result<Option<TokenData>> {
        while let Some(c) = self.peekable_char_stream.peek() {
            match c {
                ' ' | '\t' | '\n' | '\r' => {
                    self.advance(1);
                }
                _ => break,
            }
        }
        self.try_next()
    }

    fn comment(&mut self) -> Result<Option<TokenData>> {
        while let Some(c) = self.peekable_char_stream.peek() {
            match c {
                '\n' | '\r' => break,
                _ => {
                    self.advance(1);
                }
            }
        }
        self.try_next()
    }

    fn normal_identifier(&mut self) -> Result<Option<TokenData>> {
        match self.current {
            Some(c) => {
                let mut identifier_str = String::new();
                identifier_str.push(c);
                loop {
                    if let Some(nc) = self.peekable_char_stream.peek() {
                        match nc {
                            _ if is_identifier_initial(*nc) => identifier_str.push(*nc),
                            '0'..='9' | '+' | '-' | '.' | '@' => identifier_str.push(*nc),
                            _ => {
                                Self::test_delimiter(Some(self.location), *nc)?;
                                break;
                            }
                        }
                        self.advance(1);
                    } else {
                        break;
                    }
                }
                Ok(Some(TokenData::Identifier(identifier_str)))
            }
            None => Ok(None),
        }
    }

    fn dot_subsequent(&mut self, identifier_str: &mut String) -> Result<()> {
        if let Some(c) = self.peekable_char_stream.peek() {
            let valid = match c {
                '+' | '-' | '.' | '@' => true,
                _ => is_identifier_initial(*c),
            };
            match valid {
                true => loop {
                    match self.advance(1).take() {
                        Some(nc) => match nc {
                            _ if is_identifier_initial(nc) => identifier_str.push(nc),
                            '0'..='9' | '+' | '-' | '.' | '@' => identifier_str.push(nc),
                            _ => {
                                Self::test_delimiter(Some(self.location), nc)?;
                                break;
                            }
                        },
                        None => invalid_token!(
                            Some(self.location),
                            "Invalid Identifer {}",
                            identifier_str
                        ),
                    }
                },
                false => {
                    Self::test_delimiter(Some(self.location), *c)?;
                }
            }
        }
        Ok(())
    }

    fn percular_identifier(&mut self) -> Result<Option<TokenData>> {
        match self.current {
            Some(c) => {
                let mut identifier_str = String::new();
                identifier_str.push(c);
                match c {
                    '+' | '-' => {
                        let nc = self.peekable_char_stream.peek();
                        match nc {
                            Some('.') => {
                                self.advance(1);
                                identifier_str.push('.');
                                self.dot_subsequent(&mut identifier_str)?;
                            }
                            // a dot subsequent without a dot is a sign subsequent
                            Some(_) => self.dot_subsequent(&mut identifier_str)?,
                            None => {
                                self.advance(1);
                            }
                        }
                    }
                    '.' => self.dot_subsequent(&mut identifier_str)?,
                    _ => (),
                }
                Ok(Some(TokenData::Identifier(identifier_str)))
            }
            None => Ok(None),
        }
    }

    fn quote_identifier(&mut self) -> Result<Option<TokenData>> {
        let mut identifier_str = String::new();
        loop {
            match self.advance(1) {
                None => invalid_token!(
                    Some(self.location),
                    "Incomplete identifier {}",
                    identifier_str
                ),
                Some('|') => break Ok(Some(TokenData::Identifier(identifier_str))),
                Some(nc) => identifier_str.push(*nc),
            }
        }
    }

    fn string(&mut self) -> Result<Option<TokenData>> {
        match self.current {
            Some(_c) => {
                let mut string_literal = String::new();
                loop {
                    if let Some(c) = self.advance(1).take() {
                        match c {
                            '"' => break Ok(Some(TokenData::String(string_literal))),
                            '\\' => {
                                match self.advance(1) {
                                    Some(ec) => {
                                        match ec {
                                            'a' => string_literal.push('\u{007}'),
                                            'b' => string_literal.push('\u{008}'),
                                            't' => string_literal.push('\u{009}'),
                                            'n' => string_literal.push('\n'),
                                            'r' => string_literal.push('\r'),
                                            '"' => string_literal.push('"'),
                                            '\\' => string_literal.push('\\'),
                                            '|' => string_literal.push('|'),
                                            'x' => (), // TODO: 'x' for hex value
                                            ' ' => (), // TODO: space for nothing
                                            _ => invalid_token!(
                                                Some(self.location),
                                                "Unknown escape character"
                                            ),
                                        }
                                    }
                                    None => invalid_token!(
                                        Some(self.location),
                                        "Incomplete escape character"
                                    ),
                                }
                            }
                            _ => string_literal.push(c),
                        }
                    } else {
                        invalid_token!(Some(self.location), "Unclosed string literal")
                    }
                }
            }
            None => Ok(None),
        }
    }

    fn digital10(&mut self, number_literal: &mut String) -> Result<()> {
        loop {
            match self.peekable_char_stream.peek() {
                Some(nc) => match nc {
                    '0'..='9' => number_literal.push(*nc),
                    _ => {
                        break Ok(());
                    }
                },
                None => break Ok(()),
            }
            self.advance(1);
        }
    }

    fn number_suffix(&mut self, number_literal: &mut String) -> Result<()> {
        self.advance(1);
        number_literal.push('e');
        if let Some(sign) = self.peekable_char_stream.peek() {
            if (*sign == '+') || (*sign == '-') {
                number_literal.push(*sign);
                self.advance(1);
            }
        }
        self.digital10(number_literal)
    }

    fn real(&mut self, number_literal: &mut String) -> Result<()> {
        number_literal.push('.');
        self.advance(1);
        match self.peekable_char_stream.peek() {
            Some(nc) => match nc {
                'e' => self.number_suffix(number_literal),
                '0'..='9' => {
                    self.digital10(number_literal)?;
                    match self.peekable_char_stream.peek() {
                        Some('e') => self.number_suffix(number_literal),
                        Some(nnc) => Self::test_delimiter(Some(self.location), *nnc),
                        None => Ok(()),
                    }
                }
                _ => {
                    Self::test_delimiter(Some(self.location), *nc)?;
                    Ok(())
                }
            },
            None => Ok(()),
        }
    }

    fn number(&mut self) -> Result<Option<TokenData>> {
        match self.current.take() {
            Some(c) => {
                let mut number_literal = String::new();
                number_literal.push(c);
                loop {
                    let peek = self.peekable_char_stream.peek();
                    match peek {
                        Some(nc) => match nc {
                            '0'..='9' => self.digital10(&mut number_literal)?,
                            'e' => {
                                self.number_suffix(&mut number_literal)?;
                                break Ok(Some(TokenData::Real(number_literal)));
                            }
                            '.' => {
                                self.real(&mut number_literal)?;
                                break Ok(Some(TokenData::Real(number_literal)));
                            }
                            '/' => {
                                let mut denominator = String::new();
                                self.advance(1);
                                self.digital10(&mut denominator)?;
                                break Ok(Some(TokenData::Rational(
                                    number_literal.parse::<i32>().unwrap(),
                                    match denominator.parse::<u32>().unwrap() {
                                        0 => invalid_token!(
                                            Some(self.location),
                                            "rational denominator should not be 0!"
                                        ),
                                        other => other,
                                    },
                                )));
                            }
                            _ => {
                                Self::test_delimiter(Some(self.location), *nc)?;
                                break Ok(Some(TokenData::Integer(
                                    number_literal.parse::<i32>().unwrap(),
                                )));
                            }
                        },
                        None => {
                            break Ok(Some(TokenData::Integer(
                                number_literal.parse::<i32>().unwrap(),
                            )))
                        }
                    }
                }
            }
            None => Ok(None),
        }
    }
}

fn tokenize(text: &str) -> Result<Vec<TokenData>> {
    let mut iter = text.chars().peekable();
    let c = Lexer::from_char_stream(&mut iter);
    Ok(c.collect::<Result<Vec<_>>>()?
        .into_iter()
        .map(|t| t.data)
        .collect())
}

#[test]
fn simple_tokens() -> Result<()> {
    assert_eq!(
        tokenize("#t#f()#()#u8()'`,,@.")?,
        vec![
            TokenData::Boolean(true),
            TokenData::Boolean(false),
            TokenData::LeftParen,
            TokenData::RightParen,
            TokenData::VecConsIntro,
            TokenData::RightParen,
            TokenData::ByteVecConsIntro,
            TokenData::RightParen,
            TokenData::Quote,
            TokenData::Quasiquote,
            TokenData::Unquote,
            TokenData::UnquoteSplicing,
            TokenData::Period
        ]
    );
    Ok(())
}

#[test]
fn identifier() -> Result<()> {
    assert_eq!(
        tokenize(
            "
        ... +
        +soup+ <=?
        ->string a34kTMNs
        lambda list->vector
        q V17a
        |two words| |two; words|"
        )?,
        vec![
            TokenData::Identifier(String::from("...")),
            TokenData::Identifier(String::from("+")),
            TokenData::Identifier(String::from("+soup+")),
            TokenData::Identifier(String::from("<=?")),
            TokenData::Identifier(String::from("->string")),
            TokenData::Identifier(String::from("a34kTMNs")),
            TokenData::Identifier(String::from("lambda")),
            TokenData::Identifier(String::from("list->vector")),
            TokenData::Identifier(String::from("q")),
            TokenData::Identifier(String::from("V17a")),
            TokenData::Identifier(String::from("two words")),
            TokenData::Identifier(String::from("two; words"))
        ]
    );

    Ok(())
}

fn period() -> Result<()> {
    assert_eq!(tokenize(".")?, vec![TokenData::Period]);
    Ok(())
}

#[test]
fn character() -> Result<()> {
    assert_eq!(
        tokenize("#\\a#\\ #\\\t")?,
        vec![
            TokenData::Character('a'),
            TokenData::Character(' '),
            TokenData::Character('\t')
        ]
    );
    Ok(())
}

#[test]
fn string() -> Result<()> {
    assert_eq!(
        tokenize("\"()+-123\"\"\\\"\"\"\\a\\b\\t\\r\\n\\\\\\|\"")?,
        vec![
            TokenData::String(String::from("()+-123")),
            TokenData::String(String::from("\"")),
            TokenData::String(String::from("\u{007}\u{008}\t\r\n\\|"))
        ]
    );
    Ok(())
}

#[test]
fn number() -> Result<()> {
    assert_eq!(
        tokenize(
            "+123 123 -123
                    1.23 -12.34 1. 0. +.0 -.1
                    1e10 1.3e20 -43.e-12 +.12e+12
                    1/2 +1/2 -32/3
            "
        )?,
        vec![
            TokenData::Integer(123),
            TokenData::Integer(123),
            TokenData::Integer(-123),
            TokenData::Real("1.23".to_string()),
            TokenData::Real("-12.34".to_string()),
            TokenData::Real("1.".to_string()),
            TokenData::Real("0.".to_string()),
            TokenData::Real("+.0".to_string()),
            TokenData::Real("-.1".to_string()),
            TokenData::Real("1e10".to_string()),
            TokenData::Real("1.3e20".to_string()),
            TokenData::Real("-43.e-12".to_string()),
            TokenData::Real("+.12e+12".to_string()),
            TokenData::Rational(1, 2),
            TokenData::Rational(1, 2),
            TokenData::Rational(-32, 3),
        ]
    );
    assert_eq!(
        tokenize("1/0"),
        Err(SchemeError {
            category: ErrorType::Lexical,
            message: "rational denominator should not be 0!".to_string(),
            location: Some([1, 4])
        })
    );
    assert_eq!(
        tokenize("1/00"),
        Err(SchemeError {
            category: ErrorType::Lexical,
            message: "rational denominator should not be 0!".to_string(),
            location: Some([1, 5])
        })
    );
    Ok(())
}

#[test]

fn atmosphere() -> Result<()> {
    assert_eq!(
        tokenize("\t(- \n4\r(+ 1 2))")?,
        vec![
            TokenData::LeftParen,
            TokenData::Identifier(String::from("-")),
            TokenData::Integer(4),
            TokenData::LeftParen,
            TokenData::Identifier(String::from("+")),
            TokenData::Integer(1),
            TokenData::Integer(2),
            TokenData::RightParen,
            TokenData::RightParen
        ]
    );
    Ok(())
}

#[test]
fn comment() -> Result<()> {
    assert_eq!(
        tokenize("abcd;+-12\n 12;dew\r34")?,
        vec![
            TokenData::Identifier(String::from("abcd")),
            TokenData::Integer(12),
            TokenData::Integer(34)
        ]
    );
    Ok(())
}
