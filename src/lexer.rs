#![allow(dead_code)]
use std::fmt;
use std::iter::Iterator;

#[derive(PartialEq, Debug)]
enum Token {
    Identifier(String),
    Boolean(bool),
    Number(i64), // exact integers only
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

#[derive(Debug)]
struct TokenError {
    error: String,
}

impl fmt::Display for TokenError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Invalid token: {}", self.error)
    }
}

struct TokenGenerator<CharIter: Iterator<Item = char> + Clone> {
    current: Option<char>,
    text_iterator: CharIter,
}

macro_rules! invalid_token {
    ($($arg:tt)*) => (
        return Err(TokenError { error: format!($($arg)*) });
    )
}

impl<CharIter: Iterator<Item = char> + Clone> Iterator for TokenGenerator<CharIter> {
    type Item = Token;
    fn next(&mut self) -> Option<Self::Item> {
        match self.try_next() {
            Ok(ret) => ret,
            Err(_e) => None,
        }
    }
}

fn test_delimiter(c: char) -> Result<(), TokenError> {
    match c {
        ' ' | '\t' | '\n' | '\r' | '(' | ')' | '"' | ';' | '|' => Ok(()),
        _ => invalid_token!("Expect delimiter here instead of {}", c),
    }
}

fn is_identifier_initial(c: char) -> bool {
    match c {
        'a'...'z'
        | 'A'...'Z'
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

impl<CharIter: Iterator<Item = char> + Clone> TokenGenerator<CharIter> {
    pub fn new(mut text_iterator: CharIter) -> TokenGenerator<CharIter> {
        Self {
            current: text_iterator.next(),
            text_iterator: text_iterator,
        }
    }

    fn try_next(&mut self) -> Result<Option<Token>, TokenError> {
        match self.current {
            Some(c) => match c {
                ' ' | '\t' | '\n' | '\r' => self.atmosphere(),
                ';' => self.comment(),
                '(' => self.generate(Token::LeftParen),
                ')' => self.generate(Token::RightParen),
                '#' => match self.nextchar() {
                    Some(cn) => match cn {
                        '(' => self.generate(Token::VecConsIntro),
                        't' => self.generate(Token::Boolean(true)),
                        'f' => self.generate(Token::Boolean(false)),
                        '\\' => match self.nextchar() {
                            Some(cnn) => self.generate(Token::Character(cnn)),
                            None => invalid_token!("expect character after #\\"),
                        },
                        'u' => {
                            if Some('8') == self.nextchar() && Some('(') == self.nextchar() {
                                return self.generate(Token::ByteVecConsIntro);
                            } else {
                                invalid_token!("Imcomplete bytevector constant introducer");
                            }
                        }
                        _ => invalid_token!("expect '(' 't' or 'f' after #"),
                    },
                    None => invalid_token!("expect '(' 't' or 'f' after #"),
                },
                '\'' => self.generate(Token::Quote),
                '`' => self.generate(Token::Quasiquote),
                ',' => match self.text_iterator.clone().next() {
                    Some(nc) => match nc {
                        '@' => {
                            self.nextchar();
                            self.generate(Token::UnquoteSplicing)
                        }
                        _ => self.generate(Token::Unquote),
                    },
                    None => Ok(None),
                },
                '.' => self.percular_identifier(),
                '+' | '-' => match self.text_iterator.clone().next() {
                    Some('0'...'9') => self.number(),
                    _ => self.percular_identifier(),
                },
                '"' => self.string(),
                '0'...'9' => self.number(),
                '|' => self.quote_identifier(),
                _ => self.normal_identifier(),
            },
            None => Ok(None),
        }
    }

    fn nextchar(&mut self) -> Option<char> {
        self.current = self.text_iterator.next();
        self.current
    }

    fn atmosphere(&mut self) -> Result<Option<Token>, TokenError> {
        while let Some(c) = self.current {
            match c {
                ' ' | '\t' | '\n' | '\r' => (),
                _ => break,
            }
            self.nextchar();
        }
        self.try_next()
    }

    fn comment(&mut self) -> Result<Option<Token>, TokenError> {
        while let Some(c) = self.current {
            match c {
                '\n' | '\r' => break,
                _ => (),
            }
            self.nextchar();
        }
        Ok(None)
    }

    fn normal_identifier(&mut self) -> Result<Option<Token>, TokenError> {
        match self.current {
            Some(c) => {
                let mut identifier_str = String::new();
                identifier_str.push(c);
                loop {
                    if let Some(nc) = self.nextchar() {
                        match nc {
                            _ if is_identifier_initial(nc) => identifier_str.push(nc),
                            '0'...'9' | '+' | '-' | '.' | '@' => identifier_str.push(nc),
                            _ => {
                                test_delimiter(nc)?;
                                break;
                            }
                        }
                    } else {
                        break;
                    }
                }
                Ok(Some(Token::Identifier(identifier_str)))
            }
            None => Ok(None),
        }
    }

    fn dot_subsequent(&mut self, identifier_str: &mut String) -> Result<(), TokenError> {
        if let Some(c) = self.nextchar() {
            let valid = match c {
                '+' | '-' | '.' | '@' => true,
                _ => is_identifier_initial(c),
            };
            match valid {
                true => {
                    identifier_str.push(c);
                    loop {
                        match self.nextchar() {
                            Some(nc) => match nc {
                                _ if is_identifier_initial(nc) => identifier_str.push(nc),
                                '0'...'9' | '+' | '-' | '.' | '@' => identifier_str.push(nc),
                                _ => {
                                    test_delimiter(nc)?;
                                    break;
                                }
                            },
                            None => invalid_token!("Invalid Indentifer {}", identifier_str),
                        }
                    }
                }
                false => {
                    test_delimiter(c)?;
                }
            }
        }
        Ok(())
    }

    fn percular_identifier(&mut self) -> Result<Option<Token>, TokenError> {
        match self.current {
            Some(c) => {
                let mut identifier_str = String::new();
                identifier_str.push(c);
                match c {
                    '+' | '-' => {
                        let nc = self.text_iterator.clone().next();
                        match nc {
                            Some('.') => {
                                self.nextchar();
                                identifier_str.push('.');
                                self.dot_subsequent(&mut identifier_str)?;
                            }
                            // a dot subsequent without a dot is a sign subsequent
                            Some(_) => self.dot_subsequent(&mut identifier_str)?,
                            None => {
                                self.nextchar();
                            }
                        }
                    }
                    '.' => self.dot_subsequent(&mut identifier_str)?,
                    _ => (),
                }
                match identifier_str.as_str() {
                    "." => Ok(Some(Token::Period)),
                    _ => Ok(Some(Token::Identifier(identifier_str))),
                }
            }
            None => Ok(None),
        }
    }

    fn quote_identifier(&mut self) -> Result<Option<Token>, TokenError> {
        let mut identifier_str = String::new();
        loop {
            self.current = self.text_iterator.next();
            match self.current {
                None => invalid_token!("Incomplete identifier {}", identifier_str),
                Some('|') => break self.generate(Token::Identifier(identifier_str)),
                Some(nc) => identifier_str.push(nc),
            }
        }
    }

    fn string(&mut self) -> Result<Option<Token>, TokenError> {
        match self.current {
            Some(_c) => {
                let mut string_literal = String::new();
                loop {
                    if let Some(c) = self.nextchar() {
                        match c {
                            '"' => break self.generate(Token::String(string_literal)),
                            '\\' => {
                                match self.nextchar() {
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
                                            _ => invalid_token!("Unknown escape character"),
                                        }
                                    }
                                    None => invalid_token!("Incomplete escape character"),
                                }
                            }
                            _ => string_literal.push(c),
                        }
                    } else {
                        invalid_token!("Unclosed string literal")
                    }
                }
            }
            None => Ok(None),
        }
    }

    fn number(&mut self) -> Result<Option<Token>, TokenError> {
        match self.current {
            Some(c) => {
                let mut number_str = String::new();
                number_str.push(c);
                loop {
                    match self.nextchar() {
                        Some(nc) => match nc {
                            '0'...'9' => {
                                number_str.push(nc);
                            }
                            _ => {
                                test_delimiter(nc)?;
                                break;
                            }
                        },
                        None => break,
                    }
                }
                match number_str.parse() {
                    Ok(number) => Ok(Some(Token::Number(number))),
                    Err(_) => invalid_token!("Unrecognized number: {}", number_str),
                }
            }
            None => Ok(None),
        }
    }

    fn generate(&mut self, token: Token) -> Result<Option<Token>, TokenError> {
        self.nextchar();
        Ok(Some(token))
    }
}

#[test]
fn empty_text() {
    let c = TokenGenerator::new("".chars());
    assert_eq!(c.current, None);
}

#[test]
fn simple_tokens() -> Result<(), TokenError> {
    let l = TokenGenerator::new("#t#f()#()#u8()'`,,@.".chars());
    assert_eq!(
        l.collect::<Vec<Token>>(),
        vec![
            Token::Boolean(true),
            Token::Boolean(false),
            Token::LeftParen,
            Token::RightParen,
            Token::VecConsIntro,
            Token::RightParen,
            Token::ByteVecConsIntro,
            Token::RightParen,
            Token::Quote,
            Token::Quasiquote,
            Token::Unquote,
            Token::UnquoteSplicing,
            Token::Period
        ]
    );
    Ok(())
}

#[test]
fn identifier() -> Result<(), TokenError> {
    let l = TokenGenerator::new(
        "
    ... +
    +soup+ <=?
    ->string a34kTMNs
    lambda list->vector
    q V17a
    |two words| |two; words|"
            .chars(),
    );
    assert_eq!(
        l.collect::<Vec<_>>(),
        vec![
            Token::Identifier(String::from("...")),
            Token::Identifier(String::from("+")),
            Token::Identifier(String::from("+soup+")),
            Token::Identifier(String::from("<=?")),
            Token::Identifier(String::from("->string")),
            Token::Identifier(String::from("a34kTMNs")),
            Token::Identifier(String::from("lambda")),
            Token::Identifier(String::from("list->vector")),
            Token::Identifier(String::from("q")),
            Token::Identifier(String::from("V17a")),
            Token::Identifier(String::from("two words")),
            Token::Identifier(String::from("two; words"))
        ]
    );

    Ok(())
}

#[test]
fn character() -> Result<(), TokenError> {
    let l = TokenGenerator::new("#\\a#\\ #\\\t".chars());
    assert_eq!(
        l.collect::<Vec<_>>(),
        vec![
            Token::Character('a'),
            Token::Character(' '),
            Token::Character('\t')
        ]
    );
    Ok(())
}

#[test]
fn string() -> Result<(), TokenError> {
    let l = TokenGenerator::new("\"()+-123\"\"\\\"\"\"\\a\\b\\t\\r\\n\\\\\\|\"".chars());
    assert_eq!(
        l.collect::<Vec<_>>(),
        vec![
            Token::String(String::from("()+-123")),
            Token::String(String::from("\"")),
            Token::String(String::from("\u{007}\u{008}\t\r\n\\|"))
        ]
    );
    Ok(())
}

#[test]
fn number() -> Result<(), TokenError> {
    let l = TokenGenerator::new("+123 -123 + -123".chars());
    assert_eq!(
        l.collect::<Vec<_>>(),
        vec![
            Token::Number(123),
            Token::Number(-123),
            Token::Identifier(String::from("+")),
            Token::Number(-123),
        ]
    );
    Ok(())
}

#[test]

fn atmosphere() -> Result<(), TokenError> {
    let l = TokenGenerator::new("\t(- \n4\r(+ 1 2))".chars());
    assert_eq!(
        l.collect::<Vec<_>>(),
        vec![
            Token::LeftParen,
            Token::Identifier(String::from("-")),
            Token::Number(4),
            Token::LeftParen,
            Token::Identifier(String::from("+")),
            Token::Number(1),
            Token::Number(2),
            Token::RightParen,
            Token::RightParen
        ]
    );
    Ok(())
}

#[test]
fn comment() -> Result<(), TokenError> {
    let l = TokenGenerator::new("abcd;+-12\t 12".chars());
    assert_eq!(
        l.collect::<Vec<_>>(),
        vec![Token::Identifier(String::from("abcd"))]
    );
    Ok(())
}
