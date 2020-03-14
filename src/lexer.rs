#![allow(dead_code)]
use std::fmt;
use std::str;

#[derive(PartialEq, Debug)]
enum Token {
    Identifier(String),
    Boolean(bool),
    Number(i64), // exact integers only
    Character(char),
    String(String),
    LeftParen,
    RightParen,
    VecConsIntro, // #(...)
    ByteVecConsIntro, // #u8(...)
    Quote, // '
    Quasiquote, // BackQuote
    Unquote, // ,
    UnquoteSplicing, // ,@
    Period,
}

#[derive(Debug)]
struct InvalidToken {
    error: String,
}

impl fmt::Display for InvalidToken {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Invalid token: {}", self.error)
    }
}

struct Lexer {
    current: Option<char>,
    tokens: Vec<Token>,
}

macro_rules! invalid_token {
    ($($arg:tt)*) => (
        return Err(InvalidToken { error: format!($($arg)*) })
    )
}

impl Lexer {
    pub fn new() -> Lexer {
        Self {
            current: None,
            tokens: Vec::new(),
        }
    }

    pub fn tokenize(&mut self, input: &str) -> Result<(), InvalidToken> {
        let mut chars = input.chars();
        self.current = chars.next();
        while let Some(_) = self.current {
            self.token(&mut chars)?;
        }
        Ok(())
    }

    fn token(&mut self, current_iter: &mut std::str::Chars) -> Result<(), InvalidToken> {
        match self.current {
            Some(c) => match c {
                ' ' | '\t' | '\n' | '\r' => self.atmosphere(current_iter)?,
                ';' => self.comment(current_iter)?,
                '(' => self.push_advance(current_iter, Token::LeftParen),
                ')' => self.push_advance(current_iter, Token::RightParen),
                '#' => {
                    self.current = current_iter.next();
                    match self.current {
                        Some(cn) => match cn {
                            '(' => self.push_advance(current_iter, Token::VecConsIntro),
                            't' => self.push_advance(current_iter, Token::Boolean(true)),
                            'f' => self.push_advance(current_iter, Token::Boolean(false)),
                            '\\' => {
                                self.current = current_iter.next();
                                match self.current {
                                    Some(cnn) => {
                                        self.push_advance(current_iter, Token::Character(cnn))
                                    }
                                    None => invalid_token!("expect character after #\\"),
                                }
                            }
                            'u' => {
                                if Some('8') == current_iter.next()
                                    && Some('(') == current_iter.next()
                                {
                                    self.push_advance(current_iter, Token::ByteVecConsIntro);
                                } else {
                                    invalid_token!("Imcomplete bytevector constant introducer");
                                }
                            }
                            _ => invalid_token!("expect '(' 't' or 'f' after #"),
                        },
                        None => (),
                    }
                }
                '\'' => self.push_advance(current_iter, Token::Quote),
                '`' => self.push_advance(current_iter, Token::Quasiquote),
                ',' => match current_iter.clone().next() {
                    Some(nc) => match nc {
                        '@' => {
                            self.current = current_iter.next();
                            self.push_advance(current_iter, Token::UnquoteSplicing);
                        }
                        _ => self.push_advance(current_iter, Token::Unquote),
                    },
                    None => (),
                },
                '.' => self.percular_identifier(current_iter)?,
                '+' | '-' => match current_iter.clone().next() {
                    Some('0'...'9') => self.number(current_iter)?,
                    _ => self.percular_identifier(current_iter)?,
                },
                '"' => self.string(current_iter)?,
                '0'...'9' => self.number(current_iter)?,
                '|' => self.quote_identifier(current_iter)?,
                _ => self.normal_identifier(current_iter)?,
            },
            None => (),
        }
        Ok(())
    }

    fn atmosphere(&mut self, current_iter: &mut std::str::Chars) -> Result<(), InvalidToken> {
        while let Some(c) = self.current {
            match c {
                ' ' | '\t' | '\n' | '\r' => (),
                _ => break,
            }
            self.current = current_iter.next();
        }
        Ok(())
    }

    fn comment(&mut self, current_iter: &mut std::str::Chars) -> Result<(), InvalidToken> {
        while let Some(c) = self.current {
            match c {
                '\n' | '\r' => break,
                _ => (),
            }
            self.current = current_iter.next();
        }
        Ok(())
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

    fn normal_identifier(
        &mut self,
        current_iter: &mut std::str::Chars,
    ) -> Result<(), InvalidToken> {
        if let Some(c) = self.current {
            let mut identifier_str = String::new();
            identifier_str.push(c);
            loop {
                self.current = current_iter.next();
                if let Some(nc) = self.current {
                    match nc {
                        _ if Self::is_identifier_initial(nc) => identifier_str.push(nc),
                        '0'...'9' | '+' | '-' | '.' | '@' => identifier_str.push(nc),
                        _ => {
                            Lexer::test_delimiter(nc)?;
                            break;
                        }
                    }
                } else {
                    break;
                }
            }
            self.tokens.push(Token::Identifier(identifier_str));
        }
        Ok(())
    }

    fn dot_subsequent(
        &mut self,
        identifier_str: &mut String,
        current_iter: &mut std::str::Chars,
    ) -> Result<(), InvalidToken> {
        self.current = current_iter.next();
        if let Some(c) = self.current {
            let valid = match c {
                '+' | '-' | '.' | '@' => true,
                _ => Lexer::is_identifier_initial(c),
            };
            match valid {
                true => {
                    identifier_str.push(c);
                    loop {
                        self.current = current_iter.next();
                        match self.current {
                            Some(nc) => match nc {
                                _ if Lexer::is_identifier_initial(nc) => identifier_str.push(nc),
                                '0'...'9' | '+' | '-' | '.' | '@' => identifier_str.push(nc),
                                _ => {
                                    Lexer::test_delimiter(nc)?;
                                    break;
                                }
                            },
                            None => invalid_token!("Invalid Indentifer {}", identifier_str),
                        }
                    }
                }
                false => Lexer::test_delimiter(c)?,
            }
        }
        Ok(())
    }

    fn percular_identifier(
        &mut self,
        current_iter: &mut std::str::Chars,
    ) -> Result<(), InvalidToken> {
        if let Some(c) = self.current {
            let mut identifier_str = String::new();
            identifier_str.push(c);
            match c {
                '+' | '-' => {
                    let nc = current_iter.clone().next();
                    match nc {
                        Some('.') => {
                            self.current = current_iter.next();
                            identifier_str.push('.');
                            self.dot_subsequent(&mut identifier_str, current_iter)?;
                        }
                        // a dot subsequent without a dot is a sign subsequent
                        Some(_) => self.dot_subsequent(&mut identifier_str, current_iter)?,
                        None => self.current = current_iter.next(),
                    }
                }
                '.' => self.dot_subsequent(&mut identifier_str, current_iter)?,
                _ => (),
            }
            match identifier_str.as_str() {
                "." => self.tokens.push(Token::Period),
                _ => self.tokens.push(Token::Identifier(identifier_str)),
            }
        }
        Ok(())
    }

    fn quote_identifier(&mut self, current_iter: &mut std::str::Chars) -> Result<(), InvalidToken> {
        let mut identifier_str = String::new();
        loop {
            self.current = current_iter.next();
            match self.current {
                None => invalid_token!("Incomplete identifier {}", identifier_str),
                Some('|') => {
                    self.push_advance(current_iter, Token::Identifier(identifier_str));
                    break;
                }
                Some(nc) => identifier_str.push(nc),
            }
        }
        Ok(())
    }

    fn string(&mut self, current_iter: &mut std::str::Chars) -> Result<(), InvalidToken> {
        if let Some('"') = self.current {
            let mut string_literal = String::new();
            loop {
                self.current = current_iter.next();
                if let Some(c) = self.current {
                    match c {
                        '"' => {
                            self.tokens.push(Token::String(string_literal));
                            break;
                        }
                        '\\' => {
                            self.current = current_iter.next();
                            match self.current {
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
        self.current = current_iter.next();
        Ok(())
    }

    fn number(&mut self, current_iter: &mut std::str::Chars) -> Result<(), InvalidToken> {
        if let Some(c) = self.current {
            let mut number_str = String::new();
            number_str.push(c);
            loop {
                self.current = current_iter.next();
                match self.current {
                    Some(nc) => match nc {
                        '0'...'9' => {
                            number_str.push(nc);
                        }
                        _ => {
                            Lexer::test_delimiter(nc)?;
                            break;
                        }
                    },
                    None => break,
                }
            }
            match number_str.parse() {
                Ok(number) => self.tokens.push(Token::Number(number)),
                Err(_) => invalid_token!("Unrecognized number: {}", number_str),
            }
        }
        Ok(())
    }

    fn push_advance(&mut self, current_iter: &mut std::str::Chars, token: Token) {
        self.tokens.push(token);
        self.current = current_iter.next();
    }

    fn test_delimiter(c: char) -> Result<(), InvalidToken> {
        match c {
            ' ' | '\t' | '\n' | '\r' | '(' | ')' | '"' | ';' | '|' => Ok(()),
            _ => invalid_token!("Expect delimiter here instead of {}", c),
        }
    }
}

#[test]
fn empty_text() {
    let c = Lexer::new();
    assert_eq!(c.current, None);
}

#[test]
fn simple_tokens() -> Result<(), InvalidToken> {
    let mut l = Lexer::new();
    l.tokenize("#t#f()#()#u8()'`,,@.")?;
    assert_eq!(
        l.tokens,
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
fn identifier() -> Result<(), InvalidToken> {
    let mut l = Lexer::new();
    l.tokenize(
        "
    ... +
    +soup+ <=?
    ->string a34kTMNs
    lambda list->vector
    q V17a
    |two words| |two; words|",
    )?;
    assert_eq!(
        l.tokens,
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
fn character() -> Result<(), InvalidToken> {
    let mut l = Lexer::new();
    l.tokenize("#\\a#\\ #\\\t")?;
    assert_eq!(
        l.tokens,
        vec![
            Token::Character('a'),
            Token::Character(' '),
            Token::Character('\t')
        ]
    );
    Ok(())
}

#[test]
fn string() -> Result<(), InvalidToken> {
    let mut l = Lexer::new();
    l.tokenize("\"()+-123\"\"\\\"\"\"\\a\\b\\t\\r\\n\\\\\\|\"")?;
    assert_eq!(
        l.tokens,
        vec![
            Token::String(String::from("()+-123")),
            Token::String(String::from("\"")),
            Token::String(String::from("\u{007}\u{008}\t\r\n\\|"))
        ]
    );
    Ok(())
}

#[test]
fn number() -> Result<(), InvalidToken> {
    let mut l = Lexer::new();
    l.tokenize("+123 -123 + -123")?;
    assert_eq!(
        l.tokens,
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

fn atmosphere() -> Result<(), InvalidToken> {
    let mut l = Lexer::new();
    l.tokenize("\t(- \n4\r(+ 1 2))")?;
    assert_eq!(
        l.tokens,
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
fn comment() -> Result<(), InvalidToken> {
    let mut l = Lexer::new();
    l.tokenize("abcd;+-12\t 12")?;
    assert_eq!(l.tokens, vec![Token::Identifier(String::from("abcd"))]);
    Ok(())
}
