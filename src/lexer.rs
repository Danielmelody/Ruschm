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
    VecConsIntro,
    ByteVecConsIntro,
    Quote,
    Quasiquote, // BackQuote
    Unquote,
    Comma,
    CommaAt,
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
        while let Some(c) = self.current {
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
                                    Some(cnn) => self.push_advance(current_iter, Token::Character(cnn)),
                                    None => {
                                        return Err(InvalidToken {
                                            error: String::from("expect character after #\\"),
                                        })
                                    }
                                }
                            }
                            'u' => {
                                if Some('8') == current_iter.next()
                                    && Some('(') == current_iter.next()
                                {
                                    self.push_advance(current_iter, Token::ByteVecConsIntro);
                                } else {
                                    return Err(InvalidToken {
                                        error: String::from(
                                            "Imcomplete bytevector constant introducer",
                                        ),
                                    });
                                }
                            }
                            _ => {
                                return Err(InvalidToken {
                                    error: String::from("expect '(' 't' or 'f' after #"),
                                })
                            }
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
                            self.push_advance(current_iter, Token::CommaAt);
                        }
                        _ => self.push_advance(current_iter, Token::Comma),
                    },
                    None => (),
                },
                '.' => match current_iter.clone().next() {
                    Some('.') => self.identifier(current_iter)?,
                    _ => self.push_advance(current_iter, Token::Period),
                },
                '+' | '-' => {
                    match current_iter.clone().next() {
                        Some('0'...'9') => self.number(current_iter)?,
                        _ => self.push_advance(current_iter, Token::Identifier(c.to_string())),
                    };
                }
                '"' => self.string(current_iter)?,
                '0'...'9' => self.number(current_iter)?,
                _ => self.identifier(current_iter)?,
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
                _ => ()
            }
            self.current = current_iter.next();
        }
        Ok(())
    }

    fn identifier(&mut self, current_iter: &mut std::str::Chars) -> Result<(), InvalidToken> {
        fn initial(current: char) -> bool {
            match current {
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
        if let Some(c) = self.current {
            if initial(c) {
                let mut identifier_str = String::new();
                identifier_str.push(c);
                loop {
                    self.current = current_iter.next();
                    if let Some(nc) = self.current {
                        match nc {
                            _ if initial(nc) => identifier_str.push(nc),
                            '0'...'9' | '+' | '-' | '.' | '@' => identifier_str.push(nc),
                            _ => break,
                        }
                    } else {
                        break;
                    }
                }
                self.tokens.push(Token::Identifier(identifier_str));
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
                                        _ => {
                                            return Err(InvalidToken {
                                                error: String::from("Unknown escape character"),
                                            })
                                        }
                                    }
                                }
                                None => {
                                    return Err(InvalidToken {
                                        error: String::from("Incomplete escape character"),
                                    })
                                }
                            }
                        }
                        _ => string_literal.push(c),
                    }
                } else {
                    return Err(InvalidToken {
                        error: String::from("Unclosed string literal"),
                    });
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
                        _ => break,
                    },
                    None => break,
                }
            }
            match number_str.parse() {
                Ok(number) => self.tokens.push(Token::Number(number)),
                Err(_) => {
                    return Err(InvalidToken {
                        error: format!("Unrecognized number: {}", number_str),
                    })
                }
            }
        }
        Ok(())
    }

    fn push_advance(&mut self, current_iter:&mut std::str::Chars, token: Token) {
        self.tokens.push(token);
        self.current = current_iter.next();
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
            Token::Comma,
            Token::CommaAt,
            Token::Period
        ]
    );
    Ok(())
}

#[test]
fn identifier() -> Result<(), InvalidToken> {
    let mut l = Lexer::new();
    l.tokenize("+-xyz123_")?;
    assert_eq!(
        l.tokens,
        vec![
            Token::Identifier(String::from("+")),
            Token::Identifier(String::from("-")),
            Token::Identifier(String::from("xyz123_"))
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
    l.tokenize("+123-123++123")?;
    assert_eq!(
        l.tokens,
        vec![
            Token::Number(123),
            Token::Number(-123),
            Token::Identifier(String::from("+")),
            Token::Number(123),
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
    assert_eq!(
        l.tokens,
        vec![Token::Identifier(String::from("abcd"))]
    );
    Ok(())
}
