use std::fmt;
use std::str;

#[derive(PartialEq, Debug)]
enum Token {
    Identifier(String),
    Boolean(bool),
    Number(i64),
    Character(char),
    String(String),
    LeftParenthesis,
    RightParenthesis,
    SharpParenthesis,
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
        while let Some(c) = chars.next() {
            self.current = Some(c);
            self.token(&mut chars)?;
        }
        Ok(())
    }

    fn token(&mut self, current_iter: &mut std::str::Chars) -> Result<(), InvalidToken> {
        match self.current {
            Some(c) => match c {
                '(' => self.tokens.push(Token::LeftParenthesis),
                ')' => self.tokens.push(Token::RightParenthesis),
                '#' => match current_iter.next() {
                    Some(c) => match c {
                        '(' => self.tokens.push(Token::SharpParenthesis),
                        't' => self.tokens.push(Token::Boolean(true)),
                        'f' => self.tokens.push(Token::Boolean(false)),
                        _ => {
                            return Err(InvalidToken {
                                error: String::from("expect '(' 't' or 'f' after #"),
                            })
                        }
                    },
                    None => (),
                },
                '\'' => self.tokens.push(Token::Quote),
                '`' => self.tokens.push(Token::Quasiquote),
                ',' => match current_iter.clone().next() {
                    Some(nc) => match nc {
                        '@' => {
                            self.current = current_iter.next();
                            self.tokens.push(Token::CommaAt);
                        }
                        _ => self.tokens.push(Token::Comma),
                    },
                    None => (),
                },
                '.' => self.tokens.push(Token::Period),
                _ => self.identifier(current_iter)?,
            },
            None => (),
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
                | '^'
                | '_'
                | '~' => true,
                _ => false,
            }
        }
        if let Some(c) = self.current {
            match c {
                '+' => self.tokens.push(Token::Identifier(String::from("+"))),
                '-' => self.tokens.push(Token::Identifier(String::from("-"))),
                _ => {
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
            }
        }
        Ok(())
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
    l.tokenize("#t#f()#('`,,@.")?;
    assert_eq!(
        l.tokens,
        vec![
            Token::Boolean(true),
            Token::Boolean(false),
            Token::LeftParenthesis,
            Token::RightParenthesis,
            Token::SharpParenthesis,
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
