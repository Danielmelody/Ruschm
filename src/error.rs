use std::fmt;

#[derive(Debug, PartialEq)]
pub enum ErrorType {
    Lexical,
    Syntax,
    Logic,
}

#[derive(Debug, PartialEq)]
pub struct Error {
    pub category: ErrorType,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.category {
            ErrorType::Lexical => write!(f, "Invalid token: {}", self.message),
            ErrorType::Syntax => write!(f, "Syntax error: {}", self.message),
            ErrorType::Logic => write!(f, "error: {}", self.message),
        }
    }
}
