#![allow(dead_code)]

use std::error::Error;
use std::fmt;

#[derive(Debug, PartialEq)]
pub enum ErrorType {
    Lexical,
    Syntax,
    Logic,
}

#[derive(Debug, PartialEq)]
pub struct SchemeError {
    pub category: ErrorType,
    pub message: String,
}

impl Error for SchemeError {}

impl fmt::Display for SchemeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.category {
            ErrorType::Lexical => write!(f, "Invalid token: {}", self.message),
            ErrorType::Syntax => write!(f, "Syntax error: {}", self.message),
            ErrorType::Logic => write!(f, "error: {}", self.message),
        }
    }
}

macro_rules! invalid_token {
    ($($arg:tt)*) => (
        return Err(SchemeError {category: ErrorType::Lexical, message: format!($($arg)*) });
    )
}

macro_rules! syntax_error {
    ($($arg:tt)*) => (
        return Err(SchemeError {category: ErrorType::Syntax, message: format!($($arg)*) });
    )
}

macro_rules! logic_error {
    ($($arg:tt)*) => (
        return Err(SchemeError {category: ErrorType::Logic , message: format!($($arg)*) });
    )
}
