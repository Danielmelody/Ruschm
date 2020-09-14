#![allow(dead_code)]

use std::error::Error;
use std::fmt;

use fmt::Display;

#[derive(Debug, PartialEq, Clone)]
pub enum ErrorType {
    Lexical,
    Syntax,
    Logic,
}
#[derive(Debug, Clone)]
pub struct Located<T: PartialEq + Display> {
    pub data: T,
    pub location: Option<[u32; 2]>,
}

impl<T: PartialEq + Display> Located<T> {
    pub fn from_data(data: T) -> Self {
        Self {
            data,
            location: None,
        }
    }

    pub fn locate(mut self, location: Option<[u32; 2]>) -> Self {
        self.location = location;
        self
    }
}

impl<T: PartialEq + Display> PartialEq for Located<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: PartialEq + Display> Display for Located<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.data.fmt(f)
    }
}
#[derive(Debug, PartialEq, Clone)]
pub struct SchemeError {
    pub category: ErrorType,
    pub message: String,
    pub location: Option<[u32; 2]>,
}

impl Error for SchemeError {}

impl Display for SchemeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.category {
            ErrorType::Lexical => write!(f, "Invalid token: {}", self.message),
            ErrorType::Syntax => write!(f, "Syntax error: {}", self.message),
            ErrorType::Logic => write!(f, "error: {}", self.message),
        }
    }
}

macro_rules! invalid_token {
    ($location: expr, $($arg:tt)*) => (
        return Err(SchemeError {category: ErrorType::Lexical, message: format!($($arg)*), location: $location });
    )
}

macro_rules! syntax_error {
    ($location: expr, $($arg:tt)*) => (
        return Err(SchemeError {category: ErrorType::Syntax, message: format!($($arg)*), location: $location})
    )
}

macro_rules! logic_error {
    ($($arg:tt)*) => (
        return Err(SchemeError {category: ErrorType::Logic , message: format!($($arg)*), location: None});
    )
}

macro_rules! logic_error_with_location {
    ($location: expr, $($arg:tt)*) => (
        return Err(SchemeError {category: ErrorType::Logic , message: format!($($arg)*), location: $location});
    )
}
