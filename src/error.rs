use thiserror::Error;

use std::fmt;
use std::{error::Error, fmt::Debug};

use fmt::Display;

use crate::{interpreter::error::LogicError, parser::error::SyntaxError};

#[derive(Debug, PartialEq, Clone)]
pub enum ErrorType {
    Lexical,
    Syntax,
    Logic,
}
#[derive(Debug, Clone, Copy)]
pub struct Located<T: PartialEq> {
    pub data: T,
    pub location: Option<[u32; 2]>,
}

pub trait ToLocated {
    fn locate(self, location: Option<[u32; 2]>) -> Located<Self>
    where
        Self: Sized + PartialEq,
    {
        Located::<Self> {
            data: self,
            location,
        }
    }

    fn no_locate(self) -> Located<Self>
    where
        Self: Sized + PartialEq,
    {
        Located::<Self> {
            data: self,
            location: None,
        }
    }
}

impl<T: PartialEq> From<T> for Located<T> {
    fn from(data: T) -> Self {
        Self {
            data,
            location: None,
        }
    }
}

impl<T: PartialEq> PartialEq for Located<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: PartialEq + Display> Display for Located<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.data.fmt(f)
    }
}

#[derive(PartialEq, Error, Clone)]
pub enum ErrorData {
    #[error("syntax error: {0}")]
    Syntax(#[from] SyntaxError),
    #[error(transparent)]
    Logic(#[from] LogicError),
}

pub type SchemeError = Located<ErrorData>;

impl ToLocated for ErrorData {}

impl Debug for ErrorData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self, f)
    }
}

impl Error for SchemeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.data)
    }
}

#[cfg(test)]
pub(crate) fn convert_located<T: PartialEq>(datas: Vec<T>) -> Vec<Located<T>> {
    datas.into_iter().map(|d| Located::from(d)).collect()
}

macro_rules! error {
    ($arg:expr) => {
        Err(ErrorData::from($arg).no_locate());
    };
}

macro_rules! located_error {
    ($arg:expr, $loc:expr) => {
        Err(ErrorData::from($arg).locate($loc));
    };
}
