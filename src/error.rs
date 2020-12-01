use thiserror::Error;

use std::fmt;
use std::{error::Error, fmt::Debug, ops::Deref, ops::DerefMut};

use fmt::Display;

use crate::{interpreter::error::LogicError, parser::error::SyntaxError};

#[derive(Debug, Clone, Copy)]
pub struct Located<T> {
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
impl<T: PartialEq> Located<T> {
    pub fn extract_data(self) -> T {
        self.data
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

impl<T: Display> Display for Located<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.data.fmt(f)
    }
}

impl<T: PartialEq> Deref for Located<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}
impl<T: PartialEq> DerefMut for Located<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}
#[derive(PartialEq, Error, Clone)]
pub enum ErrorData {
    #[error("syntax error: {0}")]
    Syntax(#[from] SyntaxError),
    #[error(transparent)]
    Logic(#[from] LogicError),
    #[error("io error: {0}")]
    IO(String), // std::io::Error does not implement PartialEq and Clone, so use display message directly
}

pub type SchemeError = Located<ErrorData>;
impl From<std::io::Error> for SchemeError {
    fn from(io_error: std::io::Error) -> Self {
        ErrorData::IO(format!("{}", io_error)).no_locate()
    }
}

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
