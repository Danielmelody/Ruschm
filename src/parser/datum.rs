use crate::error::*;
use either::Either;
use itertools::Itertools;
use std::fmt::Display;

use super::{
    error::SyntaxError,
    pair::{GenericPair, Pairable},
};

#[derive(PartialEq, Debug, Clone)]
pub enum Primitive {
    String(String),
    Character(char),
    Boolean(bool),
    Integer(i32),
    Rational(i32, u32),
    Real(String),
}

impl Display for Primitive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Primitive::String(inner) => write!(f, "{}", inner),
            Primitive::Character(inner) => write!(f, "{}", inner),
            Primitive::Boolean(inner) => write!(f, "{}", if *inner { "#t" } else { "#f" }),
            Primitive::Integer(inner) => write!(f, "{}", inner),
            Primitive::Rational(a, b) => write!(f, "{}/{}", a, b),
            Primitive::Real(inner) => write!(f, "{}", inner),
        }
    }
}

pub type DatumPair = GenericPair<Datum>;

#[derive(PartialEq, Debug, Clone)]
pub enum DatumBody {
    Primitive(Primitive),
    Symbol(String),
    Pair(Box<DatumPair>),
    Vector(Vec<Datum>),
}

impl Pairable for Datum {
    impl_located_pairable!(DatumBody);
}

impl Display for DatumBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatumBody::Primitive(inner) => write!(f, "{}", inner),
            DatumBody::Symbol(inner) => write!(f, "{}", inner),
            DatumBody::Pair(inner) => write!(f, "{}", inner),
            DatumBody::Vector(inner) => {
                write!(f, "#({})", inner.iter().join(" "))
            }
        }
    }
}

impl Datum {
    pub fn expect_list(self) -> Result<DatumPair, SchemeError> {
        match self.data {
            DatumBody::Pair(inner) => Ok(*inner),
            _ => {
                return error!(SyntaxError::ExpectSomething(
                    "list/pair".to_string(),
                    self.to_string()
                ))
            }
        }
    }
}

impl From<DatumPair> for Datum {
    fn from(pair: DatumPair) -> Self {
        Datum {
            data: DatumBody::Pair(Box::new(pair)),
            location: None, // TODO
        }
    }
}

pub type Datum = Located<DatumBody>;

impl ToLocated for DatumBody {}
