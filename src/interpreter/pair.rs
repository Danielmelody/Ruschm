use crate::{error::*, values::RealNumberInternalTrait, values::Value};
use std::{fmt::Display, iter::FromIterator};

use super::error::LogicError;

// use super::error::error;

// r7rs 6.4. Pairs and lists

#[derive(Debug, Clone, PartialEq)]
pub struct Pair<R: RealNumberInternalTrait> {
    pub car: Value<R>,
    pub cdr: Value<R>,
}

impl<R: RealNumberInternalTrait> Pair<R> {
    pub fn new(car: Value<R>, cdr: Value<R>) -> Self {
        Self { car, cdr }
    }
}

impl<R: RealNumberInternalTrait> Display for Pair<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} ", self.car)?;
        let mut current_value = &self.cdr;
        loop {
            match current_value {
                Value::Pair(next_pair) => {
                    write!(f, "{}", next_pair.car)?;
                    match next_pair.cdr {
                        Value::EmptyList => (),
                        _ => write!(f, " ")?,
                    }
                    current_value = &next_pair.cdr;
                }
                Value::EmptyList => break,
                other => {
                    write!(f, ". {}", other)?;
                    break;
                }
            };
        }
        write!(f, ")")
    }
}

impl<R: RealNumberInternalTrait> FromIterator<Value<R>> for Value<R> {
    fn from_iter<I: IntoIterator<Item = Value<R>>>(iter: I) -> Self {
        let mut list = Value::EmptyList;
        for i in iter {
            list = Value::Pair(Box::new(Pair { car: i, cdr: list }));
        }
        list
    }
}

pub struct PairIter<R: RealNumberInternalTrait> {
    next: Option<Pair<R>>,
}

impl<R: RealNumberInternalTrait> Iterator for PairIter<R> {
    fn next(&mut self) -> Option<Self::Item> {
        match self.next.take() {
            Some(next) => {
                let current = next.car;
                self.next = match next.cdr {
                    Value::Pair(next) => Some(*next),
                    Value::EmptyList => None,
                    other => return Some(error!(LogicError::InproperList(other.to_string()))),
                };
                Some(Ok(current))
            }
            None => None,
        }
    }

    type Item = Result<Value<R>, SchemeError>;
}

impl<R: RealNumberInternalTrait> IntoIterator for Pair<R> {
    type Item = Result<Value<R>, SchemeError>;

    type IntoIter = PairIter<R>;

    fn into_iter(self) -> Self::IntoIter {
        PairIter { next: Some(self) }
    }
}
