use std::{fmt::Display, iter::FromIterator};

use super::{RealNumberInternalTrait, Value};
use crate::environment::IEnvironment;

// r7rs 6.4. Pairs and lists

#[derive(Debug, Clone, PartialEq)]
pub struct Pair<R: RealNumberInternalTrait, E: IEnvironment<R>> {
    pub car: Value<R, E>,
    pub cdr: Value<R, E>,
}

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> Pair<R, E> {
    pub fn new(car: Value<R, E>, cdr: Value<R, E>) -> Self {
        Self { car, cdr }
    }
}

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> Display for Pair<R, E> {
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

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> FromIterator<Value<R, E>> for Value<R, E> {
    fn from_iter<I: IntoIterator<Item = Value<R, E>>>(iter: I) -> Self {
        let mut list = Value::EmptyList;
        for i in iter {
            list = Value::Pair(Box::new(Pair { car: i, cdr: list }));
        }
        list
    }
}
