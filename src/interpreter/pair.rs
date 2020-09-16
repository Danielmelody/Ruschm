use std::{fmt::Display, iter::FromIterator};

use itertools::join;

use super::{RealNumberInternalTrait, Value};
use crate::{environment::IEnvironment, error::*, parser::Expression, parser::ExpressionBody};

#[derive(Debug, Clone, PartialEq)]
pub enum List<R: RealNumberInternalTrait, E: IEnvironment<R>> {
    Empty,
    Some(Box<Pair<R, E>>),
}

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> Display for List<R, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let List::Some(pair) = self {
            if pair.car
                == Value::UnEvaluated(Expression::from_data(ExpressionBody::Identifier(
                    "quote".to_string(),
                )))
            {
                return write!(f, "'{}", pair.cdr);
            };
        };
        write!(
            f,
            "({})",
            join(
                self.iter().map(|item| match item {
                    ListItem::Proper(v) => format!("{}", v),
                    ListItem::ImProper(pair) => format!("{}", pair),
                }),
                " "
            )
        )
    }
}

// r7rs 6.4. Pairs and lists
pub struct Iter<'a, R: RealNumberInternalTrait, E: IEnvironment<R>> {
    list: &'a List<R, E>,
    encounter_improper: bool,
}

pub enum ListItem<'a, R: RealNumberInternalTrait, E: IEnvironment<R>> {
    Proper(&'a Value<R, E>),
    ImProper(&'a Pair<R, E>),
}

impl<'a, R: RealNumberInternalTrait, E: IEnvironment<R>> Iterator for Iter<'a, R, E> {
    type Item = ListItem<'a, R, E>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.encounter_improper {
            return None;
        }
        match self.list {
            List::Some(head) => match &head.cdr {
                Value::List(list) => {
                    let current = &head.car;
                    self.list = &list;
                    Some(ListItem::Proper(current))
                }
                _ => {
                    self.encounter_improper = true;
                    Some(ListItem::ImProper(head.as_ref()))
                }
            },
            List::Empty => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pair<R: RealNumberInternalTrait, E: IEnvironment<R>> {
    pub car: Value<R, E>,
    pub cdr: Value<R, E>,
}

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> Display for Pair<R, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} . {})", self.car, self.cdr)
    }
}

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> List<R, E> {
    pub fn new() -> Self {
        Self::Empty
    }

    pub fn iter<'a>(&'a self) -> Iter<'a, R, E> {
        Iter {
            list: self,
            encounter_improper: false,
        }
    }

    pub fn car<'a>(&'a self) -> Result<&'a Value<R, E>, SchemeError> {
        match self {
            List::Empty => logic_error!("attempt to car an empty list"),
            List::Some(head) => Ok(&head.car),
        }
    }

    pub fn cdr<'a>(&'a self) -> Result<&'a Value<R, E>, SchemeError> {
        match self {
            List::Empty => logic_error!("attempt to cdr an empty list"),
            List::Some(head) => Ok(&head.cdr),
        }
    }
}

impl<R: RealNumberInternalTrait, E: IEnvironment<R>> FromIterator<Value<R, E>> for List<R, E> {
    fn from_iter<I: IntoIterator<Item = Value<R, E>>>(iter: I) -> Self {
        let mut c = List::new();

        for i in iter {
            c = List::Some(Box::new(Pair {
                car: i,
                cdr: Value::List(c),
            }));
        }

        c
    }
}
