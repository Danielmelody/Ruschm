use crate::error::*;
use either::Either;
use std::{fmt::Display, iter::FromIterator, mem};
// r7rs 6.4. Pairs and lists

// Some(T, T) for pair
// Some(T, T <another pair> )) for proper list
// Empty for empty list

#[derive(Debug, Clone, PartialEq)]
pub enum GenericPair<T> {
    Some(T, T),
    Empty,
}

impl<T> Default for GenericPair<T> {
    fn default() -> Self {
        GenericPair::Empty
    }
}

pub trait Pairable: From<GenericPair<Self>> {
    fn either_pair_mut(&mut self) -> Either<&mut GenericPair<Self>, &mut Self>;
    fn either_pair_ref(&self) -> Either<&GenericPair<Self>, &Self>;
    fn into_pair(self) -> Either<GenericPair<Self>, Self>;
}

macro_rules! impl_pairable {
    ($t:ident) => {
        fn either_pair_mut(&mut self) -> Either<&mut GenericPair<Self>, &mut Self> {
            match self {
                $t::Pair(pair) => Either::Left(pair),
                other => Either::Right(other),
            }
        }

        fn either_pair_ref(&self) -> Either<&GenericPair<Self>, &Self> {
            match self {
                $t::Pair(pair) => Either::Left(pair),
                other => Either::Right(other),
            }
        }

        fn into_pair(self) -> Either<GenericPair<Self>, Self> {
            match self {
                $t::Pair(pair) => Either::Left(*pair),
                other => Either::Right(other),
            }
        }
    };
}

macro_rules! impl_located_pairable {
    ($t_body:ident) => {
        fn either_pair_mut(&mut self) -> Either<&mut GenericPair<Self>, &mut Self> {
            match self {
                Self {
                    data: $t_body::Pair(pair),
                    ..
                } => Either::Left(pair.as_mut()),
                other => Either::Right(other),
            }
        }

        fn either_pair_ref(&self) -> Either<&GenericPair<Self>, &Self> {
            match &self.data {
                $t_body::Pair(pair) => Either::Left(pair.as_ref()),
                _ => Either::Right(self),
            }
        }

        fn into_pair(self) -> Either<GenericPair<Self>, Self> {
            match self {
                Self {
                    data: $t_body::Pair(pair),
                    ..
                } => Either::Left(*pair),
                other => Either::Right(other),
            }
        }
    };
}

impl<T: Pairable> GenericPair<T> {
    fn pop(&mut self) -> Option<T> {
        match mem::take(self) {
            GenericPair::Some(car, mut cdr) => match cdr.either_pair_mut() {
                Either::Left(pair) => {
                    mem::swap(self, pair);
                    Some(car)
                }
                Either::Right(_) => None, // improper list dropped here.
            },
            GenericPair::Empty => None,
        }
    }

    pub fn len(&self) -> usize {
        self.iter().count()
    }

    pub fn iter(&self) -> Iter<T> {
        Iter { next: Some(self) }
    }

    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut { next: Some(self) }
    }

    pub fn cons(car: T, cdr: T) -> Self {
        Self::Some(car, cdr)
    }

    pub fn map_ok<Target: Pairable>(
        self,
        f: &mut impl FnMut(T) -> Result<Target, SchemeError>,
    ) -> Result<GenericPair<Target>, SchemeError> {
        Ok(match self {
            GenericPair::Some(car, cdr) => GenericPair::Some(
                match car.into_pair() {
                    Either::Left(pair) => Target::from(pair.map_ok(f)?),
                    Either::Right(value) => f(value)?,
                },
                match cdr.into_pair() {
                    Either::Left(pair) => Target::from(pair.map_ok(f)?),
                    Either::Right(value) => f(value)?,
                },
            ),
            GenericPair::Empty => GenericPair::Empty,
        })
    }

    pub fn map_ok_ref<Target: Pairable>(
        &self,
        f: &mut impl FnMut(&T) -> Result<Target, SchemeError>,
    ) -> Result<GenericPair<Target>, SchemeError> {
        Ok(match self {
            GenericPair::Some(car, cdr) => GenericPair::Some(
                match car.either_pair_ref() {
                    Either::Left(pair) => Target::from(pair.map_ok_ref(f)?),
                    Either::Right(value) => f(value)?,
                },
                match cdr.either_pair_ref() {
                    Either::Left(pair) => Target::from(pair.map_ok_ref(f)?),
                    Either::Right(value) => f(value)?,
                },
            ),
            GenericPair::Empty => GenericPair::Empty,
        })
    }
}

pub struct Iter<'a, T>
where
    T: From<GenericPair<T>> + Pairable,
{
    next: Option<&'a GenericPair<T>>,
}

impl<'a, T: Pairable> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.next {
            Some(GenericPair::Some(car, cdr)) => {
                self.next = match cdr.either_pair_ref() {
                    Either::Left(pair) => Some(pair),
                    Either::Right(_) => None, // improper list cdr droped here
                };
                Some(car)
            }
            None | Some(GenericPair::Empty) => None,
        }
    }
}

pub struct IterMut<'a, T>
where
    T: Pairable + 'a,
{
    next: Option<&'a mut GenericPair<T>>,
}

impl<'a, T: Pairable> Iterator for IterMut<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.next.take() {
            Some(GenericPair::Some(car, cdr)) => {
                self.next = match cdr.either_pair_mut() {
                    Either::Left(pair) => Some(pair),
                    Either::Right(_) => None, // improper list cdr droped here
                };
                Some(car)
            }
            None | Some(GenericPair::Empty) => None,
        }
    }
}

impl<T: Display + Pairable> Display for GenericPair<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        let mut current_value = self;
        loop {
            match current_value {
                GenericPair::Some(car, cdr) => {
                    write!(f, "{}", car)?;
                    match cdr.either_pair_ref() {
                        Either::Left(pair) => {
                            match pair {
                                GenericPair::Some(_, _) => {
                                    write!(f, " ")?;
                                }
                                GenericPair::Empty => (),
                            }
                            current_value = pair;
                        }
                        Either::Right(value) => {
                            write!(f, " . {}", value)?;
                            break;
                        }
                    }
                }
                GenericPair::Empty => break,
            };
        }
        write!(f, ")")
    }
}

// collect as a list, return head node
impl<T: Pairable> FromIterator<T> for GenericPair<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut head = GenericPair::Empty;
        let mut tail = &mut head;
        for element in iter {
            match tail {
                GenericPair::Empty => {
                    head = GenericPair::Some(element, T::from(GenericPair::Empty));
                    tail = &mut head;
                }
                GenericPair::Some(_, cdr) => {
                    *cdr = T::from(GenericPair::Some(element, T::from(GenericPair::Empty)));
                    tail = cdr.either_pair_mut().left().unwrap();
                }
            }
        }
        head
    }
}

impl<T: Pairable> IntoIterator for GenericPair<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self)
    }
}

pub struct IntoIter<T>(GenericPair<T>);

impl<T: Pairable> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}

#[macro_export]
macro_rules! list {
    ($($x:expr),*) => {
        vec![$($x,)*].into_iter().collect::<GenericPair<_>>()
    };
}
