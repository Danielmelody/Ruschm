use crate::error::*;
use either::Either;
use std::{fmt::Display, iter::FromIterator, mem};
// r7rs 6.4. Pairs and lists

// Some(T, T) for pair
// Some(T, T <another pair> )) for proper list
// Empty for empty list

#[derive(Debug, Clone, PartialEq, Eq)]
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

    fn from_pair_iter(iter: impl Iterator<Item = PairIterItem<Self>>) -> Self {
        let mut head = GenericPair::Empty;
        let mut tail = &mut head;
        for element in iter {
            match element {
                PairIterItem::Proper(element) => match tail {
                    GenericPair::Empty => {
                        head = GenericPair::Some(element, Self::from(GenericPair::Empty));
                        tail = &mut head;
                    }
                    GenericPair::Some(_, cdr) => {
                        *cdr =
                            Self::from(GenericPair::Some(element, Self::from(GenericPair::Empty)));
                        tail = cdr.either_pair_mut().left().unwrap();
                    }
                },
                PairIterItem::Improper(element) => {
                    match tail {
                        GenericPair::Empty => return element,
                        GenericPair::Some(_, cdr) => {
                            *cdr = element;
                        }
                    }
                    break;
                }
            }
        }
        Self::from(head)
    }
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

#[derive(Debug)]
pub enum PairIterItem<T> {
    Proper(T),
    Improper(T),
}

impl<T: Clone> PairIterItem<T> {
    pub fn get_inside(&self) -> T {
        match self {
            PairIterItem::Proper(item) => item.clone(),
            PairIterItem::Improper(item) => item.clone(),
        }
    }

    pub fn replace_inside<N>(&self, inside: N) -> PairIterItem<N> {
        match self {
            PairIterItem::Proper(_) => PairIterItem::Proper(inside),
            PairIterItem::Improper(_) => PairIterItem::Improper(inside),
        }
    }
}

pub enum PairPopItem<T> {
    Proper(T),
    Improper(T, T),
}

impl<T: Pairable> GenericPair<T> {
    pub fn pop(&mut self) -> Option<PairPopItem<T>> {
        match mem::take(self) {
            GenericPair::Some(car, cdr) => match cdr.into_pair() {
                Either::Left(mut pair) => {
                    mem::swap(self, &mut pair);
                    Some(PairPopItem::Proper(car))
                }
                Either::Right(cdr) => Some(PairPopItem::Improper(car, cdr)),
            },
            GenericPair::Empty => None,
        }
    }

    pub fn pop_proper(&mut self) -> Result<Option<T>, SchemeError> {
        self.pop()
            .map(|item| match item {
                PairPopItem::Proper(t) => Ok(t),
                PairPopItem::Improper(_, _) => todo!(),
            })
            .transpose()
    }

    pub fn last_cdr(&self) -> Option<&T> {
        if let GenericPair::Some(_, cdr) = self {
            match cdr.either_pair_ref() {
                Either::Left(pair) => pair.last_cdr(),
                Either::Right(last) => Some(last),
            }
        } else {
            None
        }
    }

    pub fn into_pair_iter(self) -> IntoPairIter<T> {
        IntoPairIter::from(self)
    }

    pub fn from_pair_iter<I: IntoIterator<Item = PairIterItem<T>>>(
        iter: I,
    ) -> Result<Self, SchemeError> {
        match T::from_pair_iter(iter.into_iter()).into_pair() {
            Either::Left(pair) => Ok(pair),
            Either::Right(_) => todo!(),
        }
    }

    pub fn len(&self) -> usize {
        self.iter().count()
    }

    pub fn is_empty(&self) -> bool {
        self.iter().next().is_none()
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

#[derive(Clone)]
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

        while let GenericPair::Some(car, cdr) = current_value {
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
        write!(f, ")")
    }
}

// collect as a list, return head node
impl<T: Pairable> FromIterator<T> for GenericPair<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_pair_iter(iter.into_iter().map(|item| PairIterItem::Proper(item))).unwrap()
    }
}

impl<T: Pairable> IntoIterator for GenericPair<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(IntoPairIter::from(self))
    }
}

pub struct IntoPairIter<T> {
    pair: GenericPair<T>,
    improper_cdr: Option<T>,
}

impl<T: Pairable> IntoPairIter<T> {
    pub fn into_element_iter(self) -> impl Iterator<Item = T> {
        IntoIter(self)
    }
}

impl<T: Pairable> From<GenericPair<T>> for IntoPairIter<T> {
    fn from(pair: GenericPair<T>) -> Self {
        IntoPairIter {
            pair,
            improper_cdr: None,
        }
    }
}

impl<T: Pairable> Iterator for IntoPairIter<T> {
    type Item = PairIterItem<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.improper_cdr {
            Some(_) => self.improper_cdr.take().map(|i| PairIterItem::Improper(i)),
            None => self.pair.pop().map(|i| match i {
                PairPopItem::Improper(car, cdr) => {
                    self.improper_cdr = Some(cdr);
                    PairIterItem::Proper(car)
                }
                PairPopItem::Proper(t) => PairIterItem::Proper(t),
            }),
        }
    }
}

pub struct IntoIter<T>(IntoPairIter<T>);
impl<T: Pairable> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|pair_item| match pair_item {
            PairIterItem::Proper(item) => item,
            PairIterItem::Improper(item) => item,
        })
    }
}

#[macro_export]
macro_rules! list {
    ($($x:expr),*) => {
        vec![$($x,)*].into_iter().collect::<GenericPair<_>>()
    };
}
