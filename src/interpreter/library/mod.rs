#[macro_export]
macro_rules! library_name {
    ($($e:expr),+) => {
        LibraryName(vec![$($e.into()),+])
    };
}
pub mod include;
pub mod native;
use itertools::Itertools;

use crate::{environment::*, error::*, file::file_char_stream, values::*};
use std::{collections::HashMap, fmt::Display, iter::FromIterator, path::PathBuf};

// r7rs:
// ⟨library name⟩ is a list whose members are identifiers and
// exact non-negative integers.
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum LibraryNameElement {
    Identifier(String),
    Integer(u32),
}

impl ToLocated for LibraryNameElement {}

impl Display for LibraryNameElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LibraryNameElement::Identifier(s) => write!(f, "{}", s),
            LibraryNameElement::Integer(i) => write!(f, "{}", i),
        }
    }
}

impl From<&str> for LibraryNameElement {
    fn from(s: &str) -> Self {
        Self::Identifier(s.to_string())
    }
}

impl From<u32> for LibraryNameElement {
    fn from(i: u32) -> Self {
        Self::Integer(i)
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct LibraryName(pub Vec<LibraryNameElement>);

impl ToLocated for LibraryName {}

impl FromIterator<LibraryNameElement> for LibraryName {
    fn from_iter<T: IntoIterator<Item = LibraryNameElement>>(iter: T) -> Self {
        Self(iter.into_iter().collect::<Vec<_>>())
    }
}

impl From<Vec<LibraryNameElement>> for LibraryName {
    fn from(value: Vec<LibraryNameElement>) -> Self {
        Self(value)
    }
}
impl From<LibraryNameElement> for LibraryName {
    fn from(value: LibraryNameElement) -> Self {
        Self(vec![value])
    }
}

impl Display for LibraryName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({})", self.0.iter().join(" "))
    }
}
#[test]
fn library_name_display() {
    assert_eq!(format!("{}", library_name!("foo", 0, "bar")), "(foo 0 bar)");
    assert_eq!(format!("{}", library_name!(1, 0, 2)), "(1 0 2)");
}
impl LibraryName {
    pub fn join(self, other: impl Into<Self>) -> Self {
        Self(
            self.0
                .into_iter()
                .chain(other.into().0.into_iter())
                .collect(),
        )
    }
}
#[test]
fn library_name_join() {
    assert_eq!(
        library_name!("a", "b").join(library_name!("c")),
        library_name!("a", "b", "c")
    );
    assert_eq!(
        library_name!("a", "b").join(LibraryNameElement::Integer(0)),
        library_name!("a", "b", 0)
    );
}

impl LibraryName {
    fn path(&self) -> PathBuf {
        self.0
            .iter()
            .map(|element| PathBuf::from(format!("{}", element)))
            .collect()
    }
}

pub trait LibrarySearcher {
    fn search_lib(
        &self,
        library_name: Located<LibraryName>,
    ) -> Option<Box<dyn Iterator<Item = char>>>;
}

pub struct StandardLibrarySearcher {}

impl StandardLibrarySearcher {
    pub fn new() -> Self {
        Self {}
    }
}

impl StandardLibrarySearcher {
    pub fn get_include_library(library_name: &LibraryName) -> Option<&'static str> {
        match include::LIBRARY_STRS.binary_search_by(|(name, _)| name.cmp(library_name)) {
            Ok(i) => Some(include::LIBRARY_STRS[i].1),
            Err(_) => None,
        }
    }
}

impl LibrarySearcher for StandardLibrarySearcher {
    fn search_lib(
        &self,
        library_name: Located<LibraryName>,
    ) -> Option<Box<dyn Iterator<Item = char>>> {
        match Self::get_include_library(&library_name) {
            Some(library_str) => Some(Box::new(library_str.chars())),
            None => match std::env::current_dir() {
                Ok(pwd) => {
                    let path = pwd
                        .join("lib")
                        .join(library_name.extract_data().path())
                        .with_extension("sld");
                    // file exist
                    if path.exists() {
                        // directly unwrap to panic for other io error
                        Some(Box::new(file_char_stream(&path).unwrap()))
                    } else {
                        None
                    }
                }
                Err(_) => None,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Library<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    Located<LibraryName>,
    HashMap<String, Value<R, E>>,
);
impl<R: RealNumberInternalTrait, E: IEnvironment<R>> Library<R, E> {
    pub fn new(
        library_name: Located<LibraryName>,
        definitions: impl IntoIterator<Item = (String, Value<R, E>)>,
    ) -> Self {
        Self(library_name, definitions.into_iter().collect())
    }
    pub fn name(&self) -> &Located<LibraryName> {
        &self.0
    }
    pub fn iter_definitions(&self) -> impl Iterator<Item = (&String, &Value<R, E>)> {
        self.1.iter().map(|(s, value)| (s, value))
    }
}
