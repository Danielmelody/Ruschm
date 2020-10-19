use crate::error::SchemeError;

type Result<T> = std::result::Result<T, SchemeError>;

mod interpreter;
pub mod pair;
pub mod scheme;

pub use interpreter::*;
