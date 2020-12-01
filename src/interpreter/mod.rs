#[macro_use]
pub mod library;
type Result<T> = std::result::Result<T, SchemeError>;
mod interpreter;

use error::LogicError;
pub use interpreter::*;

use crate::error::SchemeError;
pub mod error;
