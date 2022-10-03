#[macro_use]
pub mod library;
type Result<T> = std::result::Result<T, SchemeError>;
#[allow(clippy::module_inception)]
mod interpreter;

use error::LogicError;
pub use interpreter::*;

use crate::error::SchemeError;
pub mod error;
