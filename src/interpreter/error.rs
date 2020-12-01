use thiserror::Error;

use crate::{
    parser::error::SyntaxError, parser::Expression, parser::ParameterFormals, values::Type,
};

use crate::parser::LibraryName;

#[derive(Error, Debug, PartialEq, Clone)]
pub enum LogicError {
    #[error("unbound symbol {0}")]
    UnboundedSymbol(String),
    #[error("{0} is not {1:?}")]
    TypeMisMatch(/* value string */ String, Type),
    #[error("unexpect statement {0:?}")]
    UnexpectedExpression(Expression),
    #[error("division by exact zero")]
    DivisionByZero,
    #[error("{0} cannot be converted to an exact number")]
    InExactConversion(String),
    #[error("expect a proper list, encounter inproper list {0}")]
    InproperList(String),
    #[error("expect a non-negative length")]
    NegativeLength,
    #[error("vector index out of bound")]
    VectorIndexOutOfBounds,
    #[error("expect parameters {0}, got arguments {1}")]
    ArgumentMissMatch(ParameterFormals, String),
    #[error("requires {0} to be mutable")]
    RequiresMutable(String),
    #[error(transparent)]
    MetaCircularSyntax(#[from] SyntaxError),
    #[error("{0}")]
    Extension(String),
    #[error("library {0} not found")]
    LibraryNotFound(LibraryName),
    #[error("detect import cyclic while importing library {0}")]
    LibraryImportCyclic(LibraryName),
}
