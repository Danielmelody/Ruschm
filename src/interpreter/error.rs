use thiserror::Error;

use crate::{
    parser::error::SyntaxError, parser::Expression, parser::ParameterFormals, values::Type,
};

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
    #[error("inproper list")]
    InproperList(String),
    #[error("expect a non-negative length")]
    NegativeLength,
    #[error("vector index out of bound")]
    VectorIndexOutOfBounds,
    #[error("expect {}{} arguments, got {}. parameter list is: {}",
    if (.0).1.is_some() { "at least " } else { "" },
    (.0).0.len(),
    .1,
    .0,)
    ]
    ArgumentMissMatch(ParameterFormals, usize),
    #[error("requires {0} to be mutable")]
    RequiresMutable(String),
    #[error(transparent)]
    MetaCircularSyntax(#[from] SyntaxError),
    #[error("{0}")]
    Extension(String),
}
