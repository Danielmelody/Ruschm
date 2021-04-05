use super::{Datum, DefinitionBody, SyntaxPattern, SyntaxTemplate, TokenData};
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Clone)]
pub enum SyntaxError {
    #[error("expect {}, got {}", .0, match .1 {Some(t) => t.to_string(), None => "end of input".to_string()})]
    TokenMisMatch(TokenData, Option<TokenData>),
    #[error("unexpected character {0}")]
    UnexpectedCharacter(char),
    #[error("unexpected token {0}")]
    UnexpectedToken(TokenData),
    #[error("unexpected datum {0}")]
    UnexpectedDatum(Datum),
    #[error("unexpected pattern {0}")]
    UnexpectedPattern(SyntaxPattern),
    #[error("unexpected template {0}")]
    UnexpectedTemplate(SyntaxTemplate),
    #[error("unexpect end of input")]
    UnexpectedEnd,
    #[error("unrecognized token")]
    UnrecognizedToken,
    #[error("unknown escape character")]
    UnknownEscape(char),
    #[error("unmactched parentheses!")]
    UnmatchedParentheses,
    #[error("try to define non-symbol {0}")]
    DefineNonSymbol(Datum),
    #[error("invalid definition {0}")]
    InvalidDefinition(Datum),
    #[error("no expression found in function body")]
    LambdaBodyNoExpression,
    #[error("expect a {0}, got {1}")]
    ExpectSomething(String, String),
    #[error("illegal sub import")]
    IllegalSubImport,
    #[error("invalid identifier {0}")]
    InvalidIdentifier(String),
    #[error("imcomplete quoted identifier {0}")]
    ImcompleteQuotedIdent(String),
    #[error("rational denominator should not be 0!")]
    RationalDivideByZero,
    #[error("empty procedure call")]
    EmptyCall,
    #[error("illegal pattern")]
    IllegalPattern,
    #[error("illegal definition")]
    IllegalDefinition,
    #[error("invalid context for definition {0:?}")]
    InvalidDefinitionContext(DefinitionBody),
    #[error("{1} is not matched to any pattern of macro {0}")]
    MacroMissMatch(String, Datum),
    #[error("keyword should be {0} instead of {1}")]
    MacroKeywordMissMatch(String, String),
    #[error("multiple expression should be packed by (begin ...)")]
    TransformOutMultipleDatum,
    #[error("{0}")]
    Extension(String),
}
