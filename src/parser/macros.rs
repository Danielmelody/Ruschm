use crate::error::Located;

use super::Primitive;

#[derive(PartialEq, Debug, Clone)]
pub struct Transformer {
    pub ellipsis: Option<String>,
    pub literals: Vec<String>,
    pub rules: Vec<(SyntaxPattern, SyntaxTemplate)>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum SyntaxPatternBody {
    Underscore,
    Ellipsis,
    Period,
    List(
        Vec<SyntaxPattern>,
        Vec<SyntaxPattern>,
        Option<Box<SyntaxPattern>>,
    ),
    Vector(
        Vec<SyntaxPattern>,
        /*ellipis follows*/ Vec<SyntaxPattern>,
    ),
    Identifier(String),
    Primitive(Primitive),
}

pub type SyntaxPattern = Located<SyntaxPatternBody>;

#[derive(PartialEq, Debug, Clone)]
pub enum SyntaxTemplateBody {
    List(
        Vec<(SyntaxTemplate, /* with ellipsis */ bool)>,
        Option<Box<SyntaxTemplate>>,
    ),
    Vector(Vec<(SyntaxTemplate, /* with ellipsis */ bool)>),
    Identifier(String),
    Primitive(Primitive),
    Ellipsis,
    Period,
}

pub type SyntaxTemplate = Located<SyntaxTemplateBody>;
