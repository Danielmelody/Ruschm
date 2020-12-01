use either::Either;

use crate::error::Located;

use super::{
    pair::{GenericPair, Pairable},
    Primitive,
};

#[derive(PartialEq, Debug, Clone)]
pub struct UserDefinedTransformer {
    pub ellipsis: Option<String>,
    pub literals: Vec<String>,
    pub rules: Vec<(SyntaxPattern, SyntaxTemplate)>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum SyntaxPatternBody {
    Underscore,
    Ellipsis,
    Pair(Box<GenericPair<SyntaxPattern>>),
    Vector(Vec<SyntaxPattern>),
    Identifier(String),
    Primitive(Primitive),
}

impl Pairable for SyntaxPattern {
    impl_located_pairable!(SyntaxPatternBody);
}

impl From<GenericPair<SyntaxPattern>> for SyntaxPattern {
    fn from(pair: GenericPair<SyntaxPattern>) -> Self {
        SyntaxPattern {
            data: SyntaxPatternBody::Pair(Box::new(pair)),
            location: None, // TODO
        }
    }
}

pub type SyntaxPattern = Located<SyntaxPatternBody>;

#[derive(PartialEq, Debug, Clone)]
pub struct SyntaxTemplateElement(pub SyntaxTemplate, pub bool);

#[derive(PartialEq, Debug, Clone)]
pub enum SyntaxTemplateBody {
    Pair(Box<GenericPair<SyntaxTemplateElement>>),
    Vector(Vec<SyntaxTemplateElement>),
    Identifier(String),
    Primitive(Primitive),
    Ellipsis,
}

pub type SyntaxTemplate = Located<SyntaxTemplateBody>;

impl Pairable for SyntaxTemplateElement {
    fn either_pair_mut(&mut self) -> Either<&mut GenericPair<Self>, &mut Self> {
        match self {
            Self(
                SyntaxTemplate {
                    data: SyntaxTemplateBody::Pair(pair),
                    ..
                },
                _,
            ) => Either::Left(pair.as_mut()),
            other => Either::Right(other),
        }
    }

    fn either_pair_ref(&self) -> Either<&GenericPair<Self>, &Self> {
        match &self.0.data {
            SyntaxTemplateBody::Pair(pair) => Either::Left(pair.as_ref()),
            _ => Either::Right(self),
        }
    }

    fn into_pair(self) -> Either<GenericPair<Self>, Self> {
        match self {
            Self(
                SyntaxTemplate {
                    data: SyntaxTemplateBody::Pair(pair),
                    ..
                },
                _,
            ) => Either::Left(*pair),
            other => Either::Right(other),
        }
    }
}

impl From<GenericPair<SyntaxTemplateElement>> for SyntaxTemplateElement {
    fn from(pair: GenericPair<SyntaxTemplateElement>) -> Self {
        SyntaxTemplateElement(
            SyntaxTemplate {
                data: SyntaxTemplateBody::Pair(Box::new(pair)),
                location: None,
            },
            false,
        )
    }
}
