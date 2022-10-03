#![allow(clippy::too_many_arguments)]
use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Debug, Display, Formatter},
};

use super::error::SyntaxError;
use super::{pair::*, Datum, DatumBody, Primitive};
use crate::error::*;
use either::Either;
use itertools::Itertools;

#[derive(PartialEq, Debug, Clone)]
pub struct UserDefinedTransformer {
    pub ellipsis: Option<String>,
    pub literals: HashSet<String>,
    pub rules: Vec<(SyntaxPattern, SyntaxTemplate)>,
}

impl UserDefinedTransformer {
    fn transform(&self, keyword: &str, datum: Datum) -> Result<Datum, SchemeError> {
        for (pattern, template) in &self.rules {
            let mut substitutions = HashMap::new();
            if pattern.match_datum(&datum, 0, &self.literals, &mut substitutions)? {
                let mut substituded = template.substitude(&substitutions)?;
                if substituded.len() != 1 {
                    return located_error!(
                        SyntaxError::TransformOutMultipleDatum,
                        pattern.location
                    );
                }
                return Ok(substituded.pop().unwrap());
            }
        }
        error!(SyntaxError::MacroMissMatch(keyword.to_string(), datum))
    }
}

#[derive(PartialEq, Clone)]
pub enum SyntaxPatternBody {
    Underscore,
    Ellipsis,
    Pair(Box<GenericPair<SyntaxPattern>>),
    Vector(Vec<SyntaxPattern>),
    Identifier(String),
    Primitive(Primitive),
}

impl ToLocated for SyntaxPatternBody {}

impl Display for SyntaxPatternBody {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            SyntaxPatternBody::Underscore => write!(f, "_"),
            SyntaxPatternBody::Ellipsis => write!(f, "..."),
            SyntaxPatternBody::Pair(p) => write!(f, "{}", p),
            SyntaxPatternBody::Vector(v) => write!(f, "#({})", v.iter().join(" ")),
            SyntaxPatternBody::Identifier(i) => write!(f, "{}", i),
            SyntaxPatternBody::Primitive(p) => write!(f, "{}", p),
        }
    }
}

impl Debug for SyntaxPatternBody {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl SyntaxPattern {
    // backtracking pattern matching
    fn match_datum_stream(
        pattern_index: usize,
        datum_index: usize,
        depth: usize,
        patterns: &Vec<SyntaxPattern>,
        datums: &Vec<Datum>,
        pattern_literals: &HashSet<String>,
        substitutions: &mut HashMap<String, (Datum, Vec<Datum>)>,
        multi_matches: Option<SyntaxPattern>,
    ) -> Result<bool, SchemeError> {
        Ok(
            match (patterns.get(pattern_index), datums.get(datum_index)) {
                (None, None) => true,
                (
                    Some(SyntaxPattern {
                        data: SyntaxPatternBody::Ellipsis,
                        ..
                    }),
                    None,
                ) if multi_matches.is_some() => Self::match_datum_stream(
                    pattern_index + 1,
                    datum_index + 1,
                    depth,
                    patterns,
                    datums,
                    pattern_literals,
                    substitutions,
                    multi_matches,
                )?,
                (Some(sub_pattern), Some(sub_datum))
                    if sub_pattern.match_datum(
                        sub_datum,
                        depth + 1,
                        pattern_literals,
                        substitutions,
                    )? =>
                {
                    match &sub_pattern.data {
                        SyntaxPatternBody::Ellipsis => {
                            if let Some(multi_match_pattern) = multi_matches {
                                let mut multi_matches_substitutions = HashMap::new();
                                if multi_match_pattern.match_datum(
                                    sub_datum,
                                    depth + 1,
                                    pattern_literals,
                                    &mut multi_matches_substitutions,
                                )? {
                                    for (var, multi_match) in multi_matches_substitutions {
                                        substitutions.get_mut(&var).unwrap().1.push(multi_match.0);
                                    }
                                }
                                if Self::match_datum_stream(
                                    pattern_index,
                                    datum_index + 1,
                                    depth,
                                    patterns,
                                    datums,
                                    pattern_literals,
                                    substitutions,
                                    Some(multi_match_pattern.clone()),
                                )? {
                                    true
                                } else {
                                    return Self::match_datum_stream(
                                        pattern_index + 1,
                                        datum_index + 1,
                                        depth,
                                        patterns,
                                        datums,
                                        pattern_literals,
                                        substitutions,
                                        Some(multi_match_pattern),
                                    );
                                }
                            } else {
                                return located_error!(
                                    SyntaxError::UnexpectedPattern(sub_pattern.clone()),
                                    sub_pattern.location
                                );
                            }
                        }
                        SyntaxPatternBody::Identifier(var) if pattern_literals.contains(var) => {
                            Self::match_datum_stream(
                                pattern_index + 1,
                                datum_index + 1,
                                depth,
                                patterns,
                                datums,
                                pattern_literals,
                                substitutions,
                                None,
                            )?
                        }
                        _ => Self::match_datum_stream(
                            pattern_index + 1,
                            datum_index + 1,
                            depth,
                            patterns,
                            datums,
                            pattern_literals,
                            substitutions,
                            Some(sub_pattern.clone()),
                        )?,
                    }
                }
                _ => false,
            },
        )
    }

    fn match_datum(
        &self,
        datum: &Datum,
        depth: usize,
        pattern_literals: &HashSet<String>,
        substitutions: &mut HashMap<String, (Datum, Vec<Datum>)>,
    ) -> Result<bool, SchemeError> {
        // println!(
        //     "{:indent$}matching '{}' with {}",
        //     "",
        //     self,
        //     datum,
        //     indent = depth,
        // );
        let result = match (&self.data, &datum.data) {
            (SyntaxPatternBody::Underscore, _) => true,
            (SyntaxPatternBody::Ellipsis, _) => true,
            (SyntaxPatternBody::Pair(pattern_pair), DatumBody::Pair(datum_pair)) => {
                if SyntaxPattern::match_datum_stream(
                    0,
                    0,
                    depth + 1,
                    &pattern_pair.iter().cloned().collect(),
                    &datum_pair.iter().cloned().collect(),
                    pattern_literals,
                    substitutions,
                    None,
                )? {
                    match (pattern_pair.last_cdr(), datum_pair.last_cdr()) {
                        (Some(last_pattern), Some(last_datum)) => last_pattern.match_datum(
                            last_datum,
                            depth + 1,
                            pattern_literals,
                            substitutions,
                        )?,
                        (None, None) => true,
                        _ => false,
                    }
                } else {
                    false
                }
            }

            (SyntaxPatternBody::Vector(sub_patterns), DatumBody::Vector(sub_data)) => {
                SyntaxPattern::match_datum_stream(
                    0,
                    0,
                    depth + 1,
                    sub_patterns,
                    sub_data,
                    pattern_literals,
                    substitutions,
                    None,
                )?
            }
            (SyntaxPatternBody::Identifier(pattern_symbol), datum_body) => {
                if !pattern_literals.contains(pattern_symbol) {
                    substitutions.insert(pattern_symbol.clone(), (datum.clone(), Vec::new()));
                    true
                } else {
                    matches!(&datum_body,
                        &DatumBody::Symbol(datum_symbol) if datum_symbol == pattern_symbol )
                }
            }
            (SyntaxPatternBody::Primitive(_), DatumBody::Primitive(_)) => true,
            _ => false,
        };

        // println!(
        //     "{:indent$}{}",
        //     "",
        //     if result { "succeeded" } else { "failed" },
        //     indent = depth
        // );
        Ok(result)
    }
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

impl Display for SyntaxTemplateElement {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.0, if self.1 { "..." } else { "" })
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum SyntaxTemplateBody {
    Pair(Box<GenericPair<SyntaxTemplateElement>>),
    Vector(Vec<SyntaxTemplateElement>),
    Identifier(String),
    Primitive(Primitive),
    Ellipsis,
}

impl Display for SyntaxTemplateBody {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            SyntaxTemplateBody::Ellipsis => write!(f, "..."),
            SyntaxTemplateBody::Pair(p) => write!(f, "{}", p),
            SyntaxTemplateBody::Vector(v) => write!(f, "#({})", v.iter().join(" ")),
            SyntaxTemplateBody::Identifier(i) => write!(f, "{}", i),
            SyntaxTemplateBody::Primitive(p) => write!(f, "{}", p),
        }
    }
}

pub type SyntaxTemplate = Located<SyntaxTemplateBody>;

impl SyntaxTemplate {
    pub fn substitude(
        &self,
        substitutions: &HashMap<String, (Datum, Vec<Datum>)>,
    ) -> Result<Vec<Datum>, SchemeError> {
        let location = self.location;
        match &self.data {
            SyntaxTemplateBody::Pair(list) => {
                let mut substituted_pair_items = vec![];
                for pair_item in list.clone().into_pair_iter() {
                    match pair_item {
                        PairIterItem::Proper(template_element) => substituted_pair_items.extend(
                            SyntaxTemplate::substitute_template_element(
                                &template_element,
                                substitutions,
                            )?,
                        ),
                        PairIterItem::Improper(SyntaxTemplateElement(last, false)) => {
                            substituted_pair_items.extend(last.substitude(substitutions)?)
                        }
                        _ => {
                            return error!(SyntaxError::UnexpectedDatum(
                                DatumBody::Symbol("...".to_string()).locate(self.location)
                            ))
                        }
                    }
                }
                let substituded_list = substituted_pair_items.into_iter().collect();

                Ok(vec![
                    DatumBody::Pair(Box::new(substituded_list)).locate(location)
                ])
            }
            SyntaxTemplateBody::Vector(vec) => {
                let mut substituted_vec = Vec::new();
                for sub_template_element in vec.iter() {
                    substituted_vec.extend(SyntaxTemplate::substitute_template_element(
                        sub_template_element,
                        substitutions,
                    )?)
                }
                Ok(vec![DatumBody::Vector(substituted_vec).locate(location)])
            }
            SyntaxTemplateBody::Identifier(var) => match substitutions.get(var) {
                Some((single_datum, _)) => Ok(vec![single_datum.clone()]),
                None => Ok(vec![DatumBody::Symbol(var.clone()).locate(location)]),
            },
            SyntaxTemplateBody::Primitive(p) => {
                Ok(vec![DatumBody::Primitive(p.clone()).locate(location)])
            }
            SyntaxTemplateBody::Ellipsis => {
                located_error!(SyntaxError::UnexpectedTemplate(self.clone()), location)
            }
        }
    }

    fn substitude_ellipsis_item(
        template: &SyntaxTemplate,
        substitutions: &HashMap<String, (Datum, Vec<Datum>)>,
        item_index: usize,
    ) -> Result<Option<Datum>, SchemeError> {
        Ok(match &template.data {
            SyntaxTemplateBody::Pair(list) => {
                let mut new_list_elements = Vec::new();
                for pair_item in list.clone().into_pair_iter() {
                    match Self::substitude_ellipsis_item(
                        &pair_item.get_inside().0,
                        substitutions,
                        item_index,
                    )? {
                        Some(sub_datum) => {
                            new_list_elements.push(pair_item.replace_inside(sub_datum))
                        }
                        None => return Ok(None),
                    }
                }
                Some(
                    DatumBody::Pair(Box::new(GenericPair::from_pair_iter(
                        new_list_elements.into_iter(),
                    )?))
                    .locate(template.location),
                )
            }
            SyntaxTemplateBody::Vector(vec) => {
                let mut new_vec = Vec::new();
                for pair_item in vec.iter() {
                    match Self::substitude_ellipsis_item(&pair_item.0, substitutions, item_index)? {
                        Some(sub_datum) => new_vec.push(sub_datum),
                        None => return Ok(None),
                    }
                }
                Some(DatumBody::Vector(new_vec).locate(template.location))
            }
            SyntaxTemplateBody::Identifier(var) => match substitutions.get(var) {
                Some((_, vec)) => {
                    if vec.is_empty() {
                        None
                    } else {
                        vec.get(item_index).cloned()
                    }
                }
                None => Some(DatumBody::Symbol(var.clone()).locate(template.location)),
            },
            SyntaxTemplateBody::Primitive(p) => {
                Some(DatumBody::Primitive(p.clone()).locate(template.location))
            }
            SyntaxTemplateBody::Ellipsis => {
                return located_error!(
                    SyntaxError::UnexpectedTemplate(template.clone()),
                    template.location
                );
            }
        })
    }

    fn substitute_template_element(
        template_element: &SyntaxTemplateElement,
        substitutions: &HashMap<String, (Datum, Vec<Datum>)>,
    ) -> Result<Vec<Datum>, SchemeError> {
        match template_element {
            SyntaxTemplateElement(sub_template, true) => {
                let mut result = sub_template.substitude(substitutions)?;
                let mut suffix_item_index = 0;
                while let Some(item) =
                    Self::substitude_ellipsis_item(sub_template, substitutions, suffix_item_index)?
                {
                    suffix_item_index += 1;
                    result.push(item)
                }
                Ok(result)
            }

            SyntaxTemplateElement(sub_template, false) => sub_template.substitude(substitutions),
        }
    }
}

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

#[derive(Debug, Clone, PartialEq)]
pub enum Transformer {
    Native(fn(Datum) -> Result<Datum, SchemeError>),
    Scheme(UserDefinedTransformer),
}

impl Display for Transformer {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Transformer::Native(_) => write!(f, "<build-in transformer>"),
            Transformer::Scheme(user) => write!(f, "{:?}", user),
        }
    }
}

impl Transformer {
    pub fn transform(&self, keyword: &str, datum: Datum) -> Result<Datum, SchemeError> {
        match self {
            Transformer::Native(f) => f(datum),
            Transformer::Scheme(user) => user.transform(keyword, datum),
        }
    }
}
