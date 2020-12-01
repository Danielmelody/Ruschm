use crate::parser::LibraryName;
use crate::{environment::*, values::*};
use std::collections::HashMap;
#[derive(Clone, Debug, PartialEq)]
pub struct Library<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    LibraryName,
    HashMap<String, Value<R, E>>,
);
impl<R: RealNumberInternalTrait, E: IEnvironment<R>> Library<R, E> {
    pub fn new(
        library_name: LibraryName,
        definitions: impl IntoIterator<Item = (String, Value<R, E>)>,
    ) -> Self {
        Self(library_name, definitions.into_iter().collect())
    }
    pub fn name(&self) -> &LibraryName {
        &self.0
    }
    pub fn iter_definitions(&self) -> impl Iterator<Item = (&String, &Value<R, E>)> {
        self.1.iter().map(|(s, value)| (s, value))
    }
}
