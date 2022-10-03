use crate::parser::LibraryName;
use crate::values::*;
use std::collections::HashMap;
#[derive(Clone, Debug, PartialEq)]
pub struct Library<R: RealNumberInternalTrait>(LibraryName, HashMap<String, Value<R>>);
impl<R: RealNumberInternalTrait> Library<R> {
    pub fn new(
        library_name: LibraryName,
        definitions: impl IntoIterator<Item = (String, Value<R>)>,
    ) -> Self {
        Self(library_name, definitions.into_iter().collect())
    }
    pub fn name(&self) -> &LibraryName {
        &self.0
    }
    pub fn iter_definitions(&self) -> impl Iterator<Item = (&String, &Value<R>)> {
        self.1.iter().map(|(s, value)| (s, value))
    }
}

#[macro_export]
macro_rules! import_library_direct {
    ($($e:expr),+) => {
        ImportSetBody::Direct(library_name![$($e),+].into()).no_locate()
    };
}


pub mod native;
