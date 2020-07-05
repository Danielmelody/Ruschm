use crate::interpreter::ValueType;
use std::collections::HashMap;

pub struct Environment<'a> {
    parent: Option<&'a Environment<'a>>,
    definitions: HashMap<String, ValueType>,
}

impl<'a> Environment<'a> {
    pub fn new() -> Self {
        Self {
            parent: None,
            definitions: HashMap::new(),
        }
    }

    pub fn child(parent: &'a Environment<'a>) -> Self {
        Self {
            parent: Some(parent),
            definitions: HashMap::new(),
        }
    }

    pub fn define(&mut self, name: String, value: ValueType) {
        self.definitions.insert(name, value);
    }

    pub fn get(&self, name: &str) -> Option<&ValueType> {
        match self.definitions.get(name) {
            None => match self.parent {
                None => return None,
                Some(parent) => parent.get(name),
            },
            value => value,
        }
    }
}
