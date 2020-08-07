use crate::interpreter::scheme;
use crate::interpreter::ValueType;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone)]
pub struct Environment {
    parent: Option<Rc<RefCell<Environment>>>,
    definitions: HashMap<String, ValueType>,
}

impl Environment {
    pub fn new() -> Self {
        Self {
            parent: None,
            definitions: scheme::base::base_library(),
        }
    }

    pub fn child(parent: Rc<RefCell<Environment>>) -> Self {
        Self {
            parent: Some(parent),
            definitions: HashMap::new(),
        }
    }

    pub fn define(&mut self, name: String, value: ValueType) {
        self.definitions.insert(name, value);
    }

    pub fn get(&self, name: &str) -> Option<ValueType> {
        match self.definitions.get(name) {
            None => match &self.parent {
                None => return None,
                Some(parent) => parent.borrow().get(name),
            },
            Some(value) => Some(value.clone()),
        }
    }
}
