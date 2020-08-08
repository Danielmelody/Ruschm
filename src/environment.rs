use crate::interpreter::scheme;
use crate::interpreter::RealNumberInternalTrait;
use crate::interpreter::ValueType;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone)]
pub struct Environment<InternalReal: RealNumberInternalTrait> {
    parent: Option<Rc<RefCell<Environment<InternalReal>>>>,
    definitions: HashMap<String, ValueType<InternalReal>>,
}

impl<InternalReal: RealNumberInternalTrait> Environment<InternalReal> {
    pub fn new() -> Self {
        Self {
            parent: None,
            definitions: scheme::base::base_library::<InternalReal>(),
        }
    }

    pub fn child(parent: Rc<RefCell<Environment<InternalReal>>>) -> Self {
        Self {
            parent: Some(parent),
            definitions: HashMap::new(),
        }
    }

    pub fn define(&mut self, name: String, value: ValueType<InternalReal>) {
        self.definitions.insert(name, value);
    }

    pub fn get(&self, name: &str) -> Option<ValueType<InternalReal>> {
        match self.definitions.get(name) {
            None => match &self.parent {
                None => return None,
                Some(parent) => parent.borrow().get(name),
            },
            Some(value) => Some(value.clone()),
        }
    }
}
