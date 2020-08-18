use crate::interpreter::scheme;
use crate::interpreter::RealNumberInternalTrait;
use crate::interpreter::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub trait IEnvironment<R: RealNumberInternalTrait>: std::fmt::Debug + Clone + PartialEq {
    fn new() -> Self
    where
        Self: Sized;
    fn define(&mut self, name: String, value: Value<R, Self>)
    where
        Self: Sized;
    fn get(&self, name: &str) -> Option<Value<R, Self>>
    where
        Self: Sized;
    fn child(parent: Rc<RefCell<Self>>) -> Self
    where
        Self: Sized;
}

#[derive(Clone, Debug, PartialEq)]
pub struct StandardEnv<R: RealNumberInternalTrait> {
    parent: Option<Rc<RefCell<StandardEnv<R>>>>,
    definitions: HashMap<String, Value<R, StandardEnv<R>>>,
}

impl<R: RealNumberInternalTrait> IEnvironment<R> for StandardEnv<R> {
    fn new() -> Self {
        Self {
            parent: None,
            definitions: scheme::base::base_library::<R, StandardEnv<R>>(),
        }
    }
    fn child(parent: Rc<RefCell<StandardEnv<R>>>) -> Self {
        Self {
            parent: Some(parent),
            definitions: HashMap::new(),
        }
    }

    fn define(&mut self, name: String, value: Value<R, Self>) {
        self.definitions.insert(name, value);
    }

    fn get(&self, name: &str) -> Option<Value<R, Self>> {
        match self.definitions.get(name) {
            None => match &self.parent {
                None => return None,
                Some(parent) => parent.borrow().get(name),
            },
            Some(value) => Some(value.clone()),
        }
    }
}
