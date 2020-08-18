use crate::interpreter::scheme;
use crate::interpreter::RealNumberInternalTrait;
use crate::interpreter::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub trait IEnvironment<R: RealNumberInternalTrait> {
    fn new() -> Self;
    fn define(&mut self, name: String, value: Value<R>);
    fn get(&self, name: &str) -> Option<Value<R>>;
    fn child(parent: Rc<RefCell<Self>>) -> Self;
}

#[derive(Clone)]
pub struct StandardEnv<InternalReal: RealNumberInternalTrait> {
    parent: Option<Rc<RefCell<StandardEnv<InternalReal>>>>,
    definitions: HashMap<String, Value<InternalReal>>,
}

impl<InternalReal: RealNumberInternalTrait> IEnvironment<InternalReal>
    for StandardEnv<InternalReal>
{
    fn new() -> Self {
        Self {
            parent: None,
            definitions: scheme::base::base_library::<InternalReal>(),
        }
    }
    fn child(parent: Rc<RefCell<StandardEnv<InternalReal>>>) -> Self {
        Self {
            parent: Some(parent),
            definitions: HashMap::new(),
        }
    }

    fn define(&mut self, name: String, value: Value<InternalReal>) {
        self.definitions.insert(name, value);
    }

    fn get(&self, name: &str) -> Option<Value<InternalReal>> {
        match self.definitions.get(name) {
            None => match &self.parent {
                None => return None,
                Some(parent) => parent.borrow().get(name),
            },
            Some(value) => Some(value.clone()),
        }
    }
}
