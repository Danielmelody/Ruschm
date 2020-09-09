use crate::error::*;
use crate::interpreter::scheme;
#[cfg(test)]
use crate::interpreter::Interpreter;
use crate::interpreter::RealNumberInternalTrait;
use crate::interpreter::Value;
use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

pub trait IEnvironment<R: RealNumberInternalTrait>: std::fmt::Debug + Clone + PartialEq {
    type DefinitionCollection: IntoIterator;

    fn new() -> Self
    where
        Self: Sized;
    fn define(&self, name: String, value: Value<R, Self>)
    where
        Self: Sized;
    fn get(&self, name: &str) -> Option<Ref<Value<R, Self>>>
    where
        Self: Sized;
    fn set(&self, name: &str, value: Value<R, Self>) -> Result<(), SchemeError>
    where
        Self: Sized;
    fn new_child(parent: Rc<Self>) -> Self
    where
        Self: Sized;

    fn local_definitions(&self) -> Ref<Self::DefinitionCollection>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct StandardEnv<R: RealNumberInternalTrait> {
    parent: Option<Rc<StandardEnv<R>>>,
    definitions: RefCell<HashMap<String, Value<R, StandardEnv<R>>>>,
}

impl<R: RealNumberInternalTrait> IEnvironment<R> for StandardEnv<R> {
    type DefinitionCollection = HashMap<String, Value<R, StandardEnv<R>>>;

    fn new() -> Self {
        Self {
            parent: None,
            definitions: RefCell::new(scheme::base::base_library::<R, StandardEnv<R>>()),
        }
    }
    fn new_child(parent: Rc<StandardEnv<R>>) -> Self {
        Self {
            parent: Some(parent),
            definitions: RefCell::new(HashMap::new()),
        }
    }

    fn define(&self, name: String, value: Value<R, Self>) {
        self.definitions.borrow_mut().insert(name, value);
    }

    fn get(&self, name: &str) -> Option<Ref<Value<R, Self>>> {
        if self.definitions.borrow().contains_key(name) {
            Some(Ref::map(self.definitions.borrow(), |definitions| {
                definitions.get(name).unwrap()
            }))
        } else {
            match &self.parent {
                Some(parent) => parent.get(name),
                None => None,
            }
        }
    }

    fn set(&self, name: &str, value: Value<R, Self>) -> Result<(), SchemeError> {
        match self.definitions.borrow_mut().get_mut(name) {
            None => match &self.parent {
                None => logic_error!("unbound variable {}", name),
                Some(parent) => parent.set(name, value)?,
            },
            Some(variable) => *variable = value,
        };
        Ok(())
    }

    fn local_definitions<'a>(&'a self) -> Ref<Self::DefinitionCollection> {
        self.definitions.borrow()
    }
}

#[test]
fn iter_envs() -> Result<(), SchemeError> {
    let it = Interpreter::<f32, StandardEnv<f32>>::new();
    {
        it.env.define("a".to_string(), Value::Void);
    }
    let env = it.env;
    {
        let definitions = env.local_definitions();
        assert_ne!(
            definitions.iter().find(|(name, _)| name.as_str() == "a"),
            None
        );
    }

    {
        let definitions = env.local_definitions();
        assert_ne!(
            definitions.iter().find(|(name, _)| name.as_str() == "sqrt"),
            None
        );
    }

    Ok(())
}
