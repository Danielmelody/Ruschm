#[cfg(test)]
use crate::error::SchemeError;
use crate::interpreter::scheme;
#[cfg(test)]
use crate::interpreter::Interpreter;
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
    fn new_child(parent: Rc<RefCell<Self>>) -> Self
    where
        Self: Sized;

    fn iter_local_definitions<'a>(
        &'a self,
    ) -> Box<dyn 'a + Iterator<Item = (&'a String, &'a Value<R, Self>)>>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct StandardEnv<R: RealNumberInternalTrait> {
    parent: Option<Rc<RefCell<StandardEnv<R>>>>,
    definitions: HashMap<String, Value<R, StandardEnv<R>>>,
}

impl<R: RealNumberInternalTrait> IEnvironment<R> for StandardEnv<R> {
    // type DefinitionCollection = std::collections::hash_map::Iter<'a， String, Value<R, StandardEnv<R>>>;

    fn new() -> Self {
        Self {
            parent: None,
            definitions: scheme::base::base_library::<R, StandardEnv<R>>(),
        }
    }
    fn new_child(parent: Rc<RefCell<StandardEnv<R>>>) -> Self {
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

    fn iter_local_definitions<'a>(
        &'a self,
    ) -> Box<dyn 'a + Iterator<Item = (&'a String, &'a Value<R, Self>)>> {
        Box::new(self.definitions.iter())
    }
}

#[test]
fn iter_envs() -> Result<(), SchemeError> {
    let it = Interpreter::<f32, StandardEnv<f32>>::new();
    {
        let mut mut_env = it.env.borrow_mut();
        mut_env.define("a".to_string(), Value::Void);
    }
    let env = it.env.borrow();
    {
        let mut definitions = env.iter_local_definitions();
        assert_ne!(definitions.find(|(name, _)| name.as_str() == "a"), None);
    }

    {
        let mut definitions = env.iter_local_definitions();
        assert_ne!(definitions.find(|(name, _)| name.as_str() == "sqrt"), None);
    }

    Ok(())
}
