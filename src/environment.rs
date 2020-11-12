use crate::interpreter::scheme;
#[cfg(test)]
use crate::interpreter::Interpreter;
use crate::values::RealNumberInternalTrait;
use crate::values::Value;
use crate::{error::*, interpreter::error::LogicError};
use cell::{Ref, RefCell, RefMut, RefVal};
use std::collections::HashMap;
#[cfg(test)]
use std::error::Error;
use std::rc::Rc;

pub type DefinitionIter<'a, R, E> = Box<dyn 'a + Iterator<Item = (&'a String, &'a Value<R, E>)>>;

pub trait IEnvironment<R: RealNumberInternalTrait>: std::fmt::Debug + Clone + PartialEq {
    fn new() -> Self
    where
        Self: Sized;
    fn define(&self, name: String, value: Value<R, Self>)
    where
        Self: Sized;
    fn get(&self, name: &str) -> Option<Ref<Value<R, Self>>>
    where
        Self: Sized;
    fn get_mut(&self, name: &str) -> Option<RefMut<Value<R, Self>>>
    where
        Self: Sized;
    fn set(&self, name: &str, value: Value<R, Self>) -> Result<(), SchemeError>
    where
        Self: Sized;
    fn new_child(parent: Rc<Self>) -> Self
    where
        Self: Sized;

    fn iter_local_definitions<'a, 'b: 'a>(&'b self) -> RefVal<'a, DefinitionIter<'b, R, Self>>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct StandardEnv<R: RealNumberInternalTrait> {
    parent: Option<Rc<StandardEnv<R>>>,
    definitions: RefCell<HashMap<String, Value<R, StandardEnv<R>>>>,
}

impl<R: RealNumberInternalTrait> IEnvironment<R> for StandardEnv<R> {
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
    fn get_mut(&self, name: &str) -> Option<RefMut<Value<R, Self>>> {
        if self.definitions.borrow().contains_key(name) {
            Some(RefMut::map(self.definitions.borrow_mut(), |definitions| {
                definitions.get_mut(name).unwrap()
            }))
        } else {
            match &self.parent {
                Some(parent) => parent.get_mut(name),
                None => None,
            }
        }
    }

    fn set(&self, name: &str, value: Value<R, Self>) -> Result<(), SchemeError> {
        match self.definitions.borrow_mut().get_mut(name) {
            None => match &self.parent {
                None => {
                    return Err(
                        ErrorData::Logic(LogicError::UnboundedSymbol(name.to_string())).no_locate(),
                    );
                }
                Some(parent) => parent.set(name, value)?,
            },
            Some(variable) => *variable = value,
        };
        Ok(())
    }

    fn iter_local_definitions<'a, 'b: 'a>(&'b self) -> RefVal<'a, DefinitionIter<'b, R, Self>> {
        Ref::map_val(
            self.definitions.borrow(),
            |definitions| -> DefinitionIter<'b, R, Self> { Box::new(definitions.iter()) },
        )
    }
}

#[test]
fn iter_envs() -> Result<(), Box<dyn std::error::Error>> {
    let it = Interpreter::<f32, StandardEnv<f32>>::new();
    {
        it.env.define("a".to_string(), Value::Void);
    }
    let env = it.env;
    {
        let mut definitions = env.iter_local_definitions();
        assert_ne!(definitions.find(|(name, _)| *name == "a"), None);
    }

    {
        let mut definitions = env.iter_local_definitions();
        assert_ne!(definitions.find(|(name, _)| *name == "sqrt"), None);
    }

    Ok(())
}
#[test]
fn get_mut() -> Result<(), Box<dyn Error>> {
    use crate::values::Number;
    use std::ops::Deref;
    let env = StandardEnv::<f32>::new();
    env.define("x".to_string(), Value::Number(Number::Integer(1)));
    {
        let value_mut = env.get_mut("x").unwrap();
        let mut i = RefMut::map(value_mut, |value_mut| {
            if let Value::Number(Number::Integer(i)) = value_mut {
                i
            } else {
                unreachable!()
            }
        });
        *i += 1;
    }
    assert_eq!(
        env.get("x").unwrap().deref(),
        &Value::Number(Number::Integer(2))
    );
    Ok(())
}
