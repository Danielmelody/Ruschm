#[cfg(test)]
use crate::interpreter::Interpreter;
use crate::values::Value;
use crate::{error::*, interpreter::error::LogicError};
use cell::{Ref, RefCell, RefMut, RefVal};
use std::collections::HashMap;
#[cfg(test)]
use std::error::Error;
use std::rc::Rc;

pub type DefinitionIter<'a, V> = Box<dyn 'a + Iterator<Item = (&'a String, &'a V)>>;

#[derive(Clone, Debug, PartialEq)]
pub struct LexicalScope<V> {
    parent: Option<Rc<LexicalScope<V>>>,
    definitions: RefCell<HashMap<String, V>>,
}

impl<V> LexicalScope<V> {
    pub fn new() -> Self {
        Self {
            parent: None,
            definitions: RefCell::new(HashMap::new()),
        }
    }
    pub fn new_child(parent: Rc<LexicalScope<V>>) -> Self {
        Self {
            parent: Some(parent),
            definitions: RefCell::new(HashMap::new()),
        }
    }

    pub fn define(&self, name: String, value: V) {
        self.definitions.borrow_mut().insert(name, value);
    }

    pub fn get(&self, name: &str) -> Option<Ref<V>> {
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
    pub fn get_mut(&self, name: &str) -> Option<RefMut<V>> {
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

    pub fn set(&self, name: &str, value: V) -> Result<(), SchemeError> {
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

    pub fn iter_local_definitions<'a, 'b: 'a>(&'b self) -> RefVal<'a, DefinitionIter<'b, V>> {
        Ref::map_val(
            self.definitions.borrow(),
            |definitions| -> DefinitionIter<'b, V> { Box::new(definitions.iter()) },
        )
    }
}

impl<V> Default for LexicalScope<V> {
    fn default() -> Self {
        Self::new()
    }
}

pub type Environment<R> = LexicalScope<Value<R>>;

#[test]
fn iter_envs() -> Result<(), Box<dyn std::error::Error>> {
    let it = Interpreter::<f32>::new_with_stdlib();
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
    let env = Environment::<f32>::new();
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
