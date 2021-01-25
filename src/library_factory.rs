use std::ops::Deref;

use crate::{
    error::{ErrorData, Located, SchemeError, ToLocated},
    interpreter::{error::LogicError, library::LibraryName},
    parser::{Lexer, LibraryDefinition, Parser, Statement},
};

pub enum GenericLibraryFactory<'a, V> {
    Native(LibraryName, Box<dyn Fn() -> Vec<(String, V)> + 'a>),
    AST(Located<LibraryDefinition>),
}

impl<'a, V> GenericLibraryFactory<'a, V> {
    pub fn get_library_name(&self) -> &LibraryName {
        match self {
            GenericLibraryFactory::Native(name, _) => name,
            GenericLibraryFactory::AST(library_definition) => &library_definition.0,
        }
    }
    pub fn from_char_stream(
        expect_library_name: &LibraryName,
        char_stream: impl Iterator<Item = char>,
    ) -> Result<Self, SchemeError> {
        let lexer = Lexer::from_char_stream(char_stream);
        let parser = Parser::from_lexer(lexer);
        for statement in parser {
            if let Statement::LibraryDefinition(library_definition) = statement? {
                if library_definition.0.deref() == expect_library_name {
                    return Ok(Self::AST(library_definition));
                }
            }
        }
        error!(LogicError::LibraryNotFound(expect_library_name.clone()))
    }
}
