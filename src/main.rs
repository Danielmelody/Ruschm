mod env;
mod error;
mod interpreter;
mod lexer;
mod parser;
use std::io;
use std::io::{BufRead, Write};

fn eval<'a>(
    it: &'a interpreter::Interpreter<'a>,
    char_stream: impl Iterator<Item = char>,
) -> Result<Option<interpreter::ValueType>, error::Error> {
    let ast: Result<parser::ParseResult, _> = lexer::TokenGenerator::new(char_stream).collect();
    // let parser = parser::Parser::new(tokens?.into_iter());
    it.eval_root_ast(&ast??)
}

fn main() {
    let stdin = io::stdin();
    let it = interpreter::Interpreter::new();
    print!("==> ");
    io::stdout().flush().unwrap();
    // let mut source = String::new();
    for line in stdin.lock().lines() {
        match eval(&it, line.unwrap().chars()) {
            Ok(opt) => {
                if let Some(value) = opt {
                    println!("{}", value);
                }
            }
            Err(e) => eprintln!("{}", e),
        }
        io::stdout().flush().unwrap();
        print!("==> ");
        io::stdout().flush().unwrap();
    }
}
