mod error;
mod interpreter;
mod lexer;
mod parser;
use std::io;
use std::io::BufRead;

fn eval(
    char_stream: impl Iterator<Item = char>,
) -> Result<Option<interpreter::ValueType>, error::Error> {
    let ast: Result<parser::ParseResult, _> = lexer::TokenGenerator::new(char_stream).collect();
    // let parser = parser::Parser::new(tokens?.into_iter());
    Ok(match ast?? {
        Some(expression) => Some(interpreter::eval_ast(&expression)?),
        None => None,
    })
}

fn main() {
    let stdin = io::stdin();
    // let mut source = String::new();
    for line in stdin.lock().lines() {
        match eval(line.unwrap().as_str().chars()) {
            Ok(opt) => {
                if let Some(value) = opt {
                    println!("{}", value);
                }
            }
            Err(e) => eprintln!("{}", e),
        }
    }
}
