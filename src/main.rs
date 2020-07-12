mod error;
mod interpreter;
mod lexer;
mod parser;
mod repl;

use std::env;

fn main() {
    match env::args().skip(1).next() {
        Some(_) => (), // TODO
        None => repl::run(),
    }
}
