mod environment;
mod error;
mod interpreter;
mod lexer;
mod parser;
mod repl;

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<(), error::Error> {
    Ok(match env::args().skip(1).next() {
        Some(file) => {
            let f = BufReader::new(File::open(file.as_str()).expect("no such file or directory"));
            let it = interpreter::Interpreter::new();
            it.eval(
                f.lines()
                    .flat_map(|line| line.unwrap().chars().collect::<Vec<_>>().into_iter()),
            )?;
        }
        None => repl::run(),
    })
}
