mod lexer;
mod parser;
mod interpreter;
use std::io;
use std::io::BufRead;

fn main() {
    let stdin = io::stdin();
    // let mut source = String::new();
    for line in stdin.lock().lines() {
        match interpreter::parse(line.unwrap().as_str()) {
            Ok(opt) => {
                if let Some(ast) = opt {
                    match interpreter::eval(ast.as_ref()) {
                        Ok(value) => println!("{}", value),
                        Err(e) => eprintln!("{}", e),
                    }
                }
            }
            Err(e) => eprintln!("{}", e),
        }
    }
}
