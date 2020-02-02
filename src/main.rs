use std::io::{self, BufRead};
mod lexer;

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        println!("{}", line.unwrap());
    }
    Ok(())
}
