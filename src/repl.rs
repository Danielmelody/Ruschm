use crate::interpreter;
use std::io;
use std::io::Write;

use rustyline::error::ReadlineError;
use rustyline::Editor;

fn check_bracket_closed(chars: impl Iterator<Item = char>) -> bool {
    let mut count = 0;
    let mut in_comment = false;
    for c in chars {
        match (c, in_comment) {
            ('(', false) => count = count + 1,
            (')', false) => count = count - 1,
            (';', false) => in_comment = true,
            ('\n', true) => in_comment = false,
            _ => (),
        }
    }
    count <= 0
}

pub fn run() {
    let it = interpreter::Interpreter::<f32>::new();
    let mut rl = Editor::<()>::new();
    io::stdout().flush().unwrap();
    let mut source = String::new();

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    println!("Ruschm Version {}", VERSION);
    loop {
        let readline = match source.is_empty() {
            true => rl.readline("> "),
            false => rl.readline(""),
        };
        match readline {
            Ok(line) => {
                if line.is_empty() {
                    continue;
                }
                source.push_str(line.as_str());
                source.push_str("\n");
                if check_bracket_closed(source.chars()) {
                    match it.eval(source.chars()) {
                        Ok(opt) => {
                            if let Some(value) = opt {
                                match value {
                                    interpreter::ValueType::Void => (),
                                    _ => println!("{}", value),
                                }
                            }
                        }
                        Err(e) => eprintln!("{}", e),
                    }
                    source.clear();
                }
            }
            Err(ReadlineError::Interrupted) => {
                source.clear();
                println!("Interrupted input by ctrl-c, use ctrl-d to exit.");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("exited. have a nice day.");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }
}
