use ruschm::{error, interpreter::Interpreter, repl};

use std::{env, process::exit};
use std::{io::Write, path::PathBuf};
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

fn main() -> Result<(), error::SchemeError> {
    match env::args().nth(1) {
        Some(file) => {
            let mut it = Interpreter::<f32>::default();
            let result = it.eval_file(PathBuf::from(file.clone()));
            match result {
                Ok(_) => (),
                Err(e) => {
                    let mut stderr = StandardStream::stderr(ColorChoice::Always);
                    stderr
                        .set_color(ColorSpec::new().set_fg(Some(Color::Red)))
                        .unwrap();
                    write!(&mut stderr, "{}", file).unwrap();
                    if let Some(location) = e.location {
                        write!(&mut stderr, ":{}:{} ", location[0], location[1]).unwrap();
                    };
                    writeln!(&mut stderr, " {}", e).unwrap();
                    exit(-1);
                }
            }
        }
        None => repl::run(),
    };
    Ok(())
}
