use ruschm::{environment::StandardEnv, error, interpreter::Interpreter, repl};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::{env, process::exit};
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

fn main() -> Result<(), error::SchemeError> {
    Ok(match env::args().skip(1).next() {
        Some(file) => {
            let f = BufReader::new(File::open(file.as_str()).expect("no such file or directory"));
            let it = Interpreter::<f32, StandardEnv<f32>>::new();
            let result = it.eval(f.lines().flat_map(|line| {
                line.unwrap()
                    .chars()
                    .chain(std::iter::once('\n'))
                    .collect::<Vec<_>>()
                    .into_iter()
            }));
            match result {
                Ok(_) => (),
                Err(e) => {
                    let mut stderr = StandardStream::stderr(ColorChoice::Always);
                    stderr
                        .set_color(&ColorSpec::new().set_fg(Some(Color::Red)))
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
    })
}
