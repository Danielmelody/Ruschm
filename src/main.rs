use ruschm::{
    environment::StandardEnv, error, file::file_char_stream, interpreter::Interpreter, repl,
};
use std::io::Write;
use std::path::Path;
use std::{env, process::exit};
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

fn main() -> Result<(), error::SchemeError> {
    Ok(match env::args().skip(1).next() {
        Some(file) => {
            let mut it = Interpreter::<f32, StandardEnv<f32>>::new();
            let result = it.eval(file_char_stream(Path::new(&file))?);
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
