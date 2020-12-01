use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

pub fn file_char_stream(path: &Path) -> Result<impl Iterator<Item = char>, std::io::Error> {
    let f = BufReader::new(File::open(path)?);
    Ok(f.lines().flat_map(|line| {
        line.unwrap()
            .chars()
            .chain(std::iter::once('\n'))
            .collect::<Vec<_>>()
            .into_iter()
    }))
}
