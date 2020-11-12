use crate::{
    environment::IEnvironment,
    interpreter::library::{Library, LibraryName},
    interpreter::Result,
    parser::ParameterFormals,
    values::{Procedure, RealNumberInternalTrait, Value},
};

fn display<R: RealNumberInternalTrait, E: IEnvironment<R>>(
    arguments: impl IntoIterator<Item = Value<R, E>>,
) -> Result<Value<R, E>> {
    print!("{}", arguments.into_iter().next().unwrap());
    Ok(Value::Void)
}

pub fn library<'a, R: RealNumberInternalTrait, E: IEnvironment<R>>() -> Library<R, E> {
    Library(
        library_name!("ruschm", "write").into(),
        vec![pure_function_mapping!(
            "display",
            vec!["value".to_string()],
            None,
            display
        )]
        .into_iter()
        .collect(),
    )
}
