use crate::{
    environment::IEnvironment,
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

pub fn library_map<'a, R: RealNumberInternalTrait, E: IEnvironment<R>>(
) -> Vec<(String, Value<R, E>)> {
    vec![pure_function_mapping!(
        "display",
        vec!["value".to_string()],
        None,
        display
    )]
}
