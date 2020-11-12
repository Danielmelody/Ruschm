use crate::{environment::IEnvironment, values::RealNumberInternalTrait};

use super::Library;

macro_rules! function_mapping {
    ($ident:tt, $fixed_parameter:expr, $variadic_parameter:expr, $function:expr) => {
        (
            $ident.to_owned(),
            Value::Procedure(Procedure::new_buildin_impure(
                $ident,
                ParameterFormals($fixed_parameter, $variadic_parameter),
                $function,
            )),
        )
    };
}

macro_rules! pure_function_mapping {
    ($ident:tt, $fixed_parameter:expr, $variadic_parameter:expr, $function:expr) => {
        (
            $ident.to_owned(),
            Value::Procedure(Procedure::new_buildin_pure(
                $ident,
                ParameterFormals($fixed_parameter, $variadic_parameter),
                $function,
            )),
        )
    };
}
pub mod base;
pub mod write;

pub fn librarys<R: RealNumberInternalTrait, E: IEnvironment<R>>() -> Vec<Library<R, E>> {
    vec![base::library(), write::library()]
}
