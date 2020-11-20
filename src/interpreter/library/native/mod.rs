macro_rules! function_mapping {
    ($ident:tt, $fixed_parameter:expr, $variadic_parameter:expr, $function:expr) => {
        (
            $ident.to_owned(),
            Value::Procedure(Procedure::new_builtin_impure(
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
            Value::Procedure(Procedure::new_builtin_pure(
                $ident,
                ParameterFormals($fixed_parameter, $variadic_parameter),
                $function,
            )),
        )
    };
}
pub mod base;
pub mod write;
