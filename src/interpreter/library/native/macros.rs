macro_rules! function_mapping {
    ($ident:tt, $parameter:expr, $function:expr) => {
        (
            $ident.to_owned(),
            Value::Procedure(Procedure::new_builtin_impure($ident, $parameter, $function)),
        )
    };
}

macro_rules! pure_function_mapping {
    ($ident:tt, $parameter:expr,  $function:expr) => {
        (
            $ident.to_owned(),
            Value::Procedure(Procedure::new_builtin_pure($ident, $parameter, $function)),
        )
    };
}
