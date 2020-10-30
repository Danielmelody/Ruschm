#[derive(PartialEq, Debug, Clone)]
pub enum Primitive {
    String(String),
    Character(char),
    Boolean(bool),
    Integer(i32),
    Rational(i32, u32),
    Real(String),
}
