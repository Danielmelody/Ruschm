use either::Either;
use std::{
    cell::RefCell,
    cell::RefMut,
    cmp::Ordering,
    fmt::{self, Debug, Display, Formatter},
    ops::Deref,
    rc::Rc,
};

use itertools::join;
use num_traits::real::Real;
use smallvec::SmallVec;

use crate::{
    environment::*,
    error::*,
    interpreter::error::LogicError,
    parser::ParameterFormals,
    parser::SchemeProcedure,
    parser::{
        pair::{GenericPair, Pairable},
        Transformer,
    },
};

type Result<T> = std::result::Result<T, SchemeError>;

pub trait RealNumberInternalTrait: Display + Debug + Real + Default + 'static
where
    Self: std::marker::Sized,
{
}

impl<T: Display + Debug + Real + Default + 'static> RealNumberInternalTrait for T {}
#[derive(Debug, Clone, Copy)]
pub enum Number<R: RealNumberInternalTrait> {
    Integer(i32),
    Real(R),
    Rational(i32, i32),
}

impl<R: RealNumberInternalTrait> Display for Number<R> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Number::Integer(n) => write!(f, "{}", n),
            Number::Real(n) => write!(f, "{:?}", n),
            Number::Rational(a, b) => write!(f, "{}/{}", a, b),
        }
    }
}

impl<R: RealNumberInternalTrait> Number<R> {
    pub(crate) fn exact_eqv(&self, other: &Self) -> bool {
        match (self, other) {
            (Number::Integer(a), Number::Integer(b)) => a.eq(&b),
            (Number::Rational(a1, b1), Number::Rational(a2, b2)) => (a1 * b2).eq(&(b1 * a2)),
            (Number::Real(a), Number::Real(b)) => a.eq(&b),
            _ => false,
        }
    }
}

// in the sense of '=', not eq?, eqv?, nor equal?
impl<R: RealNumberInternalTrait> PartialEq for Number<R> {
    fn eq(&self, other: &Number<R>) -> bool {
        match upcast_oprands((*self, *other)) {
            NumberBinaryOperand::Integer(a, b) => a.eq(&b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => (a1 * b2).eq(&(b1 * a2)),
            NumberBinaryOperand::Real(a, b) => a.eq(&b),
        }
    }
}

impl<R: RealNumberInternalTrait> PartialOrd for Number<R> {
    fn partial_cmp(&self, other: &Number<R>) -> Option<Ordering> {
        match upcast_oprands((*self, *other)) {
            NumberBinaryOperand::Integer(a, b) => a.partial_cmp(&b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => (a1 * b2).partial_cmp(&(b1 * a2)),
            NumberBinaryOperand::Real(a, b) => a.partial_cmp(&b),
        }
    }
}

pub(crate) enum NumberBinaryOperand<R: RealNumberInternalTrait> {
    Integer(i32, i32),
    Real(R, R),
    Rational(i32, i32, i32, i32),
}

// Integer => Rational => Real
pub(crate) fn upcast_oprands<R: RealNumberInternalTrait>(
    operand: (Number<R>, Number<R>),
) -> NumberBinaryOperand<R> {
    match operand {
        (Number::Rational(dividend, dividor), Number::Real(b)) => {
            NumberBinaryOperand::Real(R::from(dividend).unwrap() / R::from(dividor).unwrap(), b)
        }
        (Number::Real(a), Number::Rational(dividend, dividor)) => {
            NumberBinaryOperand::Real(a, R::from(dividend).unwrap() / R::from(dividor).unwrap())
        }
        (Number::Integer(a), Number::Real(b)) => NumberBinaryOperand::Real(R::from(a).unwrap(), b),
        (Number::Real(a), Number::Integer(b)) => NumberBinaryOperand::Real(a, R::from(b).unwrap()),
        (Number::Rational(dividend, dividor), Number::Integer(b)) => {
            NumberBinaryOperand::Rational(dividend, dividor, b, 1)
        }
        (Number::Integer(a), Number::Rational(dividend, dividor)) => {
            NumberBinaryOperand::Rational(a, 1, dividend, dividor)
        }
        (Number::Integer(a), Number::Integer(b)) => (NumberBinaryOperand::Integer(a, b)),
        (Number::Real(a), Number::Real(b)) => (NumberBinaryOperand::Real(a, b)),
        (Number::Rational(a1, a2), Number::Rational(b1, b2)) => {
            NumberBinaryOperand::Rational(a1, a2, b1, b2)
        }
    }
}

impl<R: RealNumberInternalTrait> NumberBinaryOperand<R> {
    pub fn lhs(&self) -> Number<R> {
        match self {
            NumberBinaryOperand::Integer(a, _) => Number::Integer(*a),
            NumberBinaryOperand::Real(a, _) => Number::Real(*a),
            NumberBinaryOperand::Rational(a1, a2, _, _) => Number::Rational(*a1, *a2),
        }
    }

    pub fn rhs(&self) -> Number<R> {
        match self {
            NumberBinaryOperand::Integer(_, b) => Number::Integer(*b),
            NumberBinaryOperand::Real(_, b) => Number::Real(*b),
            NumberBinaryOperand::Rational(_, _, b1, b2) => Number::Rational(*b1, *b2),
        }
    }
}

impl<R: RealNumberInternalTrait> std::ops::Add<Number<R>> for Number<R> {
    type Output = Number<R>;
    fn add(self, rhs: Number<R>) -> Number<R> {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Integer(a, b) => Number::Integer(a + b),
            NumberBinaryOperand::Real(a, b) => Number::Real(a + b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => {
                Number::Rational(a1 * b2 + a2 * b1, a2 * b2)
            }
        }
    }
}

impl<R: RealNumberInternalTrait> std::ops::Sub<Number<R>> for Number<R> {
    type Output = Number<R>;
    fn sub(self, rhs: Number<R>) -> Number<R> {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Integer(a, b) => Number::Integer(a - b),
            NumberBinaryOperand::Real(a, b) => Number::Real(a - b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => {
                Number::Rational(a1 * b2 - a2 * b1, a2 * b2)
            }
        }
    }
}

impl<R: RealNumberInternalTrait> std::ops::Mul<Number<R>> for Number<R> {
    type Output = Number<R>;
    fn mul(self, rhs: Number<R>) -> Number<R> {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Integer(a, b) => Number::Integer(a * b),
            NumberBinaryOperand::Real(a, b) => Number::Real(a * b),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => Number::Rational(a1 * b1, a2 * b2),
        }
    }
}

impl<R: RealNumberInternalTrait> std::ops::Div<Number<R>> for Number<R> {
    type Output = Result<Number<R>>;
    fn div(self, rhs: Number<R>) -> Self::Output {
        match upcast_oprands((self, rhs)) {
            NumberBinaryOperand::Integer(a, b) => {
                check_division_by_zero(b)?;
                match a % b {
                    0 => Ok(Number::Integer(a / b)),
                    _ => Ok(Number::Rational(a, b)),
                }
            }
            NumberBinaryOperand::Real(a, b) => Ok(Number::Real(a / b)),
            NumberBinaryOperand::Rational(a1, a2, b1, b2) => {
                check_division_by_zero(b1)?;
                check_division_by_zero(a2)?;
                check_division_by_zero(b2)?;
                Ok(Number::Rational(a1 * b2, a2 * b1))
            }
        }
    }
}

impl<R: RealNumberInternalTrait> Number<R> {
    pub fn abs(self) -> Number<R> {
        match self {
            Number::Integer(num) => Number::Integer(num.abs()),
            Number::Real(num) => Number::Real(num.abs()),
            Number::Rational(a, b) => Number::Rational(a.abs(), b.abs()),
        }
    }

    fn as_real(self) -> R {
        match self {
            Number::Integer(num) => R::from(num).unwrap(),
            Number::Real(num) => num,
            Number::Rational(a, b) => R::from(a).unwrap() / R::from(b).unwrap(),
        }
    }

    pub fn sqrt(self) -> Self {
        Number::Real(self.as_real().sqrt())
    }

    pub fn exp(self) -> Self {
        Number::Real(self.as_real().exp())
    }

    pub fn ln(self) -> Self {
        Number::Real(self.as_real().ln())
    }

    pub fn log(self, base: Self) -> Self {
        Number::Real(self.as_real().log(base.as_real()))
    }

    pub fn sin(self) -> Self {
        Number::Real(self.as_real().sin())
    }

    pub fn cos(self) -> Self {
        Number::Real(self.as_real().cos())
    }

    pub fn tan(self) -> Self {
        Number::Real(self.as_real().tan())
    }

    pub fn asin(self) -> Self {
        Number::Real(self.as_real().asin())
    }

    pub fn acos(self) -> Self {
        Number::Real(self.as_real().acos())
    }

    pub fn atan(self) -> Self {
        Number::Real(self.as_real().atan())
    }

    pub fn atan2(self, x: Self) -> Self {
        Number::Real(self.as_real().atan2(x.as_real()))
    }

    pub fn floor(self) -> Self {
        match self {
            Number::Integer(num) => Number::Integer(num),
            Number::Real(num) => Number::Real(num.floor()),
            Number::Rational(a, b) => Number::Integer({
                let quot = a / b;
                if quot >= 0 || quot * b == a {
                    quot
                } else {
                    quot - 1
                }
            }),
        }
    }

    pub fn ceiling(self) -> Self {
        match self {
            Number::Integer(num) => Number::Integer(num),
            Number::Real(num) => Number::Real(num.ceil()),
            Number::Rational(a, b) => Number::Integer({
                let quot = a / b;
                if quot <= 0 || quot * b == a {
                    quot
                } else {
                    quot + 1
                }
            }),
        }
    }

    pub fn floor_quotient(self, rhs: Self) -> Result<Self> {
        Ok((self / rhs)?.floor())
    }

    pub fn floor_remainder(self, rhs: Self) -> Result<Self> {
        Ok(self - self.floor_quotient(rhs)? * rhs)
    }

    // Return an exact number that is numerically closest to the given number
    pub fn exact(self) -> Result<Self> {
        match self {
            Number::Real(num) => match num.round().to_i32() {
                Some(i) => Ok(Number::Integer(i)),
                None => error!(LogicError::InExactConversion(num.to_string())),
            },
            exact => Ok(exact),
        }
    }
}

#[test]
fn number_sqrt() {
    assert_eq!(Number::<f32>::Integer(4).sqrt(), Number::Integer(2));
    assert_eq!(Number::<f32>::Rational(9, 4).sqrt(), Number::Rational(3, 2));
    assert_eq!(Number::<f32>::Real(9.0).sqrt(), Number::Real(3.0));
}

#[test]
fn number_floor() {
    assert_eq!(Number::<f32>::Integer(5).floor(), Number::Integer(5));
    assert_eq!(Number::<f32>::Rational(28, 3).floor(), Number::Integer(9));
    assert_eq!(Number::<f32>::Rational(-43, 7).floor(), Number::Integer(-7));
    assert_eq!(Number::<f32>::Rational(-15, 5).floor(), Number::Integer(-3));
    assert_eq!(Number::<f32>::Real(3.8).floor(), Number::Real(3.0));
    assert_eq!(Number::<f32>::Real(-5.3).floor(), Number::Real(-6.0));
}
#[test]
fn number_ceiling() {
    assert_eq!(Number::<f32>::Integer(5).ceiling(), Number::Integer(5));
    assert_eq!(
        Number::<f32>::Rational(28, 3).ceiling(),
        Number::Integer(10)
    );
    assert_eq!(
        Number::<f32>::Rational(-43, 7).ceiling(),
        Number::Integer(-6)
    );
    assert_eq!(
        Number::<f32>::Rational(-15, 5).ceiling(),
        Number::Integer(-3)
    );
    assert_eq!(Number::<f32>::Real(3.8).ceiling(), Number::Real(4.0));
    assert_eq!(Number::<f32>::Real(-5.3).ceiling(), Number::Real(-5.0));
}
#[test]
fn number_floor_quotient() {
    assert_eq!(
        Number::<f32>::Integer(5).floor_quotient(Number::Integer(2)),
        Ok(Number::Integer(2))
    );
    assert_eq!(
        Number::<f32>::Integer(-5).floor_quotient(Number::Integer(2)),
        Ok(Number::Integer(-3))
    );
    assert_eq!(
        Number::<f32>::Integer(5).floor_quotient(Number::Integer(-2)),
        Ok(Number::Integer(-3))
    );
    assert_eq!(
        Number::<f32>::Integer(-5).floor_quotient(Number::Integer(-2)),
        Ok(Number::Integer(2))
    );
    assert_eq!(
        Number::<f32>::Rational(25, 2).floor_quotient(Number::Integer(3)),
        Ok(Number::Integer(4))
    );
    assert_eq!(
        Number::<f32>::Rational(-25, 2).floor_quotient(Number::Integer(3)),
        Ok(Number::Integer(-5))
    );
    assert_eq!(
        Number::<f32>::Rational(33, 7).floor_quotient(Number::Rational(5, 2)),
        Ok(Number::Integer(1))
    );
    assert_eq!(
        Number::<f32>::Real(5.0).floor_quotient(Number::Real(2.0)),
        Ok(Number::Real(2.0))
    );
    assert_eq!(
        Number::<f32>::Integer(-5).floor_quotient(Number::Real(2.0)),
        Ok(Number::Real(-3.0))
    );
    assert_eq!(
        Number::<f32>::Real(5.0).floor_quotient(Number::Integer(-2)),
        Ok(Number::Real(-3.0))
    );
    assert_eq!(
        Number::<f32>::Rational(-15, 2).floor_quotient(Number::Real(-3.0)),
        Ok(Number::Real(2.0))
    );
}
#[test]
fn number_floor_remainder() {
    assert_eq!(
        Number::<f32>::Integer(5).floor_remainder(Number::Integer(2)),
        Ok(Number::Integer(1))
    );
    assert_eq!(
        Number::<f32>::Integer(-5).floor_remainder(Number::Integer(2)),
        Ok(Number::Integer(1))
    );
    assert_eq!(
        Number::<f32>::Integer(5).floor_remainder(Number::Integer(-2)),
        Ok(Number::Integer(-1))
    );
    assert_eq!(
        Number::<f32>::Integer(-5).floor_remainder(Number::Integer(-2)),
        Ok(Number::Integer(-1))
    );
    assert_eq!(
        Number::<f32>::Rational(25, 2).floor_remainder(Number::Integer(3)),
        Ok(Number::Rational(1, 2))
    );
    assert_eq!(
        Number::<f32>::Rational(-25, 2).floor_remainder(Number::Integer(3)),
        Ok(Number::Rational(5, 2))
    );
    assert_eq!(
        Number::<f32>::Rational(33, 7).floor_remainder(Number::Rational(5, 2)),
        Ok(Number::Rational(31, 14))
    );
    assert_eq!(
        Number::<f32>::Real(5.0).floor_remainder(Number::Real(2.0)),
        Ok(Number::Real(1.0))
    );
    assert_eq!(
        Number::<f32>::Integer(-5).floor_remainder(Number::Real(2.0)),
        Ok(Number::Real(1.0))
    );
    assert_eq!(
        Number::<f32>::Real(5.0).floor_remainder(Number::Integer(-2)),
        Ok(Number::Real(-1.0))
    );
    assert_eq!(
        Number::<f32>::Rational(-15, 2).floor_remainder(Number::Real(-3.0)),
        Ok(Number::Real(-1.5))
    );
}

#[test]
fn number_exact() {
    assert_eq!(Number::<f32>::Integer(5).exact(), Ok(Number::Integer(5)));
    assert_eq!(Number::<f32>::Real(5.3).exact(), Ok(Number::Integer(5)));
    assert_eq!(Number::<f32>::Real(5.8).exact(), Ok(Number::Integer(6)));
    assert_eq!(Number::<f32>::Real(-5.8).exact(), Ok(Number::Integer(-6)));
    assert_eq!(Number::<f32>::Real(-5.3).exact(), Ok(Number::Integer(-5)));
    assert_eq!(
        Number::<f32>::Real(1e30).exact(),
        error!(LogicError::InExactConversion(1e30.to_string())),
    );
    assert_eq!(
        Number::<f32>::Rational(7, 3).exact(),
        Ok(Number::Rational(7, 3))
    );
}

pub type ArgVec<R> = SmallVec<[Value<R>; 4]>;

#[derive(Clone)]
pub enum BuiltinProcedureBody<R: RealNumberInternalTrait> {
    Pure(fn(ArgVec<R>) -> Result<Value<R>>),
    Impure(Rc<dyn Fn(ArgVec<R>, Rc<Environment<R>>) -> Result<Value<R>>>),
}

impl<R: RealNumberInternalTrait> PartialEq for BuiltinProcedureBody<R> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BuiltinProcedureBody::Pure(fpa), BuiltinProcedureBody::Pure(fpb)) => fpa == fpb,
            (BuiltinProcedureBody::Impure(fpa), BuiltinProcedureBody::Impure(fpb)) => {
                Rc::ptr_eq(fpa, fpb)
            }
            _ => false,
        }
    }
}

impl<R: RealNumberInternalTrait> BuiltinProcedureBody<R> {
    pub fn apply(&self, args: ArgVec<R>, env: &Rc<Environment<R>>) -> Result<Value<R>> {
        match &self {
            Self::Pure(pointer) => pointer(args),
            Self::Impure(pointer) => pointer(args, env.clone()),
        }
    }
}
#[derive(Clone, PartialEq)]
pub struct BuiltinProcedure<R: RealNumberInternalTrait> {
    pub name: String,
    pub parameters: ParameterFormals,
    pub body: BuiltinProcedureBody<R>,
}

impl<R: RealNumberInternalTrait> Display for BuiltinProcedure<R> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "<build-in procedure ({})>", self.name)
    }
}
impl<R: RealNumberInternalTrait> Debug for BuiltinProcedure<R> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

#[derive(Clone)]
pub enum Procedure<R: RealNumberInternalTrait> {
    User(SchemeProcedure, Rc<Environment<R>>),
    Builtin(BuiltinProcedure<R>),
}

impl<R: RealNumberInternalTrait> Debug for Procedure<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::User(p, _) => write!(f, "{:?}", p),
            Self::Builtin(b) => write!(f, "{:?}", b),
        }
    }
}

impl<R: RealNumberInternalTrait> PartialEq for Procedure<R> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::User(a, _), Self::User(b, _)) => a == b,
            (Self::Builtin(a), Self::Builtin(b)) => a == b,
            _ => false,
        }
    }
}

impl ParameterFormals {}

impl<R: RealNumberInternalTrait> Procedure<R> {
    pub fn new_builtin_pure(
        name: String,
        parameters: ParameterFormals,
        function: fn(ArgVec<R>) -> Result<Value<R>>,
    ) -> Self {
        Self::Builtin(BuiltinProcedure {
            name,
            parameters,
            body: BuiltinProcedureBody::Pure(function),
        })
    }
    pub fn new_builtin_impure(
        name: String,
        parameters: ParameterFormals,
        pointer: impl Fn(ArgVec<R>, Rc<Environment<R>>) -> Result<Value<R>> + 'static,
    ) -> Self {
        Self::Builtin(BuiltinProcedure {
            name,
            parameters,
            body: BuiltinProcedureBody::Impure(Rc::new(pointer)),
        })
    }
    pub fn get_parameters(&self) -> &ParameterFormals {
        match &self {
            Procedure::User(user, ..) => &user.0,
            Procedure::Builtin(builtin) => &builtin.parameters,
        }
    }
}

impl<R: RealNumberInternalTrait> Display for Procedure<R> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match &self {
            Procedure::User(procedure, ..) => write!(f, "{}", procedure),
            Procedure::Builtin(fp) => write!(f, "{}", fp),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValueReference<T> {
    Immutable(Rc<T>),
    Mutable(Rc<RefCell<T>>),
}

impl<T: Display> Display for ValueReference<Vec<T>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Immutable(ref vec) => {
                write!(f, "{}", join(vec.iter().map(|v| format!("{}", v)), " "))
            }
            Self::Mutable(ref vec) => write!(
                f,
                "{}",
                join(vec.borrow().iter().map(|v| format!("{}", v)), " ")
            ),
        }
    }
}

impl<T: Display> ValueReference<Vec<T>> {
    pub fn new_immutable(t: Vec<T>) -> Self {
        Self::Immutable(Rc::new(t))
    }
    pub fn new_mutable(t: Vec<T>) -> Self {
        Self::Mutable(Rc::new(RefCell::new(t)))
    }
    pub fn as_ref<'a>(&'a self) -> Box<dyn 'a + Deref<Target = Vec<T>>> {
        match self {
            ValueReference::Immutable(t) => Box::new(t.as_ref()),
            ValueReference::Mutable(t) => Box::new(t.borrow()),
        }
    }
    pub fn as_mut<'a>(&'a self) -> Result<RefMut<'a, Vec<T>>> {
        match self {
            ValueReference::Immutable(_) => error!(LogicError::RequiresMutable(self.to_string())),
            ValueReference::Mutable(t) => Ok(t.borrow_mut()),
        }
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Immutable(a), Self::Immutable(b)) => Rc::ptr_eq(a, b),
            (Self::Mutable(a), Self::Mutable(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
}

#[macro_export]
macro_rules! match_expect_type {
    ($value:expr, $type:pat => $inner: expr, $type_name:expr) => {
        match $value {
            $type => Ok($inner),
            _ => Err(
                ErrorData::Logic(LogicError::TypeMisMatch($value.to_string(), $type_name))
                    .no_locate(),
            ),
        }
    };
}
#[test]
fn macro_match_expect_type() {
    assert_eq!(
        match_expect_type!(
            Value::<f32>::Number(Number::Integer(5)),
            Value::Number(Number::Integer(i)) => i, Type::Integer
        ),
        Ok(5)
    );
    assert_eq!(
        match_expect_type!(
            Value::<f32>::Number(Number::Integer(1)),
            Value::Number(Number::Integer(i)) => i + 3, Type::Integer),
        Ok(4)
    );
    assert_eq!(
        match_expect_type!(
            Value::<f32>::Number(Number::Integer(5)),
            Value::String(s) => s, Type::String
        ),
        error!(LogicError::TypeMisMatch(5.to_string(), Type::String))
    );
}

// TODO: using enum as type when RFC 1450 is stable
#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Number, // Non exhaustive, but ok
    Integer,
    Real,
    Rational,
    Boolean,
    Character,
    String,
    Symbol,
    Procedure,
    Vector,
    Pair,
    EmptyList,
    Void,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value<R: RealNumberInternalTrait> {
    Number(Number<R>),
    Boolean(bool),
    Character(char),
    String(String),
    Symbol(String),
    Procedure(Procedure<R>),
    Vector(ValueReference<Vec<Value<R>>>),
    Pair(Box<Pair<R>>),
    Transformer(Transformer),
    Void,
}

pub type Pair<R> = GenericPair<Value<R>>;

impl<R: RealNumberInternalTrait> Pairable for Value<R> {
    impl_pairable!(Value);
}

impl<R: RealNumberInternalTrait> From<Box<Pair<R>>> for Value<R> {
    fn from(pair: Box<Pair<R>>) -> Self {
        Value::Pair(pair)
    }
}

impl<R: RealNumberInternalTrait> From<Pair<R>> for Value<R> {
    fn from(pair: Pair<R>) -> Self {
        Value::Pair(Box::new(pair))
    }
}

impl<R: RealNumberInternalTrait> From<i32> for Value<R> {
    fn from(integer: i32) -> Self {
        Value::Number(Number::Integer(integer))
    }
}

impl<R: RealNumberInternalTrait> Value<R> {
    pub fn expect_number(self) -> Result<Number<R>> {
        match_expect_type!(self, Value::Number(number) => number, Type::Number)
    }
    pub fn expect_integer(self) -> Result<i32> {
        match_expect_type!(self, Value::Number(Number::Integer(i)) => i, Type::Number)
    }
    pub fn expect_real(self) -> Result<R> {
        match_expect_type!(self, Value::Number(Number::Real(r)) => r, Type::Real)
    }
    pub fn expect_vector(self) -> Result<ValueReference<Vec<Value<R>>>> {
        match_expect_type!(self, Value::Vector(vector) => vector, Type::Vector)
    }
    pub fn expect_list(self) -> Result<Pair<R>> {
        match_expect_type!(self, Value::Pair(list) => *list, Type::Pair)
    }
    pub fn expect_string(self) -> Result<String> {
        match_expect_type!(self, Value::String(string) => string, Type::String)
    }
    pub fn expect_symbol(self) -> Result<String> {
        match_expect_type!(self, Value::Symbol(string) => string, Type::Symbol)
    }
    pub fn expect_procedure(self) -> Result<Procedure<R>> {
        match_expect_type!(self, Value::Procedure(procedure) => procedure, Type::Procedure)
    }
    pub fn expect_boolean(self) -> Result<bool> {
        match_expect_type!(self, Value::Boolean(condition) => condition, Type::Boolean)
    }
}

impl<R: RealNumberInternalTrait> Display for Value<R> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Value::Number(num) => write!(f, "{}", num),
            Value::Symbol(symbol) => write!(f, "{}", symbol),
            Value::Procedure(p) => write!(f, "{}", p),
            Value::Void => write!(f, "Void"),
            Value::Boolean(true) => write!(f, "#t"),
            Value::Boolean(false) => write!(f, "#f"),
            Value::Character(c) => write!(f, "#\\{}", c),
            Value::String(ref s) => write!(f, "{}", s),
            Value::Vector(vecref) => write!(f, "#({})", vecref),
            Value::Pair(list) => write!(f, "{}", list),
            Value::Transformer(transformer) => write!(f, "{}", transformer),
        }
    }
}

// impl FromIterator ValueReference

fn check_division_by_zero(num: i32) -> Result<()> {
    match num {
        0 => error!(LogicError::DivisionByZero),
        _ => Ok(()),
    }
}
