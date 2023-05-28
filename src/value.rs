use std::{
    cell::Cell,
    fmt::Debug,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
    sync::atomic::AtomicUsize,
};

#[derive(Clone)]
pub struct Variable {
    id: usize,
    value: Rc<Cell<Option<f64>>>,
}

impl Variable {
    pub fn with_id(id: usize) -> Self {
        Self {
            id,
            value: Rc::new(Cell::new(None)),
        }
    }

    pub fn new() -> Self {
        static mut NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        Self::with_id(unsafe { NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) })
    }

    pub fn get(&self) -> Option<f64> {
        self.value.get()
    }

    pub fn assign(&self, value: f64) {
        self.value.set(Some(value));
    }

    pub fn unset(&self) {
        self.value.set(None);
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn as_value(self) -> Value {
        Value::from(self)
    }
}

impl PartialEq for Variable {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Variable {}

impl Debug for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Variable")
            .field("id", &self.id)
            .field("value", &self.get())
            .finish()
    }
}

#[derive(Clone, Debug)]
enum Operation {
    Add(Vec<Value>),
    Multiply(Vec<Value>),
    Reciprocal(Box<Value>),
    Negate(Box<Value>),

    Variable(Variable),
}

#[derive(Clone)]
pub struct Value {
    /// Will be `None` for a constant value.
    operation: Option<Operation>,
    cached_value: Option<f64>,
}

impl Value {
    pub fn constant(value: f64) -> Self {
        Self {
            operation: None,
            cached_value: Some(value),
        }
    }

    pub fn variable(variable: Variable) -> Self {
        Self {
            operation: Some(Operation::Variable(variable)),
            cached_value: None,
        }
    }

    pub fn evaluate_retaining_cache(&mut self) -> f64 {
        if let Some(value) = self.cached_value {
            return value;
        }

        let value = match self.operation.as_mut().unwrap() {
            Operation::Add(values) => values
                .iter_mut()
                .map(|value| value.evaluate_retaining_cache())
                .sum(),
            Operation::Multiply(values) => values
                .iter_mut()
                .map(|value| value.evaluate_retaining_cache())
                .product(),
            Operation::Reciprocal(value) => 1.0 / value.evaluate_retaining_cache(),
            Operation::Negate(value) => -value.evaluate_retaining_cache(),
            Operation::Variable(variable) => {
                let value = variable.get();
                value.expect("Variable has not been assigned a value")
            }
        };

        self.cached_value = Some(value);
        value
    }

    pub fn evaluate(&mut self) -> f64 {
        let result = self.evaluate_retaining_cache();
        self.clear_cache();
        result
    }

    pub fn clear_cache(&mut self) {
        match self.operation.as_mut() {
            Some(Operation::Add(values)) => {
                for value in values {
                    value.clear_cache();
                }
            }
            Some(Operation::Multiply(values)) => {
                for value in values {
                    value.clear_cache();
                }
            }
            Some(Operation::Reciprocal(value)) => {
                value.clear_cache();
            }
            Some(Operation::Negate(value)) => {
                value.clear_cache();
            }
            Some(Operation::Variable(_)) => {}
            None => {}
        }
        if self.operation.is_some() {
            self.cached_value = None;
        }
    }

    fn reciprocal(mut self) -> Self {
        match self.operation.as_ref() {
            None => Self {
                operation: None,
                cached_value: Some(1.0 / self.evaluate()),
            },
            _ => Self {
                operation: Some(Operation::Reciprocal(Box::new(self))),
                cached_value: None,
            },
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self.operation.as_ref(), other.operation.as_ref()) {
            (None, None) => self.cached_value.unwrap() == other.cached_value.unwrap(),
            (
                Some(Operation::Variable(self_variable)),
                Some(Operation::Variable(other_variable)),
            ) => self_variable == other_variable,
            (Some(Operation::Add(self_values)), Some(Operation::Add(other_values))) => {
                self_values == other_values
            }
            (Some(Operation::Multiply(self_values)), Some(Operation::Multiply(other_values))) => {
                self_values == other_values
            }
            (Some(Operation::Reciprocal(self_value)), Some(Operation::Reciprocal(other_value))) => {
                self_value == other_value
            }
            (Some(Operation::Negate(self_value)), Some(Operation::Negate(other_value))) => {
                self_value == other_value
            }
            // They are not the same type of operation.
            _ => false,
        }
    }
}

impl Eq for Value {}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref operation) = self.operation {
            operation.fmt(f)
        } else {
            self.cached_value
                .expect("Missing value for constant")
                .fmt(f)
        }
    }
}

impl Add for Value {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        match (self.operation.as_ref(), rhs.operation.as_ref()) {
            (Some(Operation::Add(self_values)), Some(Operation::Add(rhs_values))) => Self {
                operation: Some(Operation::Add(
                    [self_values.as_slice(), rhs_values.as_slice()].concat(),
                )),
                cached_value: None,
            },
            (Some(Operation::Add(self_values)), _) => Self {
                operation: Some(Operation::Add([self_values.as_slice(), &[rhs]].concat())),
                cached_value: None,
            },
            (_, Some(Operation::Add(rhs_values))) => Self {
                operation: Some(Operation::Add([&[self], rhs_values.as_slice()].concat())),
                cached_value: None,
            },
            (None, None) => {
                // This means they are both constants, so we can add them up front.
                Self {
                    operation: None,
                    cached_value: Some(self.evaluate() + rhs.evaluate()),
                }
            }
            _ => Self {
                operation: Some(Operation::Add(vec![self, rhs])),
                cached_value: None,
            },
        }
    }
}

impl Sub for Value {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl Neg for Value {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        match self.operation.as_ref() {
            None => Self {
                operation: None,
                cached_value: Some(-self.evaluate()),
            },
            _ => Self {
                operation: Some(Operation::Negate(Box::new(self))),
                cached_value: None,
            },
        }
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(mut self, mut rhs: Self) -> Self::Output {
        match (self.operation.as_ref(), rhs.operation.as_ref()) {
            (Some(Operation::Multiply(self_values)), Some(Operation::Multiply(rhs_values))) => {
                Self {
                    operation: Some(Operation::Multiply(
                        [self_values.as_slice(), rhs_values.as_slice()].concat(),
                    )),
                    cached_value: None,
                }
            }
            (Some(Operation::Multiply(self_values)), _) => Self {
                operation: Some(Operation::Multiply(
                    [self_values.as_slice(), &[rhs]].concat(),
                )),
                cached_value: None,
            },
            (_, Some(Operation::Multiply(rhs_values))) => Self {
                operation: Some(Operation::Multiply(
                    [&[self], rhs_values.as_slice()].concat(),
                )),
                cached_value: None,
            },
            (None, None) => {
                // This means they are both constants, so we can multiply them up front.
                Self {
                    operation: None,
                    cached_value: Some(self.evaluate() * rhs.evaluate()),
                }
            }
            _ => Self {
                operation: Some(Operation::Multiply(vec![self, rhs])),
                cached_value: None,
            },
        }
    }
}

impl Div for Value {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)] // Because we are using multiplication in the division implementation.
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.reciprocal()
    }
}

impl From<Variable> for Value {
    fn from(variable: Variable) -> Self {
        Value::variable(variable)
    }
}

trait Constant: Into<f64> {
    fn into_value(self) -> Value {
        Value::constant(self.into())
    }
}

impl<T: Into<f64>> Constant for T {}

impl<T: Constant> From<T> for Value {
    fn from(value: T) -> Self {
        value.into_value()
    }
}

macro_rules! impl_binary_operator_with_constant {
    ($trait:ident, $method:ident) => {
        impl<T> $trait<T> for Value
        where
            T: Constant,
        {
            type Output = Self;

            fn $method(self, rhs: T) -> Self::Output {
                self.$method(rhs.into_value())
            }
        }
    };
}

impl_binary_operator_with_constant!(Add, add);
impl_binary_operator_with_constant!(Sub, sub);
impl_binary_operator_with_constant!(Mul, mul);
impl_binary_operator_with_constant!(Div, div);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn gets_correct_result() {
        let initial_value = 1.into_value();
        let mut calculation = (initial_value + 2 + 3) * 7 / 2;
        assert_eq!(calculation.evaluate(), 21.0);

        let variable = Variable::new();
        let mut calculation1 = Value::from(variable.clone()) + calculation;
        variable.assign(8.5);
        assert_eq!(calculation1.evaluate(), 29.5);
        variable.assign(12.0);
        assert_eq!(calculation1.evaluate(), 33.0);
    }

    #[test]
    fn simplifies_correctly() {
        let initial_value = 1.into_value();
        let calculation = (initial_value + 2 + 3) * 7 / 2;
        assert_eq!(calculation, 21.into_value());

        let variable = Variable::new();
        let calculation1 = (variable.clone().as_value() + 3 + 7) / variable.clone().as_value();
        // It shouldn't simplify (at least in its current implementation).
        let expected_after_simplification =
            (variable.clone().as_value() + 3 + 7) / variable.clone().as_value();
        assert_eq!(calculation1, expected_after_simplification);
    }
}
