use std::ops::{Add, AddAssign, Index, Mul, MulAssign, Sub, SubAssign};

pub trait ElementType:
    Clone
    + Default
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
{
}

impl<T> ElementType for T where
    T: Clone
        + Default
        + Add<Output = Self>
        + AddAssign
        + Sub<Output = Self>
        + SubAssign
        + Mul<Output = Self>
        + MulAssign
{
}

#[derive(Clone)]
pub struct Matrix<T: ElementType> {
    shape: (usize, usize),
    values: Box<[T]>,
}

impl<T: ElementType> Matrix<T> {
    pub fn new(elements: Vec<Vec<T>>) -> Self {
        assert!(!elements.is_empty());
        let shape = (elements.len(), elements[0].len());
        elements
            .iter()
            .for_each(|element| assert_eq!(element.len(), shape.1));
        let values = elements
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self { shape, values }
    }

    pub fn row_vector(elements: Vec<T>) -> Self {
        Self {
            shape: (1, elements.len()),
            values: elements.into_boxed_slice(),
        }
    }

    pub fn column_vector(elements: Vec<T>) -> Self {
        Self {
            shape: (elements.len(), 1),
            values: elements.into_boxed_slice(),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn into_reshape(self, new_shape: (usize, usize)) -> Self {
        let (rows, columns) = self.shape;
        assert_eq!(rows * columns, new_shape.0 * new_shape.1);
        Self {
            shape: new_shape,
            values: self.values,
        }
    }

    pub fn reshaped(&self, new_shape: (usize, usize)) -> Self {
        self.clone().into_reshape(new_shape)
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.values.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.values.iter_mut()
    }

    pub fn transpose(&self) -> Self {
        let (rows, columns) = self.shape;
        // Special cases: row and column vectors.
        // These ones can just be reshaped. This is purely an optimization.
        if rows == 1 {
            return self.reshaped((columns, 1));
        } else if columns == 1 {
            return self.reshaped((1, rows));
        }
        Matrix::new(
            (0..columns)
                .map(|column| (0..rows).map(|row| self[(row, column)].clone()).collect())
                .collect(),
        )
    }
}

impl<T: ElementType> IntoIterator for Matrix<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_vec().into_iter()
    }
}

impl<T: ElementType> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(index.0 < self.shape.0);
        assert!(index.1 < self.shape.1);
        &self.values[index.0 * self.shape.1 + index.1]
    }
}

impl<T: ElementType> Mul<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;

    /// The most complicated of the matrix operations.
    /// The basic idea (for those who don't know already) is to construct a matrix made from the dot products of our rows and another matrix's columns.
    /// We go through every row in self, every column in the rhs, multiply corresponding elements and sum them. This gives us the element at the given position in the matrix.
    /// Hopefully the code says that too.
    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        let (my_rows, my_columns) = self.shape;
        let (rhs_rows, rhs_columns) = rhs.shape;
        assert_eq!(
            my_columns, rhs_rows,
            "Can't multiply matrices with shapes {:?} and {:?}",
            self.shape, rhs.shape
        );
        Matrix::new(
            (0..my_rows)
                .map(|my_row| {
                    (0..rhs_columns)
                        .map(|rhs_column| {
                            (0..my_columns)
                                .map(|i| self[(my_row, i)].clone() * rhs[(i, rhs_column)].clone())
                                // Its rather annoying that we can't just do .sum().
                                // The below code is effectively the same but doesn't have the same trait issues.
                                .fold(T::default(), |acc, x| acc + x)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )
    }
}

impl<T: ElementType> MulAssign<T> for Matrix<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.values.iter_mut().for_each(|x| *x *= rhs.clone());
    }
}

impl<T: ElementType> Mul<T> for &Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut result = self.clone();
        result *= rhs;
        result
    }
}

impl<T: ElementType> Add<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(
            self.shape, rhs.shape,
            "Can't add matrices with shapes {:?} and {:?}",
            self.shape, rhs.shape
        );
        Matrix::new(
            self.values
                .iter()
                .zip(rhs.values.iter())
                .map(|(a, b)| a.clone() + b.clone())
                .collect::<Vec<_>>()
                .chunks(self.shape.1)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>(),
        )
    }
}

impl<T: ElementType> Sub<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(
            self.shape, rhs.shape,
            "Can't subtract matrices with shapes {:?} and {:?}",
            self.shape, rhs.shape
        );
        Matrix::new(
            self.values
                .iter()
                .zip(rhs.values.iter())
                .map(|(a, b)| a.clone() - b.clone())
                .collect::<Vec<_>>()
                .chunks(self.shape.1)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>(),
        )
    }
}

pub trait Operation<T: ElementType> {
    fn evaluate(&self, input: Matrix<T>) -> Matrix<T>;
}