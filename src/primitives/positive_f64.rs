// SPDX-License-Identifier: CC0-1.0

//! Positive floats ("branch probabilities" for policies)
use core::iter::FusedIterator;
use core::num::NonZeroU32;
use core::{cmp, hash, ops};

use crate::Threshold;

/// A positive floating-point number.
///
/// This type guarantees that the contained value is positive: it will never
/// hold 0.0, a negative number, `-inf` or `NaN`. (Positive infinity *is* a
/// permissible value.) This guarantee makes it safe to implement [`Eq`], even
/// though the underlying [`PartialEq`] implementation passes through to `f64`.
///
/// To uphold the guarantee, arithmetic on this type saturates below at
/// [`PositiveF64::MIN_POSITIVE`]: any operation whose result would underflow
/// to a subnormal number or to 0.0 instead yields [`PositiveF64::MIN_POSITIVE`].
/// This means that once you obtain [`PositiveF64::MIN_POSITIVE`], dividing it
/// further (or multiplying it by values less than one) is a no-op.
///
/// Division involving infinity is also defined so as to avoid `NaN` and 0.0:
/// `inf / inf` is defined to be 1.0, and `<finite> / inf` underflows to 0.0
/// and therefore yields [`PositiveF64::MIN_POSITIVE`].
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct PositiveF64(f64);

impl PositiveF64 {
    /// The constant one.
    pub const ONE: Self = Self(1.0);

    /// The smallest value that arithmetic on [`PositiveF64`] can produce.
    ///
    /// This is the smallest positive normal `f64`. Operations whose results
    /// would underflow below this value saturate here instead; see the type
    /// documentation for more detail.
    pub const MIN_POSITIVE: Self = Self(f64::MIN_POSITIVE);

    /// Constant used in unit tsets
    #[cfg(test)]
    pub const ONE_QUARTER: Self = Self(0.25);

    /// Attempts to create a [`PositiveF64`] from an ordinary `f64`.
    pub fn new(f: f64) -> Option<Self> {
        // Can likely make this function const in Rust 1.83
        (f > 0.0).then_some(Self(f))
    }

    /// Given an [`Option<PositiveF64>`], if it is `Some` then add it to the value.
    /// Otherwise return the unmodified value.
    ///
    /// Returns the sum (or original value). Does not modify in-place.
    #[must_use]
    pub fn conditional_add(self, other: Option<Self>) -> Self { other.map_or(self, |i| i + self) }

    /// Takes an iterator over [`PositiveF64`] and produces a new iterator where
    /// each item is divided so that they all total to 1.
    ///
    /// On an empty iterator, returns a new empty iterator.
    ///
    /// Internally clones the iterator and runs it twice, so best to only use
    /// this with reference-based iterators obtained with e.g. `slice.iter()`
    /// rather than "owning" iterators like you'd get from `vec.into_iter()`.
    ///
    /// Note that the result is mathematically incorrect if the iterator
    /// contains multiple infinities: since `inf / inf` is defined to be 1.0,
    /// every infinite item normalizes to 1.0 and the items total to the
    /// number of infinities rather than to 1. (The correct behavior would be
    /// to yield `1 / <# infinities>` for each one, but that would require an
    /// extra counting pass, and this case never occurs in this crate's own
    /// usage.)
    pub fn normalized_iter<I>(iter: I) -> NormalizedIterator<I>
    where
        I: Iterator<Item = Self> + Clone,
    {
        // Compute the sum of all the items in the iterator. Because all items in
        // the iterator are positive, this will be 0 iff the iterator is empty.
        let sum = iter.clone().map(|x| x.0).sum::<f64>();
        NormalizedIterator { iter, sum }
    }

    /// The 'n' value of a threshold, as a [`PositiveF64`]
    pub fn n<const MAX: usize, T>(t: &Threshold<T, MAX>) -> Self {
        Self(t.n() as f64) // cast okay, worst case will lose precision
    }

    /// The ratio `k`/`n` of a threshold, as a [`PositiveF64`]. Guaranteed to be
    /// in the half-open range `(0, 1]`.
    pub fn k_over_n<const MAX: usize, T>(t: &Threshold<T, MAX>) -> Self {
        Self(t.k() as f64 / t.n() as f64) // casts okay, worst case will lose precision
    }

    /// One minus the ratio `k` / `n` of a threshold, as a [`PositiveF64`]. Guaranteed
    /// to be in the open range `(0, 1)`.
    ///
    /// Returns `None` if the return value would be 0, which is impermissible for the
    /// [`PositiveF64`] type.
    pub fn one_minus_k_over_n<const MAX: usize, T>(t: &Threshold<T, MAX>) -> Option<Self> {
        if t.is_and() {
            None
        } else {
            Some(Self(1.0 - t.k() as f64 / t.n() as f64)) // casts okay, worst case will lose precision
        }
    }
}

impl Eq for PositiveF64 {}

// We could derive PartialOrd, but we can't derive Ord, and clippy wants us
// to derive both or neither. Better to be explicit.
impl PartialOrd for PositiveF64 {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> { Some(self.cmp(other)) }
}

impl Ord for PositiveF64 {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        // will panic if given NaN
        self.0.partial_cmp(&other.0).unwrap()
    }
}

/// Hash required for using OrdF64 as key for hashmap
impl hash::Hash for PositiveF64 {
    fn hash<H: hash::Hasher>(&self, state: &mut H) { self.0.to_bits().hash(state); }
}

impl From<PositiveF64> for f64 {
    fn from(value: PositiveF64) -> Self { value.0 }
}

impl From<NonZeroU32> for PositiveF64 {
    fn from(value: NonZeroU32) -> Self { Self(f64::from(u32::from(value))) }
}

/// Clamps the result of an arithmetic operation on positive floats to the
/// range of permissible [`PositiveF64`] values.
///
/// Positive floats cannot produce `NaN` or negative values when added or
/// multiplied together, and can only produce `NaN` when divided in the
/// `inf / inf` case (which is special-cased in [`positive_div`]). They can,
/// however, underflow to a subnormal number or to 0.0; in this case we
/// saturate at `f64::MIN_POSITIVE`.
fn clamp_positive(f: f64) -> f64 {
    // As described above, `f` being NaN is impossible here. (If we did get a
    // NaN, `f64::max` would turn it into `f64::MIN_POSITIVE` anyway.)
    f.max(f64::MIN_POSITIVE)
}

/// Multiplies two positive floats, saturating below at `f64::MIN_POSITIVE`.
fn positive_mul(a: f64, b: f64) -> f64 { clamp_positive(a * b) }

/// Divides two positive floats, saturating below at `f64::MIN_POSITIVE`.
///
/// Defines `inf / inf` to be 1.0 to avoid producing `NaN`. A finite value
/// divided by `inf` underflows to 0.0 and therefore saturates at
/// `f64::MIN_POSITIVE`, like any other underflowing division.
fn positive_div(a: f64, b: f64) -> f64 {
    if a.is_infinite() && b.is_infinite() {
        1.0
    } else {
        clamp_positive(a / b)
    }
}

macro_rules! impl_op {
    ($trait:ident, $op:ident, $expr:expr) => {
        impl ops::$trait for PositiveF64 {
            type Output = Self;
            fn $op(self, rhs: Self) -> Self::Output { Self($expr(self.0, rhs.0)) }
        }

        impl ops::$trait for &PositiveF64 {
            type Output = PositiveF64;
            fn $op(self, rhs: Self) -> Self::Output { PositiveF64($expr(self.0, rhs.0)) }
        }

        impl ops::$trait<&PositiveF64> for PositiveF64 {
            type Output = Self;
            fn $op(self, rhs: &PositiveF64) -> Self::Output { Self($expr(self.0, rhs.0)) }
        }

        impl ops::$trait<PositiveF64> for &PositiveF64 {
            type Output = PositiveF64;
            fn $op(self, rhs: PositiveF64) -> Self::Output { PositiveF64($expr(self.0, rhs.0)) }
        }
    };
}

impl_op!(Add, add, f64::add);
impl_op!(Mul, mul, positive_mul);
impl_op!(Div, div, positive_div);

/// Iterator over [`PositiveF64`]s normalized to total to 1.
///
/// Constructed by [`PositiveF64::normalized_iter`]; see its documentation for
/// details, including a note on mathematically incorrect behavior when the
/// input contains multiple infinities.
pub struct NormalizedIterator<I> {
    iter: I,
    /// Sum must be nonnegative, and may only be zero if `iter` is empty.
    sum: f64,
}

impl<I> Iterator for NormalizedIterator<I>
where
    I: Iterator<Item = PositiveF64>,
{
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|x| PositiveF64(positive_div(x.0, self.sum)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

impl<I> DoubleEndedIterator for NormalizedIterator<I>
where
    I: Iterator<Item = PositiveF64> + DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .next_back()
            .map(|x| PositiveF64(positive_div(x.0, self.sum)))
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for NormalizedIterator<I> where
    I: Iterator<Item = PositiveF64> + ExactSizeIterator
{
}

impl<I: FusedIterator> FusedIterator for NormalizedIterator<I> where
    I: Iterator<Item = PositiveF64> + FusedIterator
{
}

#[cfg(test)]
mod tests {
    use super::PositiveF64;

    #[test]
    fn div_inf_by_inf_is_one() {
        let inf = PositiveF64::new(f64::INFINITY).unwrap();
        assert_eq!(inf / inf, PositiveF64::ONE);
    }

    #[test]
    fn arithmetic_saturates_at_min_positive() {
        let inf = PositiveF64::new(f64::INFINITY).unwrap();
        let two = PositiveF64::new(2.0).unwrap();
        let min = PositiveF64::MIN_POSITIVE;
        // A finite value divided by inf underflows to 0.0 and saturates.
        assert_eq!(PositiveF64::ONE / inf, min);
        // Dividing MIN_POSITIVE further would underflow to a subnormal
        // number; the division is instead a no-op.
        assert_eq!(min / two, min);
        // Multiplication saturates the same way.
        assert_eq!(min * min, min);
    }

    #[test]
    fn ordinary_arithmetic_unchanged() {
        let two = PositiveF64::new(2.0).unwrap();
        let three = PositiveF64::new(3.0).unwrap();
        assert_eq!(two * three, PositiveF64::new(6.0).unwrap());
        assert_eq!(three / two, PositiveF64::new(1.5).unwrap());
    }

    #[test]
    fn normalized_iter_extreme_values() {
        use super::PositiveF64 as P;

        // inf and inf: the sum is inf, and each item normalizes to 1.0.
        // This is mathematically incorrect (the items total to 2, not 1);
        // see the documented limitation on `normalized_iter`.
        let infs = [
            P::new(f64::INFINITY).unwrap(),
            P::new(f64::INFINITY).unwrap(),
        ];
        let normalized: Vec<P> = P::normalized_iter(infs.iter().copied()).collect();
        assert_eq!(normalized, vec![P::ONE, P::ONE]);

        // A tiny value next to an infinite one: saturates at MIN_POSITIVE.
        // Collected in reverse to also exercise `next_back`, which the
        // compiler uses via `rev()`.
        let mixed = [P::MIN_POSITIVE, P::new(f64::INFINITY).unwrap()];
        let normalized: Vec<P> = P::normalized_iter(mixed.iter().copied()).rev().collect();
        assert_eq!(normalized, vec![P::ONE, P::MIN_POSITIVE]);
    }
}
