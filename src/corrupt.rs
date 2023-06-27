use std::ops::Range;

use rand::rngs::StdRng;

pub struct IntegerMutateParams<T> {
    /// Range of values that may be randomly added or subtracted.
    pub add_range: Range<T>,
}

pub fn mutate_u8(rng: &mut StdRng, x: &mut u8, params: &IntegerMutateParams<u8>) {
    todo!()
}