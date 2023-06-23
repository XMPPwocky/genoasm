pub mod genoasm;
pub trait Animal {
    fn spontaneous_generation() -> Self;

    fn befriend(&self, friend: &Self) -> Self;

    fn mutate(&self) -> Self;
}
