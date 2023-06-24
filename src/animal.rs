use crate::similarity::Spectrogram;

pub mod genoasm;
pub trait Animal {
    fn spontaneous_generation() -> Self;
    fn befriend(&self, friend: &Self) -> Self;
    fn mutate(&self) -> Self;
}

pub struct AnimalInfo {
    pub cost: f64,
    pub audio: Vec<i16>,
    pub spectrogram: Spectrogram,
}