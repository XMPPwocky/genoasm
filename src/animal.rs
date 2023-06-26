use core::sync::atomic::{AtomicUsize, Ordering};

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

    pub wins: AtomicUsize,
    pub trials: AtomicUsize
}
impl AnimalInfo {
    pub fn win_rate(&self) -> f64 {
        let wins = self.wins.load(Ordering::SeqCst);
        let trials = self.trials.load(Ordering::SeqCst);
        let prior = 0.1;

        let x = if trials == 0 {
            prior
        } else {
            (prior + ((1.0 - prior) * wins as f64 / trials as f64 / 2.0)).powf(trials as f64 / 64.0)
        };
        (x * 0.995) + 0.005
    }
}