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
        let trials = self.trials.load(Ordering::SeqCst).max(1);
  
        // wilson
        let n = trials as f64;
        let x = wins as f64 / n;
        let z = 5.0f64; // 1.96f64;
        //(x + z.powi(2)/2.0) / (n + z.powi(2))
        // modified- prior = p=0.1. no idea if this is sound, probably not lmao
        (x + z.powi(2)/10.0) / (n + z.powi(2))
    }
}
impl Clone for AnimalInfo {
    fn clone(&self) -> Self {
        AnimalInfo {
            cost: self.cost,
            audio: self.audio.clone(),
            spectrogram: self.spectrogram.clone(),
            
            wins: AtomicUsize::new(self.wins.load(Ordering::SeqCst)),
            trials: AtomicUsize::new(self.trials.load(Ordering::SeqCst))
        }
    }
}