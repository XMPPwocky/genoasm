use crate::animal::AnimalInfo;
use std::collections::BTreeMap;

pub struct Ecosystem<Animal> {
    population: Vec<(Animal, AnimalInfo)>,
    max_size: usize
}
impl<Animal: Send> Ecosystem<Animal> {
    pub fn new(max_size: usize) -> Self {
        assert!(max_size > 0);
        Ecosystem {
            population: Vec::with_capacity(max_size),
            max_size
        }
    }

    pub fn iter(&self) -> impl Iterator<Item=&(Animal, AnimalInfo)> {
        self.population.iter()
    }

    pub fn extend(&mut self, iter: impl Iterator<Item=(Animal, AnimalInfo)>) {
        self.population.extend(iter);

        self.sort();

        if self.population.len() > self.max_size {
            self.population.drain(self.max_size..);
        }
        self.check_rep();
    }

    pub fn update_costs(&mut self, mut cost_fn: impl FnMut(&(Animal, AnimalInfo)) -> f64) {
        // no drain for btreemap :(

        for p in &mut self.population {
            p.1.cost = cost_fn(p);
        }

        self.sort();
        self.check_rep();
    }

    fn sort(&mut self) {
        use rayon::prelude::*;
        self.population.par_sort_unstable_by(|a, b| a.1.cost.partial_cmp(&b.1.cost).unwrap());
    }

    fn check_rep(&self) {
        use std::cmp::Ordering;

        for window in self.population.windows(2) {
            let prev = window[0].1.cost;
            let next = window[1].1.cost;
            let res = prev.partial_cmp(&next);

            if !matches!(res, Some(Ordering::Less)|Some(Ordering::Equal)) {
                panic!("ecosystem corrupted (sort): {prev} {next} {res:?}");
            }
        }
    }
}