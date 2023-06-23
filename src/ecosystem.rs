pub struct Ecosystem<Animal> {
    pub population: Generation<Animal>,
}

pub struct Generation<Animal> {
    pub number: usize,
    pub individuals: Vec<Animal>,
}
