use realfft::{RealToComplex};
pub type Spectrogram = Vec<Vec<f32>>;

pub fn compute_spectrogram(inp: &[i16], r2c: &dyn RealToComplex<f32>) -> Spectrogram {
    let mut spectrums = vec![];
    let mut indata = r2c.make_input_vec();

    for inp_chunk in inp.windows(r2c.len()).step_by(r2c.len() / 3) {
        for (i, (x, z)) in indata.iter_mut().zip(inp_chunk.iter()).enumerate() {
            // hann window
            let window = ((i as f32 / inp_chunk.len() as f32) * std::f32::consts::PI).sin().powi(2);
            *x = *z as f32 * window;
        }


        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut indata, &mut spectrum).unwrap();

        let power_spec = spectrum
            .into_iter()
            .map(|complex| complex.norm_sqr())
            .collect();
        
        spectrums.push(power_spec);
    }

    spectrums
}

pub fn compare_spectrograms(a: &Spectrogram, b: &Spectrogram) -> f64 {
    let mut out = 0.0;

    for (a, b) in a.iter().zip(b.iter()) {
        for (_i, (&l, &r)) in a.iter().zip(b.iter()).enumerate() {
            //let pos = (i as f64) / (a.len() as f64);
            let scale = 1.0; //(1.0 - pos);

            let diff = l - r;
            out += (diff as f64).powi(2) * scale;
        }
    }
    out
}
pub fn spectral_fitness(candidate: &[i16], seed: &[i16], r2c: &dyn RealToComplex<f32>) -> f64 {
    assert_eq!(candidate.len(), seed.len());

    let buf = compute_spectrogram(candidate, r2c);
    let seed = compute_spectrogram(seed, r2c); // this should be cached

    compare_spectrograms(&buf, &seed)
}
