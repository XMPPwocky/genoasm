use realfft::{RealToComplex};
pub type Spectrogram = Vec<Vec<f32>>;

fn bin_to_band(bin: usize, base: f32) {
    let bin = bin as f32;

}
pub fn compute_spectrogram(inp: &[i16], r2c: &dyn RealToComplex<f32>) -> Spectrogram {
    let n_bands = inp.len().next_power_of_two().trailing_zeros() + 1;

    let mut spectrums = vec![];

    let mut indata = r2c.make_input_vec();


    let mut spectrum = r2c.make_output_vec();

    for inp_chunk in inp.windows(r2c.len()).step_by(r2c.len() / 3) {
        let mut spectrum_binned = vec![0.0; n_bands];

        for (i, (x, z)) in indata.iter_mut().zip(inp_chunk.iter()).enumerate() {
            // hann window
            let window = ((i as f32 / inp_chunk.len() as f32) * std::f32::consts::PI).sin().powi(2);
            *x = *z as f32 * window;
        }

        r2c.process(&mut indata, &mut spectrum).unwrap();

        let power_spec = spectrum
            .iter()
            .map(|complex| complex.norm_sqr());
        
        for (bin, power) in power_spec.enumerate() {
            let band = bin.next_power_of_two().trailing_zeros();
            spectrum_binned[band as usize] += power;
        }

        spectrums.push(spectrum_binned);
    }

    spectrums
}

pub fn compare_spectrograms(a: &Spectrogram, b: &Spectrogram) -> f64 {
    let mut out = 0.0;

    for (a, b) in a.iter().zip(b.iter()) {
        for (i, (&l, &r)) in a.iter().zip(b.iter()).enumerate() {
            let pos = (i as f64) / (a.len() as f64);
            let scale = 1.0 - pos;

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
