use realfft::{RealToComplex};
pub type Spectrogram = (usize, Vec<f32>);

const BAND_LOG: f32 = 1.618;

fn bin_to_band(bin: usize, base: f32) -> usize {
    let bin = bin as f32;
    
    (bin + 1.0).log(base) as usize
}
pub fn compute_spectrogram(inp: &[i16], r2c: &dyn RealToComplex<f32>) -> Spectrogram {
    let n_bands = bin_to_band(inp.len() - 1, BAND_LOG) + 1;

    let mut spectrums = vec![];

    let mut indata = r2c.make_input_vec();

    let mut spectrum = r2c.make_output_vec();

    for inp_chunk in inp.windows(r2c.len()).step_by(r2c.len() / 3) {
        let spec_start = spectrums.len();
        spectrums.extend(std::iter::repeat(0.0).take(n_bands));
        let spectrum_binned = &mut spectrums[spec_start..];

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
            let band = bin_to_band(bin, BAND_LOG);
            spectrum_binned[band] += power;
        }
    }

    (n_bands, spectrums)
}

pub fn compare_spectrograms(a: &Spectrogram, b: &Spectrogram) -> f64 {
    assert_eq!(a.0, b.0);
    let n_bands = a.0;

    let mut out = 0.0;

    for (a, b) in a.1.chunks(n_bands).zip(b.1.chunks(n_bands)) {
        for (i, (&l, &r)) in a.iter().zip(b.iter()).enumerate() {
            let pos = (i as f64) / (a.len() as f64);
            let scale = 1.0 - pos;

            let diff = (l - r).powi(2);
            out += diff as f64 * scale;
        }
    }
    out / (a.1.len() as f64)
}
pub fn spectral_fitness(candidate: &[i16], seed: &[i16], r2c: &dyn RealToComplex<f32>) -> f64 {
    assert_eq!(candidate.len(), seed.len());

    let buf = compute_spectrogram(candidate, r2c);
    let seed = compute_spectrogram(seed, r2c); // this should be cached

    compare_spectrograms(&buf, &seed)
}
