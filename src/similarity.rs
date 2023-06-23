use realfft::{RealToComplex};
use rustfft::num_complex::Complex;
pub fn fft(inp: &[i16], r2c: &dyn RealToComplex<f32>) -> Vec<Vec<Complex<f32>>> {
    let mut spectrums = vec![];
    let mut indata = r2c.make_input_vec();

    for inp_chunk in inp.windows(r2c.len()).step_by(r2c.len() / 2) {
        for (i, (x, z)) in indata.iter_mut().zip(inp_chunk.iter()).enumerate() {
            // cos window
            let window = ((i as f32 / inp_chunk.len() as f32) * std::f32::consts::PI).sin();
            *x = *z as f32 * window;
        }


        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut indata, &mut spectrum).unwrap();
        spectrums.push(spectrum);
    }

    spectrums
}
pub fn spectral_fitness(candidate: &[i16], seed: &[i16], r2c: &dyn RealToComplex<f32>) -> f64 {
    assert_eq!(candidate.len(), seed.len());

    let mut out = 0.0;

    let buf = fft(candidate, r2c);
    let seed = fft(seed, r2c); // this should be cached

    for (seed, buf) in seed.iter().zip(buf.iter()) {
        for (i, (&a, &b)) in seed.iter().zip(buf.iter()).enumerate() {
            let pos = (i as f64) / (seed.len() as f64);
            let scale = 1.0 - pos;

            let diff = (a.norm() - b.norm()).abs();
            out += diff.abs() as f64 * scale;
        }
    }
    out
}
