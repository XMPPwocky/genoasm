use realfft::{RealToComplex};
pub type Spectrogram = (usize, Vec<f32>);

//const BAND_LOG: f32 = 1.618;

const NUM_BANDS: usize = 24;
use crate::SAMPLE_RATE;

fn hz_to_mel(hz: f32) -> f32 {
    1000.0/(2.0f32.ln()) * (1.0 + hz/1000.0).ln()
}
fn bin_to_band(bin: usize, num_bins: usize) -> usize {
    let mel_max_band = hz_to_mel(SAMPLE_RATE/2.0);

    let mel_this_band = hz_to_mel(SAMPLE_RATE / 2.0 * (bin as f32 / num_bins as f32));

    let pos = mel_this_band / mel_max_band;
    
    (pos * NUM_BANDS as f32).floor() as usize
}
fn a_weight(hz: f32) -> f32 {
    if hz < 1.0 { return 0.0; }
    (12194.0f32.powi(2) * hz.powi(4))
        / (
            (hz.powi(2) + 20.6f32.powi(2)) 
            * f32::sqrt((hz.powi(2) + 107.7f32.powi(2))*(hz.powi(2) + 737.9f32.powi(2)))
            * (hz.powi(2) + 12194.0f32.powi(2)))

    // (20.0 * ra.log10() + 2.0)
}
pub fn compute_spectrogram(inp: &[i16], r2c: &dyn RealToComplex<f32>) -> Spectrogram {
    let mut spectrums = vec![];

    // keep track of the area of each band
    // so we can normalize later
    let mut band_area = vec![0.0; NUM_BANDS];
    for i in 0..r2c.len() {
        band_area[bin_to_band(i, r2c.len())] += 1.0;
    }

    let mut indata = r2c.make_input_vec();

    let mut spectrum = r2c.make_output_vec();

    for inp_chunk in inp.windows(r2c.len()).step_by(r2c.len() / 3) {
        let spec_start = spectrums.len();
        spectrums.extend(std::iter::repeat(0.0).take(NUM_BANDS));
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
            let band = bin_to_band(bin, r2c.len());
            let hz = (bin as f32 / r2c.len() as f32) * SAMPLE_RATE / 2.0;
            spectrum_binned[band] += power * a_weight(hz) / band_area[band];
        }
    }

    (NUM_BANDS, spectrums)
}

pub fn compare_spectrograms(a: &Spectrogram, b: &Spectrogram) -> f64 {
    assert_eq!(a.0, b.0);
    let n_bands = a.0;

    let mut out = 0.0;


    for (a, b) in a.1.chunks(n_bands).zip(b.1.chunks(n_bands)) {
        let mut chunk_score = 0.0;
        for (&l, &r) in a.iter().zip(b.iter()) {
            let diff = l as f64 - r as f64;
            let lg = diff.abs();
            chunk_score += lg;
        }

        out += chunk_score.powi(2);
    }
    out / (a.1.len() as f64)
}
pub fn spectral_fitness(candidate: &[i16], seed: &[i16], r2c: &dyn RealToComplex<f32>) -> f64 {
    assert_eq!(candidate.len(), seed.len());

    let buf = compute_spectrogram(candidate, r2c);
    let seed = compute_spectrogram(seed, r2c); // this should be cached

    compare_spectrograms(&buf, &seed)
}
