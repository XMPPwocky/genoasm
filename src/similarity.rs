use realfft::RealToComplex;
pub type Spectrogram = (usize, Vec<f32>);

//const BAND_LOG: f32 = 1.618;

const NUM_BANDS: usize = 32;
use crate::SAMPLE_RATE;

fn hz_to_mel(hz: f32) -> f32 {
    1000.0 / (2.0f32.ln()) * (1.0 + hz / 1000.0).ln()
}
fn bin_to_band(bin: usize, num_bins: usize) -> usize {
    let mel_max_band = hz_to_mel(SAMPLE_RATE / 2.0);

    let mel_this_band = hz_to_mel(SAMPLE_RATE / 2.0 * (bin as f32 / num_bins as f32));

    let pos = mel_this_band / mel_max_band;

    (pos * NUM_BANDS as f32).floor() as usize
}
// as a raw gain, not dB
pub fn a_weight(hz: f32) -> f32 {
    let hz = f32::max(10.0, hz);
    
    let numerator = 12194.0f32.powi(2) * hz.powi(4);
    
    let denominator =
        (hz.powi(2) + 20.6f32.powi(2))
        * (hz.powi(2) + 12194.0f32.powi(2))
        * f32::sqrt((hz.powi(2) + 107.7f32.powi(2)))
        * f32::sqrt((hz.powi(2) + 737.9f32.powi(2)));
        
    numerator / denominator

    //(20.0 * (ra/0.794).log10())
}
pub fn compute_spectrogram(inp: &[i16], r2c: &dyn RealToComplex<f32>) -> Spectrogram {
    let mut spectrums = vec![];


    let mut indata = r2c.make_input_vec();
    let mut spectrum = r2c.make_output_vec();

    // keep track of the area of each band
    // so we can normalize later
    /*let mut band_area = vec![0.0; NUM_BANDS];
    for i in 0..r2c.complex_len() {
        band_area[bin_to_band(i, r2c.complex_len())] += 1.0;
    }*/

    for inp_chunk in inp.windows(r2c.len()).step_by(r2c.len() / 2) {
        let spec_start = spectrums.len();
        spectrums.extend(std::iter::repeat(0.0).take(NUM_BANDS));
        let spectrum_binned = &mut spectrums[spec_start..];

        for (i, (x, z)) in indata.iter_mut().zip(inp_chunk.iter()).enumerate() {
            // hann window
            let window = ((i as f32 / inp_chunk.len() as f32) * std::f32::consts::PI)
                .sin()
                .powi(2);
            *x = *z as f32 * window;
        }

        r2c.process(&mut indata, &mut spectrum).unwrap();

        let power_spec = spectrum.iter().map(|complex| complex.norm());

        for (bin, val) in power_spec.enumerate() {
            let band = bin_to_band(bin, r2c.complex_len());
            let hz = bin as f32 * SAMPLE_RATE / r2c.len() as f32;
            spectrum_binned[band] += val * a_weight(hz); // / band_area[band];
        }
    }

    (spectrum.len(), spectrums)
}

pub fn compare_spectrograms_internal<'a>(
    a: &'a Spectrogram,
    b: &'a Spectrogram,
) -> impl Iterator<Item = f64> + 'a {
    assert_eq!(a.0, b.0);
    let n_bands = a.0;

    a.1.chunks(n_bands)
        .zip(b.1.chunks(n_bands))
        .map(|(a, b)| {
            a.iter().zip(b.iter()).map(|(&l, &r)| {
                let (l, r) = (l as f64, r as f64);

                (l - r).powi(2) // / f64::max(1e-30, r.norm_sqr() as f64)

                //let diff = (l.ln() - r.ln()).abs();
                
            }).sum::<f64>()
            //if d.is_finite() { d } else { 1e50 }
        })
}

pub fn compare_spectrograms(a: &Spectrogram, b: &Spectrogram) -> f64 {
    compare_spectrograms_internal(a, b).sum()
}

pub fn spectrogram_error_vector(a: &Spectrogram, b: &Spectrogram) -> ErrorVector {
    ErrorVector(compare_spectrograms_internal(a, b).collect())
}

#[derive(Clone)]
pub struct ErrorVector(pub Vec<f64>);
impl ErrorVector {
    pub fn ones(len: usize) -> Self {
        ErrorVector(std::iter::repeat(1.0).take(len).collect())
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn sum(&self) -> f64 {
        self.0.iter().cloned().sum()
    }
    pub fn dot(&self, other: &ErrorVector) -> f64 {
        self.0
            .iter()
            .cloned()
            .zip(other.0.iter().cloned())
            .map(|(a, b)| a * b)
            .sum()
    }
    pub fn inv_dot(&self, other: &ErrorVector) -> f64 {
        self.0
            .iter()
            .cloned()
            .zip(other.0.iter().cloned())
            .map(|(a, b)| a * ((1.0 / b).clamp(0.0, 1e50)))
            .sum()
    }
    pub fn scale(&mut self, scale: f64) {
        for x in self.0.iter_mut() {
            *x *= scale;
        }
    }
    pub fn normalize(&mut self) {
        let scale = self.sum();
        self.scale(1.0 / scale);
    }
}
impl<'a> std::ops::AddAssign<&'a Self> for ErrorVector {
    fn add_assign(&mut self, rhs: &'a Self) {
        for (out, r) in self.0.iter_mut().zip(rhs.0.iter()) {
            *out += *r;
        }
    }
}
