pub fn normalize_audio(inp: &[i16]) -> Vec<i16> {
    let highest_peak = inp
        .iter()
        .map(|x| x.unsigned_abs() as u32)
        .max()
        .unwrap_or(1)
        .max(1);
    let gain = i16::MAX as f32 / highest_peak as f32;

    inp.iter().map(|&x| (x as f32 * gain) as i16).collect()
}

pub fn mix_audio(a: &[i16], b: &[i16], x: f32) -> Vec<i16> {
    a.iter()
        .cloned()
        .zip(b.iter().cloned())
        .map(|(a, b)| (a as f32, b as f32))
        .map(|(a, b)| (a * (1.0 - x)) + (b * x))
        .map(|out| out as i16)
        .collect()
}
