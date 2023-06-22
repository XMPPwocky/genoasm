use animal::{genoasm, Animal};
use rand::Rng;
use tracing::info;

pub mod animal;
pub mod similarity;
pub mod ecosystem;
pub mod corrupt;

fn screen(gen: &genoasm::Genoasm) -> bool {
    let v = gen.feed(&[0x0; 4096]);
    if v.iter().filter(|x| **x != 0).count() < 16 { 
        return false;
    }
    if (v[v.len() / 2..]).iter().filter(|x| **x != 0).count() < 8 { 
        return false;
    }
    let v2 = gen.feed(&[0xAD; 4096]);
    if v == v2 { return false; }
    true
}
fn main() -> color_eyre::Result<()> {
    tracing_subscriber::fmt::init();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create("test1.wav", spec).unwrap();

    let mut garbo = vec![];

    let mut rng = rand::thread_rng();

    let seed: Result<Vec<i16>, _> = {
        let mut reader = hound::WavReader::open("../../Downloads/rainyUI/rainyUI_13.wav")?;
        reader.samples::<i16>().collect()
    };

    let seed = seed?;

    let mut eve;
    info!("Generating Eve(s)");
    for i in 0..4 {
        loop {
            eve = genoasm::Genoasm::spontaneous_generation();
            if screen(&eve) { break }
        }
        
        let aud = eve.feed(&seed);
        garbo.push((aud, eve.clone()));
        garbo.push((seed.clone(), eve)); // HACK
    }

    while garbo.len() < 64 {
        let (aud, gen) = {
            let (aud1, eve) = &garbo[rng.gen_range(0..garbo.len())];
            let (aud2, _) = &garbo[rng.gen_range(0..garbo.len())];

            let aud: Vec<i16> = aud1.iter().zip(aud2.iter()).map(|(&a, &b)| a+b).collect();

            let aud = eve.feed(&aud);
            let gen = eve.mutate();
            (aud, gen)
        };
        if screen(&gen) {
            let aud = gen.feed(&aud);

            garbo.push((aud, gen));
            info!("Population size: {:?}", garbo.len());
        }
    }

    for i in 0..2048 {
        let (aud, gen) = {
            let (aud1, eve) = &garbo[rng.gen_range(0..garbo.len())];
            let (aud2, _) = &garbo[rng.gen_range(0..garbo.len())];

            let aud: Vec<i16> = aud1.iter().zip(aud2.iter()).map(|(&a, &b)| a+b).collect();

            let aud = eve.feed(&aud);
            let gen = eve.mutate();
            (aud, gen)
        };
        if screen(&gen) {
            let aud = gen.feed(&aud);

            for &j in &aud {
                writer.write_sample(j)?;
            }

            let x = garbo.len();
            garbo[rng.gen_range(0..x)] = (aud, gen);
        }
    }

    writer.finalize().unwrap();

    Ok(())
}