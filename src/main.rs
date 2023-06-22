use animal::{genoasm, Animal};
use rand::Rng;
use tracing::info;
use core::cmp::Ordering;

pub mod animal;
pub mod similarity;
pub mod ecosystem;
pub mod corrupt;


use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input file path
    #[arg(short, long)]
    input: String,

    /// Output file path
    #[arg(short, long)]
    output: String,

    /// Number of Eves (spontaneously generated individuals)
    #[arg(short, long, default_value_t=3)]
    num_eves: u32,

    /// Population size
    #[arg(short, long, default_value_t=256)]
    population_size: usize,

    /// Generations
    #[arg(short, long, default_value_t=8192)]
    generations: usize
}


fn fitness(candidate: &[i16], seed: &[i16]) -> f32 {
    let mut out = 0.0;
    // let's just a least-squares yeah?

    for (&c, &s) in candidate.iter().zip(seed.iter()) {
        let difference = ((c as f32) - (s as f32)) / i16::MAX as f32;
        let sqr = difference*difference;
        out -= sqr; // minus because bigger error = worse
    }
    -out // HACK for me screwing up the sort order later lmao 
}

fn screen(gen: &genoasm::Genoasm) -> bool {
    let v = gen.feed(&[0x1; 2048]);
    if v.iter().filter(|x| **x != 0).count() < 512 { 
        return false;
    }
    if (v[v.len() / 2..]).iter().filter(|x| **x != 0).count() < 64 { 
        return false;
    }
    let v2 = gen.feed(&[0xAD; 2048]);
    if v == v2 { return false; }
    if v2.iter().filter(|x| **x != 0).count() < 256 { 
        return false;
    }
    let v3 = gen.feed(&[0x13; 2048]);
    if v3 == v2 { return false; }
    true
}
fn main() -> color_eyre::Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut garbo = vec![];

    let mut rng = rand::thread_rng();

    let seed: Result<Vec<i16>, _> = {
        let mut reader = hound::WavReader::open(args.input)?;
        if reader.spec().channels == 2 {
            reader.samples::<i16>().step_by(2).collect() // just take one channel ig
        } else {
            reader.samples::<i16>().collect()
        }
    };

    let seed = seed?;
    // normalize 4 later
    let gain = i16::MAX as f32 / *seed.iter().max().unwrap_or(&1) as f32;
    let seed: Vec<_> = seed.into_iter().map(|x| (x as f32 * gain) as i16).collect();

    let mut eve;
    info!("Generating Eve(s)");
    for i in 0..args.num_eves {
        loop {
            eve = genoasm::Genoasm::spontaneous_generation();
            if screen(&eve) { break }
        }
        println!("{i}");
        
        let aud = eve.feed(&seed);
        let f = fitness(&aud, &seed);
        garbo.push((f, aud, eve.clone()));
    }

    for i in 0..args.generations {
        while garbo.len() < args.population_size {
            let (aud, gen) = {
                let (_, aud1, eve) = &garbo[rng.gen_range(0..garbo.len())];
                let (_, aud2, _) = &garbo[rng.gen_range(0..garbo.len())];

                let aud: Vec<i16> = aud1.iter().zip(aud2.iter()).map(|(&a, &b)| a+b).collect();

                let gen = eve.mutate();
                (aud, gen)
            };
            if screen(&gen) {
                let aud = gen.feed(&aud);
                let f = fitness(&aud, &seed);
                garbo.push((f, aud, gen));
                info!("Population size: {:?}", garbo.len());
            }
        }
        
        println!("Generation {:?}", i);
        let (aud, gen) = {
            let mut v;
            loop {
                let idx = rng.gen_range(0..garbo.len());
                v = &garbo[idx];
                if rng.gen_bool((idx as f64 / (garbo.len() as f64 + 1.0)).powf(0.8)) {
                    continue;
                }
                break;
            };
            let (_, aud1, eve) = v;
            loop {
                let idx = rng.gen_range(0..garbo.len());
                v = &garbo[idx];
                if rng.gen_bool((idx as f64 / (garbo.len() as f64 + 1.0)).powf(0.8)) {
                    continue;
                }
                break;
            };
            let (_, aud2, _) = v;

            let aud: Vec<i16> = aud1.iter().zip(aud2.iter()).map(|(&a, &b)| a+b).collect();

            let gen = eve.mutate();
            (aud, gen)
        };
        if screen(&gen) {
            let aud = gen.feed(&aud);

            let f = fitness(&aud, &seed);
    
            //let pos = garbo.partition_point(|&(f2, _, _)| f2 > f);
            //if pos == garbo.len() { continue; }
            //println!("{:?} {:?} {:?}", pos, garbo[pos].0, f);
            garbo.push((f, aud, gen)); // think this copies the whole genome lmao

            // and this is just silly - use partition_point you goober
            garbo.sort_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
            });

            let q = rng.gen_range(1..15);
            let death = rng.gen_range(garbo.len() * q / 16 .. garbo.len());
            garbo.remove(death);
            // horrible wasteful augh
            garbo.dedup_by(|a, b| a.1 == b.1); // remove anything with same audio

            //println!();
            //assert!(garbo.is_sorted_by(|a, b| a.0.partial_cmp(b.0)));
        }
    }

    let mut writer = hound::WavWriter::create(args.output, spec).unwrap();

    for (_, aud, _) in garbo {
        for s in aud {
            writer.write_sample(s)?;
        }
    }

    writer.finalize().unwrap();

    Ok(())
}