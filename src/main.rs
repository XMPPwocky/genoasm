use animal::{genoasm, Animal};
use tui::{widgets::{Dataset, Chart, Block, GraphType, Axis, Paragraph, Wrap, Borders}, symbols, style::{Style, Color, Modifier}, text::{Span, Spans}, layout::{Alignment, Layout, Direction, Constraint}};
use rand::{distributions::Uniform, Rng};
use rayon::prelude::*;
use realfft::RealFftPlanner;

use tracing::{debug};

pub mod animal;
pub mod corrupt;
pub mod ecosystem;
pub mod similarity;
pub mod util;

use clap::Parser;

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
    #[arg(short, long, default_value_t = 3)]
    num_eves: u32,

    /// Population size
    #[arg(short, long, default_value_t = 256)]
    population_size: usize,

    /// Generations
    #[arg(short, long, default_value_t = 8192)]
    generations: usize,
}

use util::normalize_audio;

use crate::{similarity::spectral_fitness, util::mix_audio};

fn screen(gen: &genoasm::Genoasm) -> bool {
    let (v, _) = gen.feed(&[0x1; 1024]);
    if v.iter().filter(|x| **x != 0).count() < 64 {
        return false;
    }
    if (v[v.len() / 2..]).iter().filter(|x| **x != 0).count() < 32 {
        return false;
    }
    let (v2, _) = gen.feed(&[0xAD; 1024]);
    if v == v2 {
        return false;
    }
    if v2.iter().filter(|x| **x != 0).count() < 64 {
        return false;
    }
    /*et v3 = gen.feed(&[0x13; 1024]);
    if v3 == v2 { return false; }*/
    true
}
fn main() -> color_eyre::Result<()> {
    tracing_subscriber::fmt::init();

    use std::io;
    use termion::raw::IntoRawMode;
    use tui::{backend::TermionBackend, Terminal};

    let stdout = io::stdout().into_raw_mode()?;
    let backend = TermionBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let args = Args::parse();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut garbo = vec![];

    let seed: Result<Vec<i16>, _> = {
        let mut reader = hound::WavReader::open(args.input)?;
        if reader.spec().channels == 2 {
            reader.samples::<i16>().step_by(2).collect() // just take one channel ig
        } else {
            reader.samples::<i16>().collect()
        }
    };

    let mut seed = normalize_audio(&seed?);
    while !seed.len().is_power_of_two() {
        seed.push(0); // we want that fast FFT... sigh...
    }

    // prep fft stuff for scoring similarity
    let mut real_planner = RealFftPlanner::<f32>::new();
    let r2c = real_planner.plan_fft_forward(1024);

    let mut rng = rand::thread_rng();

    let noise: Vec<i16> = (&mut rng)
        .sample_iter(Uniform::from(i16::MIN..i16::MAX))
        .take(seed.len())
        .collect();

    let noisy_seed = mix_audio(&seed, &noise, 0.9);
    // let noisy_seed = normalize_audio(&noisy_seed);

    let mut eve;
    debug!("Generating Eve(s)");
    for i in 0..args.num_eves {
        debug!("{i}/{} Eves", args.num_eves);

        loop {
            eve = genoasm::Genoasm::spontaneous_generation();
            if screen(&eve) {
                break;
            }
        }

        let (aud, gas) = eve.feed(&noisy_seed);
        let f = spectral_fitness(&aud, &seed, &*r2c) * (gas as f64);
        garbo.push((f, aud, eve.clone()));
    }

    let mut f_history = vec![];
    let mut a_history = vec![];
    for i in 0..args.generations {
        while garbo.len() > args.population_size {
            if rng.gen_bool(0.9) {
                let q = rng.gen_range(8..16);
                let death = rng.gen_range(garbo.len() * q / 16..garbo.len());
                garbo.remove(death);
            } else {
                garbo.pop();
            }
        }
        while garbo.len() < args.population_size {
            let (aud, gen) = {
                let (_, aud1, eve) = &garbo[rng.gen_range(0..garbo.len())];
                let (_, aud2, _) = &garbo[rng.gen_range(0..garbo.len())];

                let aud: Vec<i16> = aud1.iter().zip(aud2.iter()).map(|(&a, &b)| a + b).collect();

                let gen = if rng.gen_bool(0.99) {
                    eve.mutate()
                } else {
                    Animal::spontaneous_generation()
                };
                (aud, gen)
            };
            if screen(&gen) {
                let (aud, gas) = gen.feed(&aud);
                let f = spectral_fitness(&aud, &seed, &*r2c) * gas as f64;
                garbo.push((f, aud, gen));
                debug!("Population size: {:?}", garbo.len());
            }
        }

        debug!("Generation {:?}", i);

        let news = (0..128)
            .into_par_iter()
            .filter_map(|_| {
                let mut rng = rand::thread_rng();

                let (aud, gen) = {
                    let mut v;
                    loop {
                        let idx = rng.gen_range(0..garbo.len());
                        v = &garbo[idx];
                        if rng.gen_bool((idx as f64 / (garbo.len() as f64 + 1.0)).powf(0.6)) {
                            continue;
                        }
                        break;
                    }

                    let (_, aud1, eve) = v;
                    loop {
                        let idx = rng.gen_range(0..garbo.len());
                        v = &garbo[idx];
                        if rng.gen_bool((idx as f64 / (garbo.len() as f64 + 1.0)).powf(0.6)) {
                            continue;
                        }
                        break;
                    }
                    let (_, aud2, _) = v;

                    let aud: Vec<i16> =
                        aud1.iter().zip(aud2.iter()).map(|(&a, &b)| a + b).collect();

                    let gen = eve.mutate();
                    (aud, gen)
                };

                if screen(&gen) {
                    let (aud, gas) = gen.feed(&aud);

                    let f = spectral_fitness(&aud, &seed, &*r2c) * gas as f64;

                    Some((f, aud, gen))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let cutoff = garbo[garbo.len() - 1].0;
        for (f, aud, gen) in news {
            if f >= cutoff && rng.gen_bool(0.99) {
                continue;
            }
            // horrible wasteful augh
            // do insertion sort lol
            garbo.push((f, aud, gen));
        }

        garbo.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        garbo.dedup_by(|a, b| a.1 == b.1); // remove anything with same audio (won't work if two things have exact same fitness whatever)
        f_history.push((i as f64, garbo[0].0));
        let avg = (garbo.iter().map(|x| x.0).sum::<f64>()) / garbo.len() as f64;
        a_history.push((i as f64, avg));

        let datasets = vec![
            Dataset::default()
                .name("Min")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Cyan))
                .data(&f_history),
            Dataset::default()
                .name("Mean")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Green))
                .data(&a_history)
        ];
        let annoying_max = if i > 64 { a_history[64].1 } else { a_history[0].1};
        let chart = Chart::new(datasets)
            .block(Block::default().title("LOSS").borders(Borders::ALL))
            .x_axis(Axis::default()
                .title(Span::styled("Generation", Style::default().fg(Color::LightRed)))
                .style(Style::default().fg(Color::White))
                .bounds([0.0, args.generations as f64]))
            .y_axis(Axis::default()
                .title(Span::styled("Loss", Style::default().fg(Color::LightRed)))
                .style(Style::default().fg(Color::White))
                .bounds([f_history[f_history.len() - 1].1, annoying_max]));

        terminal.clear()?;
        terminal.draw(|f| {
            let size = f.size();
    
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(0), Constraint::Percentage(20)].as_ref())
                .split(size);

            let chunks_bar = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3+2), Constraint::Min(0)].as_ref())
                .split(chunks[1]);

            let text = vec![
                Spans::from(vec![
                    Span::styled("geno", Style::default().fg(Color::LightBlue)),
                    Span::styled("ASM", Style::default().fg(Color::Green))
                ]),
                Spans::from(vec![
                    Span::styled("another bad ", Style::default()),
                    Span::styled("mimir", Style::default().fg(Color::LightCyan)),
                    Span::styled(" idea", Style::default()),
                ]),
                Spans::from(Span::styled("no copyright intended", Style::default().add_modifier(Modifier::ITALIC))),
            ];
            let para1 = Paragraph::new(text)
                .block(Block::default().title("ABOUT").borders(Borders::ALL))
                .style(Style::default().fg(Color::White).bg(Color::Black))
                .alignment(Alignment::Center)
                .wrap(Wrap { trim: true });

            
                let text = vec![
                    Spans::from(Span::raw(format!("gen {}/{}", i, args.generations))),
                    Spans::from(vec![
                        Span::raw("best loss: "),
                        Span::styled(format!("{:16}", garbo[0].0), Style::default().add_modifier(Modifier::BOLD))
                    ]),
    
                    Spans::from(vec![
                        Span::styled("vibes: ", Style::default()),
                        Span::styled("middling", Style::default().add_modifier(Modifier::BOLD))
                    ]),
                    
                    Spans::from(vec![
                        Span::raw("num threads: "),
                        Span::styled(format!("{}", rayon::current_num_threads()), Style::default().add_modifier(Modifier::BOLD))
                    ])
    
                ];
                let para2 = Paragraph::new(text)
                    .block(Block::default().title("STATS").borders(Borders::ALL))
                    .style(Style::default().fg(Color::White).bg(Color::Black))
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: true });
    
                
            f.render_widget(chart, chunks[0]);
            f.render_widget(para1, chunks_bar[0]);
            f.render_widget(para2, chunks_bar[1]);
        })?;
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
