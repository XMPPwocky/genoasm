use animal::{genoasm, Animal};
use tui::{widgets::{Dataset, Chart, Block, GraphType, Axis, Paragraph, Wrap, Borders, Gauge}, symbols, style::{Style, Color, Modifier}, text::{Span, Spans}, layout::{Alignment, Layout, Direction, Constraint}};
use rand::Rng;
use rayon::prelude::*;
use realfft::RealFftPlanner;

use tracing::{debug};

use util::normalize_audio;

use crate::{similarity::{spectral_fitness, compute_spectrogram, compare_spectrograms}, util::mix_audio, animal::{genoasm::Genoasm, AnimalInfo}};

pub mod animal;
pub mod corrupt;
pub mod ecosystem;
pub mod similarity;
pub mod util;
pub mod vm;

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

    /// Best output file path
    #[arg(short, long)]
    best_output: Option<String>,


    /// Number of Eves (spontaneously generated individuals)
    #[arg(short, long, default_value_t = 3)]
    num_eves: u32,

    /// Population size
    #[arg(short, long, default_value_t = 256)]
    population_size: usize,

    /// Generations.
    #[arg(short, long, default_value_t = 8192)]
    generations: usize,

    /// Min difference between parent and child.
    /// 
    /// This probably should be higher for longer inputs...
    #[arg(short, long, default_value_t = 1e6)]
    parent_child_diff: f64,

    /// FFT size used for similarity computations
    #[arg(short, long, default_value_t = 512)]
    fft_size: usize,

    /// Attenuates genoASM's tendency to prefer lower-loss parents
    /// when generating children. (i.e. higher explore = less preference for low-loss parents)
    /// 
    /// range: 0..+inf, though 0..1.0 is probably best
    /// 
    /// extremely low (i.e. much less than 0.1)
    /// values may cause slightly noticeable CPU usage because of the use of
    /// rejection sampling
    #[arg(short, long, default_value_t = 0.2)]
    explore: f64,
}

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

    let mut garbo: Vec<(Genoasm, AnimalInfo)> = vec![];

    let seed: Result<Vec<i16>, _> = {
        let mut reader = hound::WavReader::open(args.input)?;
        if reader.spec().channels == 2 {
            reader.samples::<i16>().step_by(2).collect() // just take one channel ig
        } else {
            reader.samples::<i16>().collect()
        }
    };

    // prep fft stuff for scoring similarity
    let mut real_planner = RealFftPlanner::<f32>::new();
    let r2c = real_planner.plan_fft_forward(args.fft_size);


    let seed = normalize_audio(&seed?);
    let seed_spec = compute_spectrogram(&seed, &*r2c);

    let mut rng = rand::thread_rng();

    let noisy_seed: Vec<i16> = seed.iter().cloned()
        .map(|x| if rng.gen_bool(0.03) { x } else { 0 })
        .collect::<Vec<i16>>();

    //lt noisy_seed = mix_audio(&seed, &noise, 0.1);
    let noisy_seed = normalize_audio(&noisy_seed);

    let mut eve;
    debug!("Generating Eve(s)");
    for i in 0..args.num_eves {
        debug!("{i}/{} Eves", args.num_eves);


        terminal.clear()?;
        terminal.draw(|f| {
            let size = f.size();

            let gauge = Gauge::default()
                .label("Generating Eve(s)")
                .gauge_style(Style::default().fg(Color::LightCyan).bg(Color::Black))
                .ratio((i as f64) / args.num_eves as f64);

            f.render_widget(gauge, size)
        })?;

        loop {
            eve = genoasm::Genoasm::spontaneous_generation();
            if screen(&eve) {
                break;
            }
        }

        let (aud, gas) = eve.feed(&noisy_seed);
        let spec = compute_spectrogram(&aud, &*r2c);
        let f = spectral_fitness(&aud, &seed, &*r2c) * (gas as f64);
        let info = AnimalInfo {
            cost: f,
            spectrogram: spec,
            audio: aud
        };
        garbo.push((eve.clone(), info));
    }

    // do this up here i guess too
    garbo.par_sort_unstable_by(|a, b| a.1.cost.partial_cmp(&b.1.cost).unwrap());
    let mut best_cost = garbo[0].1.cost;

    let mut f_history = vec![];
    let mut a_history = vec![];

    let mut best_writer = args.best_output.as_ref()
        .map(|filename| hound::WavWriter::create(filename, spec).unwrap());

    for i in 0..args.generations {
        if garbo[0].1.cost < best_cost {
            best_cost = garbo[0].1.cost;

            if let Some(bw) = best_writer.as_mut() {
                for s in &garbo[0].1.audio {
                    bw.write_sample(*s)?;
                }
                bw.flush()?;
            }
        }
        if i % 128 == 0 {
            let mut writer = hound::WavWriter::create(&args.output, spec).unwrap();

            for (_, info) in &garbo {
                for s in &info.audio {
                    writer.write_sample(*s)?;
                }
            }
        
            writer.finalize().unwrap();        
        }


        while garbo.len() > args.population_size {
            // free perf: use retain or whatever to not do an O(n) operation in a loop
            let q = 8;
            let death = rng.gen_range(garbo.len() * q / 16..garbo.len() * 31/32);
            garbo.remove(death);
        }
        while garbo.len() < args.population_size {
            let (par_info, gen) = {
                let (eve, eve_info) = &garbo[rng.gen_range(0..garbo.len())];
                let (adam, adam_info)= &garbo[rng.gen_range(0..garbo.len())];

                if rng.gen_bool(0.2) {
                    (eve_info, eve.mutate())
                } else {
                    (adam_info, eve.befriend(adam).mutate())
                }
            };
            if !screen(&gen) {
                continue; // screening failed, not viable
            }
            let (aud, gas) = gen.feed(&par_info.audio);
            let spec = compute_spectrogram(&aud, &*r2c);
            let sim = compare_spectrograms(&spec, &par_info.spectrogram);

            if sim > args.parent_child_diff {
                let f = compare_spectrograms(&spec, &seed_spec) * gas as f64;
                let info = AnimalInfo { cost: f, audio: aud, spectrogram:  spec };
                garbo.push((gen, info));
                debug!("Population size: {:?}", garbo.len());
            }
        }

        debug!("Generation {:?}", i);

        let news = (0..64)
            .into_par_iter()
            .filter_map(|_| {
                let mut rng = rand::thread_rng();

                let (gen, parent_info, parent_audio) = {
                    let mut v;
                    loop {
                        let idx = rng.gen_range(0..garbo.len());
                        v = &garbo[idx];
                        if rng.gen_bool((idx as f64 / (garbo.len() as f64 + 1.0)).powf(args.explore)) {
                            continue;
                        }
                        break;
                    }

                    let (eve, eve_info) = v;
                    loop {
                        let idx = rng.gen_range(0..garbo.len());
                        v = &garbo[idx];
                        if rng.gen_bool((idx as f64 / (garbo.len() as f64 + 1.0)).powf(args.explore)) {
                            continue;
                        }
                        break;
                    }
                    let (_adam, adam_info) = v;

                    let gen = eve.mutate();
                    (gen, eve_info, &mix_audio(&eve_info.audio, &adam_info.audio, 0.5))
                };

                if !screen(&gen) {
                    return None; // screening failed, not viable
                }

                let (aud, gas) = gen.feed(parent_audio);
    
                //gotta do this here because we mix
                let parent_spec = compute_spectrogram(parent_audio, &*r2c);
                let spec = compute_spectrogram(&aud, &*r2c);

                let sim = compare_spectrograms(&spec, &parent_spec);
                let f = compare_spectrograms(&spec, &seed_spec) * (gas as f64);
                let info = AnimalInfo {
                    cost: f,
                    audio: aud,
                    spectrogram: spec
                };
                if sim > args.parent_child_diff || f < parent_info.cost {
                    Some((gen, info))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let cutoff = garbo[garbo.len() - 1].1.cost;
        for (gen, info) in news {
            if info.cost >= cutoff && rng.gen_bool(0.995) {
                continue;
            }

            // horrible wasteful augh
            // do insertion sort lol
            garbo.push((gen, info));
        }

        // again there's no excuse not to do insertion sort here
        // partition_point just always screws me up w/ off-by-ones
        garbo.par_sort_unstable_by(|a, b| a.1.cost.partial_cmp(&b.1.cost).unwrap());
        garbo.dedup_by(|a, b| {
            // cache this. what are you Doing. FREEPERF
            let sim = compare_spectrograms(&a.1.spectrogram, &b.1.spectrogram);
            sim <= args.parent_child_diff
        }); 
        f_history.push((i as f64, garbo[0].1.cost.ln()));
        // i don't think we need this is_finite anymore, spectral similarity used to occasionally let a lil' +/-inf slip through
        // which screwed up averages forever

        // secretly upper 50th %ile now :U
        let avg = (garbo.iter().map(|x| x.1.cost).take(garbo.len()/2).filter(|x| x.is_finite()).sum::<f64>()) / garbo.len() as f64;
        a_history.push((i as f64, avg.ln()));

        let datasets = vec![
            Dataset::default()
                .name("Mean")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Green))
                .data(&a_history),
            Dataset::default()
                .name("Min")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Cyan))
                .data(&f_history),
        ];
        let annoying_max = a_history[0].1;
        let chart = Chart::new(datasets)
            .block(Block::default().title("LOSS").borders(Borders::ALL))
            .x_axis(Axis::default()
                .title(Span::styled("Generation", Style::default().fg(Color::LightRed)))
                .style(Style::default().fg(Color::White))
                .bounds([0.0, i as f64]))
            .y_axis(Axis::default()
                .title(Span::styled("Loss", Style::default().fg(Color::LightRed)))
                .style(Style::default().fg(Color::White))
                .bounds([f_history[f_history.len() - 1].1, annoying_max]));

        terminal.clear()?;
        // inexcusable to just have this ,, sitting here in the main loop lmao
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
                        Span::styled(format!("{:16}", garbo[0].1.cost), Style::default().add_modifier(Modifier::BOLD))
                    ]),
    
                    Spans::from(vec![
                        Span::styled("vibes: ", Style::default()),
                        Span::styled("sleepy", Style::default().add_modifier(Modifier::BOLD))
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
    
    if let Some(bw) = best_writer { bw.finalize()?; }

    Ok(())
}
