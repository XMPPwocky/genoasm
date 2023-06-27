use core::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use std::collections::VecDeque;
use animal::{genoasm, Animal};
use tui::{widgets::{Dataset, Chart, Block, GraphType, Axis, Paragraph, Wrap, Borders, Gauge}, symbols, style::{Style, Color, Modifier}, text::{Span, Spans}, layout::{Alignment, Layout, Direction, Constraint}};
use rand::{Rng, seq::IteratorRandom};
use rayon::prelude::*;
use realfft::RealFftPlanner;

use tracing::{debug};

use util::normalize_audio;

use crate::{similarity::{spectral_fitness, compute_spectrogram, compare_spectrograms}, animal::{genoasm::Genoasm, AnimalInfo}};

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
    const SCREEN_LEN: usize = 4096;
    let (v, _) = gen.feed(&[0x42; SCREEN_LEN], Some(&[0x1; SCREEN_LEN]));
    //if v_gas < 1024 { return false; }

    if v.iter().skip(SCREEN_LEN/2).filter(|&&x| x != 0).count() < 1500 { return false; }

    let (v2, _) = gen.feed(&[0x42; SCREEN_LEN], Some(&[0x7734; SCREEN_LEN]));
    if v == v2 {
        return false;
    }

    /*et v3 = gen.feed(&[0x13; 1024]);
    if v3 == v2 { return false; }*/
    true
}
fn main() -> color_eyre::Result<()> {
    tracing_subscriber::fmt::init();

    use std::io;
    let stdout = io::stdout();
    let backend = tui::backend::CrosstermBackend::new(stdout);

    let mut terminal = tui::Terminal::new(backend)?;

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

    let mut noisy_seed: Vec<i16> = Vec::with_capacity(seed.len());
    let mut j = seed[0];
    for (i, &m) in seed.iter().enumerate() {
        //j = m;
        //if rng.gen_bool(0.5) { j = m; }

       //if i % 22050 < 8192 {
        //    noisy_seed.push(j);
        //} else {
            noisy_seed.push(0);
//}
        ////noisy_seed.push(rng.gen());
    }



    //let noisy_seed = mix_audio(&seed, &noise, 0.1);
    let noisy_seed = normalize_audio(&noisy_seed);
    let noisy_seed_info = AnimalInfo { cost: 0.0, audio: noisy_seed.clone(), spectrogram: compute_spectrogram(&noisy_seed, &*r2c),
        wins: AtomicUsize::new(0), trials: AtomicUsize::new(0) };


    let mut eve;
    debug!("Generating Eve(s)");

    terminal.clear()?;

    for i in 0..args.num_eves {
        debug!("{i}/{} Eves", args.num_eves);

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

        let (aud, gas) = eve.feed(&noisy_seed, None);
        let spec = compute_spectrogram(&aud, &*r2c);
        let f = spectral_fitness(&aud, &seed, &*r2c) * (gas as f64);
        let info = AnimalInfo {
            cost: f,
            spectrogram: spec,
            audio: aud,
            wins: AtomicUsize::new(0), trials: AtomicUsize::new(0)
        };
        garbo.push((eve.clone(), info));
    }

    // do this up here i guess too
    garbo.par_sort_unstable_by(|a, b| a.1.cost.partial_cmp(&b.1.cost).unwrap());

    let mut best_cost = std::f64::INFINITY;
    let mut last_best_time = Instant::now();

    const NUM_ALL_STARS: usize = 4096;
    let mut all_stars = VecDeque::with_capacity(NUM_ALL_STARS);

    let mut f_history = vec![];
    let mut a_history = vec![];

    for i in 0..args.generations {
        if garbo[0].1.cost < best_cost {
            best_cost = garbo[0].1.cost;
            last_best_time = Instant::now();

            if all_stars.len() == NUM_ALL_STARS {
                all_stars.pop_front();
            }
            all_stars.push_back(garbo[0].clone());

            if let Some(best_dir) = args.best_output.as_ref() {
                let path = std::path::PathBuf::from(best_dir).join(format!("{}.wav", i));

                let mut bw = hound::WavWriter::create(path, spec)?;

                for s in &garbo[0].1.audio {
                    bw.write_sample(*s)?;
                }
                bw.finalize()?;
            }
        }

        while garbo.len() > args.population_size {
            // free perf: use retain or whatever to not do an O(n) operation in a loop
            let mut death;
            loop {
                let q = rng.gen_range(48..64);
                death = rng.gen_range(garbo.len() * q / 64..garbo.len());
                if rng.gen_bool((1.0 - garbo[death].1.win_rate())*0.99 + 0.01) { break }
                
            }
            garbo.remove(death);
        }

        let cutoff = garbo[garbo.len() * 2 / 3].1.cost;
        let good_cutoff = garbo[garbo.len() / 4].1.cost;


        debug!("Generation {:?}", i);

        let (tx, rx) = std::sync::mpsc::channel();
        {
            let garbo = &garbo;
            let noisy_seed_info = &noisy_seed_info;
            let r2c = &*r2c;
            let all_stars = &all_stars;
            let seed_spec = &seed_spec;
            rayon::scope(move |s| {
                for _ in 0..512 {
                    let m_tx = tx.clone();
        
                    s.spawn(move |_| {
                        let mut rng = rand::thread_rng();
                            let (gen, par_info, par2_info) = {
                                let mut v;
                                if rng.gen_bool(0.2) {
                                    // use an all-star
                                    v = all_stars.iter().choose(&mut rng).expect("no all-stars? hey now");
                                } else {
                                    loop {
                                        let idx = rng.gen_range(0..garbo.len());
                                        v = &garbo[idx];
                                        if rng.gen_bool((idx as f64 / (garbo.len() as f64 + 1.0)).powf(args.explore)  * (1.0 - v.1.win_rate())) {
                                            continue;
                                        }
                                        break;
                                    }
                                }
                                let (eve, eve_info) = v;

                                if rng.gen_bool(0.05) {
                                    // use an all-star
                                    v = all_stars.iter().choose(&mut rng).expect("no all-stars? hey now");
                                } else {
                                    loop {
                                        let idx = rng.gen_range(0..garbo.len());
                                        v = &garbo[idx];
                                        if rng.gen_bool((idx as f64 / (garbo.len() as f64 + 1.0)).powf(args.explore)  * (1.0 - v.1.win_rate())) {
                                            continue;
                                        }
                                        break;
                                    }
                                }

                                let (adam,  adam_info) = v;

                                let (eve_info, adam_info, eve) = if rng.gen_bool(0.1) {
                                    (eve_info, noisy_seed_info, eve.mutate().befriend(&Animal::spontaneous_generation()))
                                } else if rng.gen_bool(0.25) {
                                    (eve_info, adam_info, eve.mutate())
                                } 
                                else {
                                    (eve_info, adam_info, eve.befriend(adam).mutate())
                                };

                                (eve, eve_info, adam_info)
                            };

                            par_info.trials.fetch_add(1, Ordering::SeqCst);
                            //par2_info.trials.fetch_add(1, Ordering::SeqCst);

                            if !screen(&gen) {
                                return;
                            }

                            let (aud, gas) = gen.feed(&garbo[0].1.audio, Some(&par2_info.audio));
                
                            let spec = compute_spectrogram(&aud, r2c);

                            let sim = compare_spectrograms(&spec, &par_info.spectrogram)
                                .min(compare_spectrograms(&spec, &par2_info.spectrogram)); 
                            let f = compare_spectrograms(&spec, &seed_spec) * (gas as f64);
                            let info = AnimalInfo {
                                cost: f,
                                audio: aud,
                                spectrogram: spec,
                                wins: AtomicUsize::new(0), trials: AtomicUsize::new(0)
                            };
                            if sim > args.parent_child_diff && f < cutoff {
                                if f < good_cutoff {
                                    par_info.wins.fetch_add(1, Ordering::SeqCst);
                                    //par2_info.wins.fetch_add(1, Ordering::SeqCst);
                                }
                                m_tx.send((gen, info)).unwrap();
                            }
                    })
                }
            });
        }

        garbo.extend(rx.iter());

        // again there's no excuse not to do insertion sort here
        // partition_point just always screws me up w/ off-by-ones
        garbo.par_sort_unstable_by(|a, b| a.1.cost.partial_cmp(&b.1.cost).unwrap());
        garbo.dedup_by(|a, b| {
            // cache this. what are you Doing. FREEPERF
            let sim = compare_spectrograms(&a.1.spectrogram, &b.1.spectrogram);
            let same_bucket = sim <= args.parent_child_diff;
            if same_bucket {
                // b will die!
                // add its stats to a's, so they may live on
                // i think this will lose info still if there's more than 2 "same-bucket" things in garbo
                // we shouldn't use dedup_by like this
                // but it's better than nothing
                a.1.trials.fetch_add(b.1.trials.load(Ordering::SeqCst), Ordering::SeqCst);
                a.1.wins.fetch_add(b.1.wins.load(Ordering::SeqCst), Ordering::SeqCst);
            }
            same_bucket
        });
        f_history.push((i as f64, garbo[0].1.cost.ln()));

        // secretly upper 25th %ile now :U
        let avg = garbo[garbo.len() / 4].1.cost;
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
        let (annoying_gen, annoying_max) = f_history[f_history.len().saturating_sub(1024)];
        let chart_loss = Chart::new(datasets)
            .block(Block::default().title("LOSS").borders(Borders::ALL))
            .x_axis(Axis::default()
                .title(Span::styled("Generation", Style::default().fg(Color::LightRed)))
                .style(Style::default().fg(Color::White))
                .bounds([annoying_gen as f64, i as f64]))
            .y_axis(Axis::default()
                .title(Span::styled("Loss", Style::default().fg(Color::LightRed)))
                .style(Style::default().fg(Color::White))
                .bounds([f_history[f_history.len() - 1].1, annoying_max]));


                let data = garbo.iter().enumerate().map(|(i, x)| (i as f64, x.1.win_rate())).collect::<Vec<_>>();
                let datasets = vec![
                    Dataset::default()
                        .name("log winrate")
                        .marker(symbols::Marker::Braille)
                        .graph_type(GraphType::Line)
                        .style(Style::default().fg(Color::LightCyan))
                        .data(&data)
                ];
                let chart_winrate = Chart::new(datasets)
                    .block(Block::default().title("WINRATE").borders(Borders::ALL))
                    .x_axis(Axis::default()
                        .title(Span::styled("Rank", Style::default().fg(Color::LightRed)))
                        .style(Style::default().fg(Color::White))
                        .bounds([0.0, garbo.len() as f64]))
                    .y_axis(Axis::default()
                        .title(Span::styled("Winrate", Style::default().fg(Color::LightRed)))
                        .style(Style::default().fg(Color::White))
                        .bounds([0.0, 1.0]));
        


        terminal.clear()?;
        // inexcusable to just have this ,, sitting here in the main loop lmao
        terminal.draw(|f| {
            let time_since_last_best = Instant::now().duration_since(last_best_time).as_secs_f32();
            // FIXME: colorize ^ 

            let size = f.size();
    
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(0), Constraint::Percentage(20)].as_ref())
                .split(size);

            let chunks_bar = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(6+2), Constraint::Min(0)].as_ref())
                .split(chunks[1]);

            let chunks_graph = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)].as_ref())
                .split(chunks[0]);


            let text = vec![
                Spans::from(vec![
                    Span::styled("geno", Style::default().fg(Color::LightBlue)),
                    Span::styled("ASM", Style::default().fg(Color::Green))
                ]),
                Spans::from(vec![
                    Span::styled("ver: 0.5-", Style::default()),
                    Span::styled("HOMEOBOX", Style::default().fg(Color::LightMagenta).add_modifier(Modifier::ITALIC)),
                ]),
                Spans::from(vec![
                    Span::styled("another bad ", Style::default()),
                    Span::styled("mimir", Style::default().fg(Color::LightCyan)),
                    Span::styled(" idea", Style::default()),
                ]),
                Spans::from(Span::styled("no copyright intended", Style::default().add_modifier(Modifier::ITALIC))),

                Spans::from(vec![
                    Span::styled("questions?", Style::default()),
                ]),
                Spans::from(vec![
                    Span::styled("mimir@xmppwocky.net", Style::default().fg(Color::LightCyan).add_modifier(Modifier::BOLD)),
                ]),
            ];
            let para1 = Paragraph::new(text)
                .block(Block::default().title("ABOUT").borders(Borders::ALL))
                .style(Style::default().fg(Color::White).bg(Color::Black))
                .alignment(Alignment::Center)
                .wrap(Wrap { trim: true });


                let label_style = Style::default().fg(Color::Gray);
                let text = vec![
                    Spans::from(vec![
                        Span::styled("mode: ", label_style),
                        Span::styled("mimicSpectra", Style::default().add_modifier(Modifier::BOLD).fg(Color::DarkGray))
                    ]),

                    Spans::from(Span::raw(format!("gen. {}/{}", i, args.generations))),
                    Spans::from(vec![
                        Span::styled("least unfit: ", label_style),
                        Span::styled(format!("{:+6e}", garbo[0].1.cost), Style::default().add_modifier(Modifier::BOLD))
                    ]),
    
                    Spans::from(vec![
                        Span::styled("25th %ile unfit: ", label_style),
                        // fixme: calc 50th %ile loss and save it separately
                        // instead of exponentiating the log-loss in ahistory
                        Span::styled(format!("{:+6e}", a_history[a_history.len() - 1].1.exp()), Style::default().add_modifier(Modifier::BOLD))
                    ]),

                    Spans::from(vec![
                        Span::styled("time since last best: ", label_style),
                        Span::styled(format!("{:8.1}s", time_since_last_best), Style::default().add_modifier(Modifier::BOLD))
                    ]),

                    Spans::from(vec![
                        Span::styled("num. animals: ", label_style),
                        Span::styled(format!("{}", garbo.len()), Style::default().add_modifier(Modifier::BOLD))
                    ]),
                    
                    Spans::from(vec![
                        Span::styled("num. threads: ", label_style),
                        Span::styled(format!("{}", rayon::current_num_threads()), Style::default().add_modifier(Modifier::BOLD))
                    ]),
                        
                    Spans::from(vec![
                        Span::styled("vibe alignment: ", label_style),
                        Span::styled("aethereal", Style::default().add_modifier(Modifier::BOLD).fg(Color::Magenta))
                    ]),
    
                ];
                let para2 = Paragraph::new(text)
                    .block(Block::default().title("STATS").borders(Borders::ALL))
                    .style(Style::default().fg(Color::White).bg(Color::Black))
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: true });
    
                
            f.render_widget(chart_loss, chunks_graph[0]);
            f.render_widget(chart_winrate, chunks_graph[1]);
            f.render_widget(para1, chunks_bar[0]);
            f.render_widget(para2, chunks_bar[1]);
        })?;
    }

    Ok(())
}
