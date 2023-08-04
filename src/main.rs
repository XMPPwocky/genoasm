use core::sync::atomic::{AtomicUsize, Ordering};
use std::{io::Write, time::Instant};

use animal::{genoasm, Animal};
use flate2::write::GzEncoder;
use flate2::Compression;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::{seq::IteratorRandom, Rng};
use rayon::prelude::*;
use realfft::RealFftPlanner;
use std::collections::VecDeque;
use std::sync::atomic::Ordering::SeqCst;
use tui::{
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    symbols,
    text::{Span, Spans},
    widgets::{Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Wrap},
};

use tracing::debug;
const TABOO_LEN: usize = 256;

const SAMPLE_RATE: f32 = 22050.0; // 44100.0;

use util::normalize_audio;

use crate::similarity::{spectrogram_error_vector, ErrorVector};

use crate::{
    animal::{genoasm::Genoasm, AnimalInfo},
    similarity::{compare_spectrograms, compute_spectrogram},
};

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
    #[arg(short, long, default_value_t = 1024)]
    population_size: usize,

    /// Generations.
    #[arg(short, long, default_value_t = usize::MAX)]
    generations: usize,

    /// Min difference between parent and child.
    ///
    /// This probably should be higher for longer inputs...
    #[arg(short, long, default_value_t = 0.0)]
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
    #[arg(short, long, default_value_t = 1.0)]
    explore: f64,

    #[arg(short, long, default_value_t = 1.0)]
    taboo_similarity: f64,

    #[arg(short, long, default_value_t = 0.01)]
    taboo_pop_odds: f64,
}

struct Stats {
    trials_count: AtomicUsize,

    taboo_rejects_count: AtomicUsize,
}

fn screen(gen: &genoasm::Genoasm) -> bool {
    const SCREEN_LEN: usize = 4096;
    let gas_limit = SCREEN_LEN as u64 * 96;

    let (_v, v_gas) = gen.feed(&[0x7714; SCREEN_LEN], None, gas_limit);
    if v_gas < 4096 { return false; }

    /* 

    if v.iter().skip(SCREEN_LEN/2).filter(|&&x| x != 0).count() < SCREEN_LEN/4 { return false; }

    let (v2, _) = gen.feed(&[0x42; SCREEN_LEN], None, gas_limit);
    if v == v2 {
        return false;
    }


    let (v3, _) = gen.feed(&[0x7714; SCREEN_LEN], Some(&[0x7734; SCREEN_LEN]), gas_limit);
    if v == v3 {
        return false;
    }*/

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
        sample_rate: SAMPLE_RATE as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut population: Vec<(Genoasm, AnimalInfo)> = vec![];
    let mut taboo = VecDeque::with_capacity(TABOO_LEN);

    let stats = Stats {
        trials_count: AtomicUsize::new(0),
        taboo_rejects_count: AtomicUsize::new(0),
    };

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
    for _m in seed.iter() {
        //j = m;
        //if rng.gen_bool(0.5) { j = m; }

        //if i % 22050 < 8192 {
        //    noisy_seed.push(j);
        //} else {
        noisy_seed.push(rng.gen());
        //}
        ////noisy_seed.push(rng.gen());
    }

    //let noisy_seed = mix_audio(&seed, &noise, 0.1);
    let noisy_seed = normalize_audio(&noisy_seed);
    let noisy_seed_spec = compute_spectrogram(&noisy_seed, &*r2c);
    let noisy_seed_err = spectrogram_error_vector(&seed_spec, &noisy_seed_spec);
    let f = noisy_seed_err.sum();

    let gas_limit = 128 * seed.len() as u64;
    let noisy_seed_info = AnimalInfo {
        cost: f,
        audio: noisy_seed.clone(),
        spectrogram: noisy_seed_spec,
        parent_sims: (f64::INFINITY, f64::INFINITY),
        error_vector: noisy_seed_err,
        error_vector_sum: f,
        wins: AtomicUsize::new(0),
        trials: AtomicUsize::new(0),
        gas: gas_limit,
    };

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

        let (aud, gas) = eve.feed(&noisy_seed, None, gas_limit);
        let spec = compute_spectrogram(&aud, &*r2c);
        let e = spectrogram_error_vector(&spec, &noisy_seed_info.spectrogram);
        let f = e.sum();
        
        let info = AnimalInfo {
            cost: f,
            error_vector: e,
            error_vector_sum: f,
            spectrogram: spec,
            parent_sims: (0.0, 0.0),
            gas,
            audio: aud,
            wins: AtomicUsize::new(0),
            trials: AtomicUsize::new(0),
        };
        population.push((eve.clone(), info));
    }

    // do this up here i guess too
    population.par_sort_unstable_by(|a, b| a.1.cost.partial_cmp(&b.1.cost).unwrap());

    let eves = population.clone();

    let mut best_cost = std::f64::INFINITY;
    let mut last_best_time = Instant::now();

    const NUM_ALL_STARS: usize = 2048;
    let mut all_stars = VecDeque::with_capacity(NUM_ALL_STARS);
    all_stars.extend(eves.iter().cloned());

    let mut f_history = vec![];
    let mut a_history = vec![];
    let mut global_error = ErrorVector::ones(population[0].1.error_vector.len());
    global_error.normalize();

    for current_generation in 0..args.generations {
        if rng.gen_bool(0.00005) {
            // Meteor strike!
            taboo.clear();
            taboo.extend(
                population
                    .drain(..)
                    .map(|(_animal, info)| info.spectrogram)
                    .take(TABOO_LEN),
            );
            population = eves.clone();
        }
        let mut ugh = 0;
        let l = taboo.len();

        // split  to fn FIXME
        taboo.retain(|_| {
            let idx = ugh as f64 / l as f64;
            ugh += 1;
            !rng.gen_bool(args.taboo_pop_odds * (1.0 - idx))
        });
        while taboo.len() > TABOO_LEN {
            ugh = 0;
            taboo.retain(|_| {
                let idx = ugh as f64 / l as f64;
                ugh += 1;
                !rng.gen_bool(args.taboo_pop_odds * (1.0 - idx))
            });
        }

        /*{
            let x = &mut population[0];

            x.0.simplify(&x.1.audio, x.1.gas, &noisy_seed, None);
        }*/

        let best = population
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1 .1
                    .error_vector
                    .sum()
                    .partial_cmp(&b.1 .1.error_vector.sum())
                    .unwrap()
            })
            .unwrap()
            .0;

        if population[best].1.error_vector.sum() < best_cost {
            if let Some(best_dir) = args.best_output.as_ref() {
                let path =
                    std::path::PathBuf::from(best_dir).join(format!("{}.wav", current_generation));

                let mut bw = hound::WavWriter::create(path, spec)?;

                for s in &population[best].1.audio {
                    bw.write_sample(*s)?;
                }
                bw.finalize()?;

                let path =
                    std::path::PathBuf::from(best_dir).join(format!("{}.dna", current_generation));
                let f_raw = std::io::BufWriter::new(std::fs::File::create(path)?);
                let mut f = GzEncoder::new(f_raw, Compression::default());
                for (i, insn) in population[best].0.instructions.iter().enumerate() {
                    insn.write(&mut f, i as u16)?;
                }
                f.write_all(b"\n.lut_start\n")?;
                for elem in population[best].0.lut.iter() {
                    f.write_all(&elem.to_le_bytes())?;
                }
                f.finish()?;
            }
            best_cost = population[best].1.error_vector.sum();
            last_best_time = Instant::now();

            if all_stars.len() == NUM_ALL_STARS {
                let idx = rng.gen_range(0..all_stars.len());
                all_stars[idx] = population[best].clone();
            } else {
                all_stars.push_back(population[best].clone());
            }

            if taboo.len() >= TABOO_LEN {
                taboo.pop_back();
            }

            /*population.retain(|(_animal, info)| {
                compare_spectrograms(&spec, &info.spectrogram) >= f64::min(info.parent_sims.0, info.parent_sims.1)
            });*/
            while population.len() < 32 {
                let h = rng.gen_range(0..eves.len());
                population.push(eves[h].clone());
            }
            //taboo.push_front(spec);
            //population.drain(1..population.len() / 4); // kill all but the best rockstars
            //population.truncate(population.len() * 3 / 4); // make room for non-taboo explores
            //}

            // update global error
            let mut gen_error = population[best].1.error_vector.clone();

            /*for (_animal, info) in &population[1..population.len()] {
                gen_error += &info.error_vector;
            }*/
            gen_error.normalize();
            gen_error.scale(0.005);
            global_error.scale(0.995);
            global_error += &gen_error;
            //global_error.normalize();
            
    
            /*for (_animal, info) in &mut population {
                info.cost = /*info.error_vector.dot(&global_error) + */ info.error_vector.sum();
            }*/
        }




        let cutoff = population[population.len() - 1].1.cost;


        population.par_sort_unstable_by(|a, b| a.1.cost.partial_cmp(&b.1.cost).unwrap());

        let weights = population.iter().enumerate()
            .map(|(idx, animal)| {
                let wr_mod = animal.1.win_rate();
                let gas_mod = (animal.1.gas as f64 + 1.0).powf(-0.5);
                (1.0 - (idx as f64 / (population.len() as f64 + 1.0))
                                            .powf(args.explore))
                                            * wr_mod
                                            * gas_mod
            }).collect::<Vec<_>>();
        let windex = WeightedIndex::new(&weights).unwrap();

        debug!("Generation {:?}", current_generation);

        let (tx, rx) = std::sync::mpsc::channel();
        {
            let population = &population;
            let _noisy_seed_info = &noisy_seed_info;
            let r2c = &*r2c;
            let all_stars = &all_stars;
            let seed_spec = &seed_spec;
            let taboo = &taboo;
            let stats = &stats;
            let noisy_seed = &noisy_seed;
            let _global_error = &global_error;
            let windex = &windex;

            rayon::scope(move |s| {
                for _ in 0..1024 {
                    let m_tx = tx.clone();

                    s.spawn(move |_| {
                        let mut rng = rand::thread_rng();
                        let (gen, par_info, par2_info) = {
                            let v = if taboo.is_empty() && rng.gen_bool(
                                0.05
                            ) {
                                // use an all-star
                                all_stars
                                    .iter()
                                    .choose(&mut rng)
                                    .expect("no all-stars? hey now")
                            } else {
                                &population[windex.sample(&mut rng)]
                            };
                            let (eve, eve_info) = v;

                            let v = if taboo.is_empty() && rng.gen_bool(
                                0.05
                            ) {
                                // use an all-star
                                all_stars
                                    .iter()
                                    .choose(&mut rng)
                                    .expect("no all-stars? hey now")
                            } else {
                                &population[windex.sample(&mut rng)]
                            };

                            let (adam, adam_info) = v;

                            let (eve_info, adam_info, eve) = if rng.gen_bool(0.1) {
                                (
                                    eve_info,
                                    adam_info,
                                    eve.mutate(),
                                )
                            } else {
                                (eve_info, adam_info, eve.befriend(adam).mutate())
                            };

                            (eve, eve_info, adam_info)
                        };

                        stats.trials_count.fetch_add(1, SeqCst);

                        par_info.trials.fetch_add(1, Ordering::SeqCst);
                        //par2_info.trials.fetch_add(1, Ordering::SeqCst);

                        if !screen(&gen) {
                            return;
                        }

                        let (aud, gas) = gen.feed(noisy_seed, None, gas_limit); //&audio_parent, Some(&par2_info.audio));
                                                                                // silly hack....
                                                                                //if aud[aud.len() - 1] == 0 { return; }

                        let spec = compute_spectrogram(&aud, r2c);

                        let (sim1, sim2) = (
                            compare_spectrograms(&spec, &par_info.spectrogram),
                            compare_spectrograms(&spec, &par2_info.spectrogram),
                        );

                        let e = spectrogram_error_vector(&spec, seed_spec);
                        let e_sum = e.sum();
                        let f = e_sum; // e.dot(global_error) + e_sum;
                        let parent_wins = par_info.wins.load(Ordering::SeqCst) + 1;
                        let parent_trials = par_info.wins.load(Ordering::SeqCst) + 1;
                        let parent_winrate = parent_wins as f64 / parent_trials as f64;
                        let fake_trials = (parent_trials / 3).clamp(1, 1024);
                        let fake_wins = (parent_winrate * fake_trials as f64) as usize;
                        
                        let info = AnimalInfo {
                            cost: f,
                            error_vector: e,
                            error_vector_sum: e_sum,
                            parent_sims: (sim1, sim2),
                            gas,
                            audio: aud,
                            spectrogram: spec,
                            wins: AtomicUsize::new(fake_wins),
                            trials: AtomicUsize::new(fake_trials),
                        };
                        if f64::min(sim1, sim2) > args.parent_child_diff && f < cutoff {
                            // regardless of taboo, credit parent(s)
                            // medidate on min/max switch here
                            if e_sum < f64::min(par_info.error_vector_sum, par2_info.error_vector_sum) {
                                par_info.wins.fetch_add(1, Ordering::SeqCst);
                               // par2_info.wins.fetch_add(1, Ordering::SeqCst);
                            }

                            let taboo_sim = taboo
                                .iter()
                                .map(|x| compare_spectrograms(&info.spectrogram, x))
                                .min_by(|x, y| x.partial_cmp(y).unwrap())
                                .unwrap_or(std::f64::INFINITY);

                            // a
                            if taboo_sim < f64::min(sim1, sim2) {
                                stats.taboo_rejects_count.fetch_add(1, SeqCst);
                            } else {
                                m_tx.send((gen, info)).unwrap();
                            }
                        }
                    })
                }
            });
        }

        for (animal, info) in rx.iter() {
            let pos = population.partition_point(|x| x.1.cost < info.cost);
            if pos == population.len() {
                continue; // obsoleted by a previous find this generation
            }
            if population.len() >= args.population_size {
                population.pop().unwrap();
            }

            population.insert(pos, (animal, info));
        }

        // again there's no excuse not to do insertion sort here
        // partition_point just always screws me up w/ off-by-ones
        //population.par_sort_unstable_by(|a, b| a.1.cost.partial_cmp(&b.1.cost).unwrap());
        /*let mut seens: HashMap<Box<[vm::Instruction; NUM_INSTRUCTIONS]>, usize> = HashMap::new();

        // perf: goofy clone here
        for (i, elem) in population.iter().enumerate() {
            if let Some(&prev) = seens.get(&elem.0.instructions) {
                // b will die!
                // add its stats to a's, so they may live on
                // i think this will lose info still if there's more than 2 "same-bucket" things in garbo
                // we shouldn't use dedup_by like this
                // but it's better than nothing
                population[prev]
                    .1
                    .trials
                    .fetch_add(elem.1.trials.load(Ordering::SeqCst), Ordering::SeqCst);
                population[prev]
                    .1
                    .wins
                    .fetch_add(elem.1.wins.load(Ordering::SeqCst), Ordering::SeqCst);
            } else {
                seens.insert(elem.0.instructions.clone(), i);
            }
        }*/

        f_history.push((current_generation as f64, best_cost));

        // secretly upper 25th %ile now :U
        let avg = population[population.len() / 4].1.cost;
        a_history.push((current_generation as f64, avg.ln()));

        /*let mut ugh = 0;
        population.retain(|elem| {
            let agh = ugh;
            ugh = ugh + 1;
            seens.get(&elem.0.instructions) == Some(&agh)
        });*/

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
        let (annoying_gen, annoying_max) = f_history[f_history.len().saturating_sub(2048)];
        let chart_loss = Chart::new(datasets)
            .block(Block::default().title("LOSS").borders(Borders::ALL))
            .x_axis(
                Axis::default()
                    .title(Span::styled(
                        "Generation",
                        Style::default().fg(Color::Magenta),
                    ))
                    .style(Style::default().fg(Color::White))
                    .bounds([annoying_gen, current_generation as f64]),
            )
            .y_axis(
                Axis::default()
                    .title(Span::styled("Loss", Style::default().fg(Color::Magenta)))
                    .style(Style::default().fg(Color::White))
                    .bounds([f_history[f_history.len() - 1].1, annoying_max]),
            );

        let data = weights
            .iter()
            .enumerate()
            .map(|(i, x)| (i as f64, *x))
            .collect::<Vec<_>>();

        let max_wr = data
            .iter()
            .cloned()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .1;
        let datasets = vec![Dataset::default()
            .name("log winrate")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::LightCyan))
            .data(&data)];
        let chart_winrate = Chart::new(datasets)
            .block(Block::default().title("WINRATE").borders(Borders::ALL))
            .x_axis(
                Axis::default()
                    .title(Span::styled("Rank", Style::default().fg(Color::LightRed)))
                    .style(Style::default().fg(Color::White))
                    .bounds([0.0, population.len() as f64]),
            )
            .y_axis(
                Axis::default()
                    .title(Span::styled(
                        "Winrate",
                        Style::default().fg(Color::LightRed),
                    ))
                    .style(Style::default().fg(Color::White))
                    .bounds([0.0, max_wr]),
            );

        terminal.clear()?;
        // inexcusable to just have this ,, sitting here in the main loop lmao
        terminal.draw(|f| {
            let time_since_last_best = Instant::now().duration_since(last_best_time).as_secs_f32();
            // FIXME: colorize ^

            let size = f.size();

            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(0), Constraint::Percentage(33)].as_ref())
                .split(size);

            let chunks_bar = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(6 + 2), Constraint::Min(0)].as_ref())
                .split(chunks[1]);

            let chunks_graph = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)].as_ref())
                .split(chunks[0]);

            let text = vec![
                Spans::from(vec![
                    Span::styled("geno", Style::default().fg(Color::LightBlue)),
                    Span::styled("ASM", Style::default().fg(Color::Green)),
                ]),
                Spans::from(vec![
                    Span::styled("ver: 0.13-", Style::default()),
                    Span::styled(
                        "CNIDARIA",
                        Style::default()
                            .fg(Color::LightYellow)
                            .add_modifier(Modifier::ITALIC),
                    ),
                ]),
                Spans::from(vec![
                    Span::styled("another bad ", Style::default()),
                    Span::styled("mimir", Style::default().fg(Color::LightCyan)),
                    Span::styled(" idea", Style::default()),
                ]),
                Spans::from(Span::styled(
                    "no copyright intended",
                    Style::default().add_modifier(Modifier::ITALIC),
                )),
                Spans::from(vec![Span::styled("questions?", Style::default())]),
                Spans::from(vec![Span::styled(
                    "mimir@xmppwocky.net",
                    Style::default()
                        .fg(Color::LightCyan)
                        .add_modifier(Modifier::BOLD),
                )]),
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
                    Span::styled(
                        "mimicSpectra",
                        Style::default()
                            .add_modifier(Modifier::BOLD)
                            .fg(Color::DarkGray),
                    ),
                ]),
                Spans::from(Span::raw(format!(
                    "gen. {}/{}",
                    current_generation, args.generations
                ))),
                Spans::from(vec![
                    Span::styled("least unfit: ", label_style),
                    Span::styled(
                        format!("{:1.8e}", best_cost),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                ]),
                Spans::from(vec![
                    Span::styled("25th %ile unfit-rel: ", label_style),
                    // fixme: calc 50th %ile loss and save it separately
                    // instead of exponentiating the log-loss in ahistory
                    Span::styled(
                        format!("{:1.8e}", a_history[a_history.len() - 1].1.exp()),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                ]),
                Spans::from(vec![
                    Span::styled("best gas: ", label_style),
                    // fixme: calc 50th %ile loss and save it separately
                    // instead of exponentiating the log-loss in ahistory
                    Span::styled(
                        format!("{}", population[0].1.gas),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                ]),
                Spans::from(vec![
                    Span::styled("time since last best: ", label_style),
                    Span::styled(
                        format!("{:6.1}s", time_since_last_best),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                ]),
                Spans::from(vec![
                    Span::styled("num. threads: ", label_style),
                    Span::styled(
                        format!("{}", rayon::current_num_threads()),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                ]),
                Spans::from(vec![
                    Span::styled("num. trials: ", label_style),
                    Span::styled(
                        format!("{}", stats.trials_count.load(SeqCst)),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                ]),
                Spans::from(vec![
                    Span::styled("num. animals: ", label_style),
                    Span::styled(
                        format!("{}", population.len()),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                ]),
                Spans::from(vec![
                    Span::styled("num. taboos: ", label_style),
                    Span::styled(
                        format!("{}", taboo.len()),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                ]),
                Spans::from(vec![
                    Span::styled("taboo rejects: ", label_style),
                    Span::styled(
                        format!("{}", stats.taboo_rejects_count.load(SeqCst)),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                ]),
                Spans::from(vec![
                    Span::styled("vibe alignment: ", label_style),
                    Span::styled(
                        "void",
                        Style::default()
                            .add_modifier(Modifier::BOLD)
                            .fg(Color::Magenta),
                    ),
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
