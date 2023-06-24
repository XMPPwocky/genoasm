use rand::Rng;

use crate::{animal::Animal, util::normalize_audio, vm::*};

#[derive(Clone)]
pub struct Genoasm {
    pub instructions: Box<[Instruction; NUM_INSTRUCTIONS]>,
    pub lut: Box<[i16; LUT_SIZE]>,
}
impl Genoasm {
    pub fn feed(&self, audio: &[i16]) -> (Vec<i16>, u64) {
        let out_areg: Vec<i16> = std::iter::repeat(0).take(audio.len()).collect();

        let mut aregs: [Vec<i16>; NUM_REGISTERS as usize] = Default::default();
        aregs[0] = audio.to_vec();
        aregs[1] = out_areg;
        aregs[2] = self.lut.to_vec();
        for areg in &mut aregs[2..] {
            areg.push(0);
        }

        let gas_limit = 128 * audio.len() as u64;

        let mut vm = VmState::new(aregs, gas_limit);

        let mut status = VmRunResult::Continue;
        while status == VmRunResult::Continue {
            status = vm.run_insn(&self.instructions[vm.pc as usize]);
        }
        /*if status == VmRunResult::OutOfGas {
            // penalize gas guzzlers - return 0... inefficient lol
            return std::iter::repeat(0).take(audio.len()).collect();
        }*/
        let f = vm.aregs[1].clone(); // useless clone lol
        (
            normalize_audio(&f),
            1 
            //(u64::BITS - (gas_limit - vm.gas_remaining()).leading_zeros()) as u64 // hack dont scale this here you doof
        )
    }
}

impl Animal for Genoasm {
    fn spontaneous_generation() -> Self {
        let mut rng = rand::thread_rng();
        let mut instructions = Box::new([Instruction([0; 4]); NUM_INSTRUCTIONS]);
        for insn in &mut *instructions {
            insn.0[0] = rng.gen_range(0..=Opcode::Filter as u8);
            for q in &mut insn.0[1..] {
                if rng.gen_bool(0.975) {
                    *q = rng.gen();
                } else {
                    let magics = [0u8, 1, 0x7F, 0x80, 0xFF];
                    *q = magics[rng.gen_range(0..magics.len())];
                }
            }
        }

        let mut lut = Box::new([0; LUT_SIZE]);
        for e in &mut *lut {
            *e = 0; // i guess? maybe less prone to generating random noise
        }

        Genoasm { instructions, lut }
    }

    fn befriend(&self, friend: &Self) -> Self {
        let mut rng = rand::thread_rng();

        let insn_split_point = rng.gen_range(0..NUM_INSTRUCTIONS);
        let insn_end = insn_split_point + rng.gen_range(0..NUM_INSTRUCTIONS - insn_split_point);
        let lut_split_point = rng.gen_range(0..LUT_SIZE);
        let lut_end = lut_split_point + rng.gen_range(0..LUT_SIZE - lut_split_point);

        
        let mut instructions = self.instructions.clone();
        let mut lut = self.lut.clone();

        instructions[insn_split_point..insn_end].copy_from_slice(&friend.instructions[insn_split_point..insn_end]);
        lut[lut_split_point..lut_end].copy_from_slice(&friend.lut[lut_split_point..lut_end]);

        Genoasm { instructions, lut }
    }

    fn mutate(&self) -> Self {
        let mut ant = self.clone();
        let mut rng = rand::thread_rng();

        // mutate instructions
        for _ in 0..(1<<rng.gen_range(8..=12)) {
            match rng.gen_range(0..=2) {
                0 => {
                    let idx = rng.gen_range(0..NUM_INSTRUCTIONS);
                    let offset = rng.gen_range(0..4);
                    let shift = rng.gen_range(0..8);
                    ant.instructions[idx].0[offset] ^= 1 << shift;
                },
                1 => {
                    let idx = rng.gen_range(0..NUM_INSTRUCTIONS);
                    let offset = rng.gen_range(0..4);
                    let new = rng.gen();
                    ant.instructions[idx].0[offset] = new;
                }
                _ => {
                    let idx = rng.gen_range(0..NUM_INSTRUCTIONS);
                    let offset = rng.gen_range(0..4);
                    let add = rng.gen_range(-16..=16);
                    ant.instructions[idx].0[offset] =
                        ant.instructions[idx].0[offset].wrapping_add_signed(add);
                }
            }
        }

        // mutate LUT
        for _ in 0..768 {
            match rng.gen_range(0..=3) {
                0 => {
                    let idx = rng.gen_range(0..LUT_SIZE);
                    let shift = rng.gen_range(0..8);
                    ant.lut[idx] ^= 1 << shift;
                }
                1 => {
                    let idx = rng.gen_range(0..LUT_SIZE);
                    let add = rng.gen_range(-16..=16);
                    ant.lut[idx] = ant.lut[idx].wrapping_add(add);
                }
                2 => {
                    let idx = rng.gen_range(0..LUT_SIZE - 1);
                    ant.lut[idx + 1] = ant.lut[idx];
                }
                _ => {
                    let idx = rng.gen_range(0..LUT_SIZE - 1);
                    ant.lut[idx] = ant.lut[idx + 1];
                }
            }
        }

        ant
    }
}
