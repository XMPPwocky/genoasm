use core::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use rand::{Rng, rngs::ThreadRng, distributions::WeightedIndex};
use serde::{Deserialize, Serialize};

use crate::{animal::Animal, util::normalize_audio, vm::*};
use serde_with::serde_as;

#[serde_as]
#[derive(Clone, Deserialize, Serialize)]
pub struct Genoasm {
    #[serde_as(as = "Box<[_; NUM_INSTRUCTIONS]>")]
    pub instructions: Box<[Instruction; NUM_INSTRUCTIONS]>,
    #[serde_as(as = "Box<[_; LUT_SIZE]>")]
    pub lut: Box<[i16; LUT_SIZE]>,
}
impl Genoasm {
    pub fn feed(&self, audio: &[i16], audio2: Option<&[i16]>, gas_limit: u64) -> (Vec<i16>, u64, u64) {
        let out_areg: Vec<i16> = std::iter::repeat(0).take(audio.len()).collect();

        let mut aregs: [Vec<i16>; NUM_REGISTERS as usize] = Default::default();
        aregs[0] = audio.to_vec();
        aregs[1] = audio2
            .map(|aud2| aud2.to_vec())
            .unwrap_or_else(|| audio.to_vec());
        aregs[2] = self.lut.to_vec();
        aregs[3] = out_areg;

        for areg in &mut aregs[4..] {
            // this is nasty but easier than thinking about the zero-length case later :U
            areg.push(0);
        }

        let mut vm = VmState::new(aregs, gas_limit);

        // all go
        let mut status = VmRunResult::Continue;
        while status == VmRunResult::Continue {
            status = vm.run_insn(&self.instructions[vm.pc as usize % NUM_INSTRUCTIONS]);
        }
        let mut hasher = DefaultHasher::new();

        for x in self.instructions.iter()
            .cloned()
            .enumerate()
            .map(|(idx, insn)| {
                if vm.covmap_get(idx as u16) { Some(insn) } else { None }
            }) {
                x.hash(&mut hasher);
            }
            

        let f = normalize_audio(&vm.aregs[3]); // useless clone lol
        (
            f,
            gas_limit - vm.gas_remaining(),
            hasher.finish()
        )
    }

    pub fn simplify(
        &mut self,
        expected: &[i16],
        gas_limit: u64,
        in_audio: &[i16],
        in_audio2: Option<&[i16]>,
    ) {
        assert_eq!(expected.len(), in_audio.len());

        let old_instructions = self.instructions.clone();

        let mut block_size = 64;

        while block_size > 0 {
            for i in (0..self.instructions.len()).step_by(block_size) {
                let mut changed = false;
                for j in i..(i + block_size).clamp(0, self.instructions.len()) {
                    changed |= self.instructions[j].get_opcode() != Some(Opcode::Die);
                    self.instructions[j].0[0] = Opcode::Die as u8;
                }
                if changed {
                    let out = self.feed(in_audio, in_audio2, gas_limit).0;

                    if out == expected {
                        continue;
                    }
                } else {
                    continue;
                }
                // revert
                for j in i..(i + block_size).clamp(0, self.instructions.len()) {
                    self.instructions[j].0[0] = old_instructions[j].0[0];
                }
            }

            block_size /= 2;
        }

        // unsled NOPs
        let mut streak = 0;
        for i in 0..self.instructions.len() {
            if self.instructions[i].0[0] == Opcode::Nop as u8 {
                streak += 1;
            } else {
                for j in 0..streak {
                    let offset = j; // + 1;

                    self.instructions[i - offset - 1].0[0] = Opcode::Jmp as u8;
                    self.instructions[i - offset - 1].0[2] = (offset & 0xFF) as u8;
                    self.instructions[i - offset - 1].0[3] = (offset >> 8) as u8;
                }
                streak = 0;
            }
        }

        // jump forwarding

        /*let mut changed = true;
        while changed {
            changed = false;

            for i in 0..self.instructions.len() {
                if self.instructions[i].get_opcode() == Some(Opcode::Jmp) {
                    let offset = self.instructions[i].get_operand_imm16(1);
                    let target = (i + offset as usize + 1) % NUM_INSTRUCTIONS;
                    if target == i {
                        // just die instead of infinite looping
                        self.instructions[i].0[0] = Opcode::Die as u8;
                        changed = true;
                    } else if self.instructions[target].get_opcode() == Some(Opcode::Jmp) {
                        // we have a jmp to a jmp

                        let d_offset = self.instructions[target].get_operand_imm16(1);
                        let d_target = (target + d_offset as usize + 1) % NUM_INSTRUCTIONS;

                        let new_offset = (d_target.wrapping_sub(i + 1)) % NUM_INSTRUCTIONS;
                        assert_eq!((i + new_offset + 1) % NUM_INSTRUCTIONS, d_target);

                        changed |= offset != new_offset as u16;
                        self.instructions[i].set_operand_imm16(1, new_offset as u16);

                    }
                }
            }
        }*/

        // double check safety
        // HACK: try adn detect if original timed out here
        /*if gas_limit < (in_audio.len() * 256) as u64 {
            let out = self.feed(in_audio, in_audio2, gas_limit).0;
            assert_eq!(&out, &expected);
        }*/
    }
}

fn random_instruction(rng: &mut ThreadRng) -> Instruction {
    let mut insn = Instruction([0; 4]);
    insn.0[0] = rng.gen_range(0..=Opcode::Filter as u8);
    for q in &mut insn.0[1..] {
        if rng.gen_bool(0.8) {
            *q = rng.gen();
        } else {
            *q = MAGICS[rng.gen_range(0..MAGICS.len())];
        }
    }

    insn
}
const MAGICS: &[u8] = &[0u8, 1, 2, 3, 4, 0x7F, 0x80, 0xFF];
impl Animal for Genoasm {
    fn spontaneous_generation() -> Self {
        let mut rng = rand::thread_rng();
        let mut instructions = Box::new([Instruction([0; 4]); NUM_INSTRUCTIONS]);
        for insn in &mut *instructions {
            *insn = random_instruction(&mut rng);
        }

        let mut lut = Box::new([0; LUT_SIZE]);
        for e in &mut *lut {
            // if we felt like it,
            // we could use sample_iter here
            // which might be faster, somehow? or just a wrapper
            *e = 0; // rng.gen();
        }

        Genoasm { instructions, lut }
    }

    fn befriend(&self, friend: &Self) -> Self {
        let mut rng = rand::thread_rng();
        let lut_split_point = rng.gen_range(0..LUT_SIZE);
        let lut_end = lut_split_point + rng.gen_range(0..LUT_SIZE - lut_split_point);

        let mut instructions = self.instructions.clone();
        let mut lut = self.lut.clone();

        lut[lut_split_point..lut_end].copy_from_slice(&friend.lut[lut_split_point..lut_end]);

        {
            let insn_split_point = rng.gen_range(0..NUM_INSTRUCTIONS);
            let insn_splice_len =
                rng.gen_range(0..NUM_INSTRUCTIONS - insn_split_point) >> rng.gen_range(0..4);

            let spin = if rng.gen_bool(0.95) {
                0
            } else {
                rng.gen_range(0..NUM_INSTRUCTIONS)
            };

            for i in 0..insn_splice_len {
                instructions[insn_split_point + i] =
                    instructions[(insn_split_point + spin + i) % NUM_INSTRUCTIONS];
            }
        }
        Genoasm { instructions, lut }
    }

    fn mutate(&self) -> Self {
        let mut ant = self.clone();
        let mut rng = rand::thread_rng();

        // mutate instructions
        let windex = WeightedIndex::new(vec![2.0, 1.0, 0.5, 0.1, 0.1, 1.0]).unwrap();

        for _ in 0..(1 << rng.gen_range(3..10)) {
            match rng.sample(&windex) {
                0 => {
                    let idx = rng.gen_range(0..NUM_INSTRUCTIONS);
                    let offset = rng.gen_range(1..4);
                    let shift = rng.gen_range(0..8);
                    ant.instructions[idx].0[offset] ^= 1 << shift;
                }
                1 => {
                    // randomize u8 or use magic
                    let idx = rng.gen_range(0..NUM_INSTRUCTIONS);
                    let offset = rng.gen_range(1..4);
                    let new = if rng.gen_bool(0.5) {
                        rng.gen()
                    } else {
                        MAGICS[rng.gen_range(0..MAGICS.len())]
                    };
                    ant.instructions[idx].0[offset] = new;
                }
                2 => {
                    // randomize opcode
                    let idx = rng.gen_range(0..NUM_INSTRUCTIONS);
                    ant.instructions[idx].0[0] = rng.gen_range(0..=Opcode::Filter as u8);
                },
                3 => {
                    // delete a random instruction
                    let pos = rng.gen_range(0..NUM_INSTRUCTIONS - 1);
                    for i in pos..NUM_INSTRUCTIONS - 1 {
                        ant.instructions[i] = ant.instructions[i+1];
                    }
                },
                4 => {
                    // insert some instruction
                    let pos = rng.gen_range(0..NUM_INSTRUCTIONS);
                    
                    // move instructions after pos forwards
                    for i in (pos + 1..NUM_INSTRUCTIONS).rev() {
                        ant.instructions[i] = ant.instructions[i - 1];
                    }
                    ant.instructions[pos] = random_instruction(&mut rng);
                }
                5 => {
                    // random operand add/sub
                    let idx = rng.gen_range(0..NUM_INSTRUCTIONS);
                    let offset = rng.gen_range(1..4);
                    let add = rng.gen_range(-8..=8);
                    ant.instructions[idx].0[offset] =
                        ant.instructions[idx].0[offset].wrapping_add_signed(add);
                }
                _ => unreachable!()
            }
        }

        // mutate LUT
        for _ in 0..32 {
            match rng.gen_range(0..=3) {
                0 => {
                    let idx = rng.gen_range(0..LUT_SIZE);
                    let shift = rng.gen_range(0..16);
                    ant.lut[idx] ^= 1 << shift;
                }
                1 => {
                    let idx = rng.gen_range(0..LUT_SIZE);
                    let add = rng.gen_range(-1024..=1024);
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
