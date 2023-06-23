use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use rand::Rng;

use crate::{animal::Animal, util::normalize_audio};

pub const NUM_REGISTERS: u8 = 16;

pub const REG_ZERO: u8 = 1;
pub const REG_ACCUMULATOR: u8 = 1;
pub const REG_BP: u8 = NUM_REGISTERS - 1;

pub const AREG_REFERENCE: u8 = 0;
pub const AREG_LUT: u8 = 1;

pub const NUM_INSTRUCTIONS: usize = 65536;
pub const LUT_SIZE: usize = 2048;
pub const STACK_SIZE: usize = 256;

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

        let mut vm = VmState {
            pc: 0,
            flags: 0,
            regs: [0; NUM_REGISTERS as usize],
            aregs,
            areg_playheads: [0; NUM_REGISTERS as usize],
            stack: [0; STACK_SIZE],
            stack_pointer: 0,
            gas: gas_limit,
        };

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
            (u64::BITS - (gas_limit - vm.gas).leading_zeros()) as u64 // hack dont scale this here you doof
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
        for _ in 0..(1<<rng.gen_range(8..=17)) {
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

pub enum Flag {
    Equal = 0,
    SignedLessThan = 1,
    UnsignedLessThan = 2,
    Negative = 3,
}

#[repr(u8)]
#[derive(FromPrimitive, Debug, Copy, Clone, PartialEq, Eq)]
pub enum Opcode {
    Nop,

    /// Jump to PC + (IMM16_B) if IMM8[0] == 0, else jump to IMM16_B
    Jmp,
    SkipIf, // Jump to PC + 8 (skip next insn) if (FLAGS & IMMA_8 == IMMA_8) && (FLAGS & IMMB_8 == 0)

    Const16, // REG_A = IMM16_B
    Mov,     // REG_A = REG_B

    Add,  // add. accum = REG_A + REG_B + IMM8_C
    Sub,  // subtract. see above but negating REG_B
    Test, // compare REG_A to (REG_B + IMM8_C), setting flags

    Mul, // signed integer mult: accum = REG_B * REG_C, high bits in REG_A
    Div, // signed integer div. div by zero produces zero. REG_A = REG_B / REG_C

    //Neg, // bitwise negate REG_A = ~REG_B
    Xor, // bitwise XOR: accum = A^(B+IMM_C)

    Push, // push REG_A onto stack
    Pop,  // set REG_A from stack. terminates program if called with empty stack.

    Call, // push PC+4 to stack, then push stack pointer to stack, then act as Jmp
    Ret,  // pop stack pointer, pop pc

    In,   // In one sample to AREG (at playhead, advancing)
    Out,  // Out one sample from AREG (at playhead, advancing)
    Tell, // store AREG_A playhead to REG_B
    Seek, // Set AREG_A playhead to REG_B

    Filter, // set accumulator to the convolution of the next IMM8_A samples from AREG_B (at playhead) w/ next samples from AREG_C
}

#[derive(Debug, Copy, Clone)]
pub struct Instruction([u8; 4]);
impl Instruction {
    pub fn get_opcode(&self) -> Option<Opcode> {
        FromPrimitive::from_u8(self.0[0])
    }

    pub fn get_operand_imm8(&self, idx: u8) -> u8 {
        self.0[idx as usize + 1]
    }

    pub fn get_operand_imm16(&self, idx: u8) -> u16 {
        let idx = idx as usize;
        u16::from_le_bytes(self.0[idx + 1..idx + 3].try_into().expect("infallible..."))
    }
}

#[derive(Debug, PartialEq)]
pub enum VmRunResult {
    Continue,
    Stop,
    OutOfGas,
}

pub struct VmState {
    pub pc: u16,
    flags: u8,

    regs: [u16; NUM_REGISTERS as usize],

    aregs: [Vec<i16>; NUM_REGISTERS as usize],
    areg_playheads: [usize; NUM_REGISTERS as usize],

    stack: [u16; STACK_SIZE],
    stack_pointer: u16,

    gas: u64,
}
impl VmState {
    fn burn_gas(&mut self, gas: u64) {
        if self.gas > gas {
            self.gas -= gas;
        } else {
            self.gas = 0;
        }
    }
    fn get_reg(&mut self, idx: u8) -> u16 {
        let idx = idx % NUM_REGISTERS;

        if idx == REG_ZERO {
            0
        } else {
            self.regs[idx as usize]
        }
    }
    fn set_reg(&mut self, idx: u8, val: u16) {
        let idx = idx % NUM_REGISTERS;
        self.regs[idx as usize] = val;
    }

    fn advance_playhead(&mut self, idx: usize) {
        self.areg_playheads[idx] += 1;
        self.areg_playheads[idx] %= self.aregs[idx].len();
    }

    fn push_stack(&mut self, val: u16) {
        self.stack[self.stack_pointer as usize] = val;
        self.stack_pointer = (self.stack_pointer + 1) % STACK_SIZE as u16;
    }
    fn pop_stack(&mut self) -> Option<u16> {
        if self.stack_pointer == 0 {
            None
        } else {
            self.stack_pointer -= 1;
            Some(self.stack[self.stack_pointer as usize])
        }
    }

    pub fn run_insn(&mut self, insn: &Instruction) -> VmRunResult {
        if self.gas == 0 {
            return VmRunResult::OutOfGas;
        }
        self.gas -= 1;

        self.pc += 1;

        let Some(opcode) = insn.get_opcode() else {
            return VmRunResult::Continue; // illegal instruction, not fatal :)
        };

        let mut res = VmRunResult::Continue;

        use Opcode::*;

        match opcode {
            Nop => (),

            Jmp | Call => {
                if opcode == Call {
                    let prev_bp = self.get_reg(REG_BP);

                    self.push_stack(self.pc);
                    self.push_stack(prev_bp);
                    self.set_reg(REG_BP, self.stack_pointer);
                    // then "fallthru" to jmp :3
                }

                if insn.get_operand_imm8(0) & 0x1 == 0 {
                    self.pc = self
                        .pc
                        .wrapping_add(insn.get_operand_imm16(1))
                        .wrapping_sub(1);
                } else {
                    self.pc = insn.get_operand_imm16(1);
                }
            }
            SkipIf => {
                let a = insn.get_operand_imm8(0);
                let b = insn.get_operand_imm8(1);

                let taken = (self.flags & a == a) && (self.flags & b == 0);

                if taken {
                    self.pc += 1;
                }
            }
            Const16 => {
                self.set_reg(insn.get_operand_imm8(0), insn.get_operand_imm16(1));
            }
            Mov => {
                let val = self.get_reg(insn.get_operand_imm8(1));
                self.set_reg(insn.get_operand_imm8(1), val);
            }
            Add => {
                let sum = self
                    .get_reg(insn.get_operand_imm8(0))
                    .wrapping_add(self.get_reg(insn.get_operand_imm8(1)))
                    .wrapping_add(insn.get_operand_imm8(2) as u16);

                self.set_reg(REG_ACCUMULATOR, sum);
            }
            Sub => {
                let sum = self.get_reg(insn.get_operand_imm8(0)).wrapping_sub(
                    self.get_reg(insn.get_operand_imm8(1))
                        .wrapping_add(insn.get_operand_imm8(2) as u16),
                );

                self.set_reg(REG_ACCUMULATOR, sum);
            }
            Test => {
                let lhs = self.get_reg(insn.get_operand_imm8(0));
                let rhs = self
                    .get_reg(insn.get_operand_imm8(1))
                    .wrapping_add(insn.get_operand_imm8(2) as u16);

                self.flags = 0;

                if lhs == rhs {
                    self.flags |= Flag::Equal as u8;
                }
                if lhs < rhs {
                    self.flags |= Flag::UnsignedLessThan as u8;
                }
                if (lhs as i16) < (rhs as i16) {
                    self.flags |= Flag::SignedLessThan as u8;
                }
            }
            Mul => {
                let lhs = self.get_reg(insn.get_operand_imm8(1)) as i16 as i32;
                let rhs = self.get_reg(insn.get_operand_imm8(2)) as i16 as i32;

                let product = lhs * rhs;
                let hi = (product >> 16) as u16;
                let lo = product as u16;

                self.set_reg(REG_ACCUMULATOR, hi);
                self.set_reg(insn.get_operand_imm8(0), lo);
            }
            Div => {
                let lhs = self.get_reg(insn.get_operand_imm8(1)) as i16;
                let rhs = self.get_reg(insn.get_operand_imm8(2)) as i16;

                let product = lhs.checked_div(rhs).unwrap_or(0);

                self.set_reg(REG_ACCUMULATOR, product as u16);
            }

            Xor => {
                let lhs = self.get_reg(insn.get_operand_imm8(0));
                let rhs = self
                    .get_reg(insn.get_operand_imm8(1))
                    .wrapping_add(insn.get_operand_imm8(2) as u16);

                self.set_reg(REG_ACCUMULATOR, lhs ^ rhs);
            }

            Push => {
                let val = self.get_reg(insn.get_operand_imm8(0));
                self.push_stack(val);
            }
            Pop => {
                if let Some(val) = self.pop_stack() {
                    self.set_reg(insn.get_operand_imm8(0), val);
                } else {
                    res = VmRunResult::Stop
                }
            }

            // CALL implemented at JMP
            Ret => {
                self.stack_pointer = self.get_reg(REG_BP) % STACK_SIZE as u16;
                if let Some((bp, pc)) = self
                    .pop_stack()
                    .and_then(|bp| self.pop_stack().map(|pc| (bp, pc)))
                {
                    self.set_reg(REG_BP, bp);
                    self.pc = pc;
                } else {
                    res = VmRunResult::Stop;
                }
            }

            In => {
                let sample = {
                    //let idx = (insn.get_operand_imm8(0) % NUM_REGISTERS) as usize;
                    let idx = 0; // HACK for effiency lol
                    let sample = self.aregs[idx][self.areg_playheads[idx]];
                    self.advance_playhead(idx);
                    sample
                };
                self.set_reg(insn.get_operand_imm8(1), sample as u16);
            }
            Out => {
                let sample = self.get_reg(insn.get_operand_imm8(1));

                let idx = 1; // HACK, EFF //insn.get_operand_imm8(0) % NUM_REGISTERS) as usize;
                self.aregs[idx][self.areg_playheads[idx]] = sample as i16;
                self.advance_playhead(idx);

                if self.areg_playheads[idx] == 0 {
                    res = VmRunResult::Stop;
                }
            }
            Tell => {
                let idx = (insn.get_operand_imm8(0) % NUM_REGISTERS) as usize;

                self.set_reg(insn.get_operand_imm8(1), self.areg_playheads[idx] as u16);
            }
            Seek => {
                let idx = (insn.get_operand_imm8(0) % NUM_REGISTERS) as usize;

                let val = self.get_reg(insn.get_operand_imm8(1)) as usize % (self.aregs[idx].len());

                let imm = insn.get_operand_imm8(2);

                if imm & 0x1 == 0 {
                    self.areg_playheads[idx] = val;
                } else {
                    self.areg_playheads[idx] =
                        (self.areg_playheads[idx] + val) % self.aregs[idx].len();
                }
            }
            Filter => {
                let imm = insn.get_operand_imm8(0);
                let kernel_size = imm >> 1;
                let incr_playhead = (imm & 1) == 1;

                self.burn_gas(kernel_size as u64 / 2);

                let kernel_idx = (insn.get_operand_imm8(1) % NUM_REGISTERS) as usize;
                let kernel = &self.aregs[kernel_idx];

                let audio_idx = (insn.get_operand_imm8(2) % NUM_REGISTERS) as usize;
                let audio = &self.aregs[audio_idx];

                let mut kernel_playhead = self.areg_playheads[kernel_idx];
                let mut audio_playhead = self.areg_playheads[audio_idx];

                let mut out_full = 0i32;
                for _ in 0..kernel_size {
                    kernel_playhead = (kernel_playhead + 1) % kernel.len();
                    audio_playhead = (audio_playhead + 1) % audio.len();
                    out_full += kernel[kernel_playhead] as i32 * audio[audio_playhead] as i32;
                }

                if incr_playhead {
                    self.advance_playhead(audio_idx);
                }

                let out = (out_full >> 16) as i16;
                self.set_reg(REG_ACCUMULATOR, out as u16);
            }
        }

        res
    }
}
