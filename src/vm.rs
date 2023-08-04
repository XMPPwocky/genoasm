use std::io::{self, Write};

use num_traits::FromPrimitive;

use num_derive::FromPrimitive;
use serde::{Deserialize, Serialize};

pub const NUM_REGISTERS: u8 = 16;

pub const REG_ZERO: u8 = 1;
pub const REG_ACCUMULATOR: u8 = 1;
pub const REG_BP: u8 = NUM_REGISTERS - 1;

pub const AREG_REFERENCE: u8 = 0;
pub const AREG_LUT: u8 = 1;

pub const NUM_INSTRUCTIONS: usize = 4096;
pub const LUT_SIZE: usize = 256;
pub const STACK_SIZE: usize = 256;

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

    pub aregs: [Vec<i16>; NUM_REGISTERS as usize], // public because HACK
    areg_playheads: [usize; NUM_REGISTERS as usize],

    stack: [u16; STACK_SIZE],
    stack_pointer: u16,

    gas: u64,
}
impl VmState {
    pub fn new(aregs: [Vec<i16>; NUM_REGISTERS as usize], gas_limit: u64) -> Self {
        VmState {
            pc: 0,
            flags: 0,
            regs: [0; NUM_REGISTERS as usize],
            aregs,
            areg_playheads: [0; NUM_REGISTERS as usize],
            stack: [0; STACK_SIZE],
            stack_pointer: 0,
            gas: gas_limit,
        }
    }
    fn burn_gas(&mut self, gas: u64) {
        self.gas = self.gas.saturating_sub(gas);
    }
    fn get_reg(&self, idx: u8) -> u16 {
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

    fn unadvance_playhead(&mut self, idx: usize) {
        self.areg_playheads[idx] += self.aregs[idx].len() - 1;
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
        self.burn_gas(1);

        self.pc += 1;

        let Some(opcode) = insn.get_opcode() else {
            return VmRunResult::Continue; // illegal instruction, not fatal :)
        };

        let mut res = VmRunResult::Continue;

        use Opcode::*;

        match opcode {
            Nop | Maximum => (),

            Call | Jmp => {

                if opcode == Call {
                    let prev_bp = self.get_reg(REG_BP);

                    self.push_stack(self.pc);
                    self.push_stack(prev_bp);
                    self.set_reg(REG_BP, self.stack_pointer);
                }

                self.pc = self.pc.wrapping_add_signed(insn.get_operand_imm16(1) as i16) % NUM_INSTRUCTIONS as u16;
            }
            JmpIf => {
                let a = insn.get_operand_imm8(0);
                let b = insn.get_operand_imm8(1);

                let taken = (self.flags & a == a) && (self.flags & b == 0);

                if taken {
                    // offset
                    self.pc = self
                        .pc
                        .wrapping_sub(128)
                        .wrapping_add(insn.get_operand_imm8(2) as u16);
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
                let idx = (insn.get_operand_imm8(0) % 3) as usize; // HACK for effiency lol
                let sample = self.aregs[idx][self.areg_playheads[idx]];
                if insn.get_operand_imm8(2) & 0x1 == 0x1 {
                    self.unadvance_playhead(idx);
                } else {
                    self.advance_playhead(idx);
                }
                self.set_reg(insn.get_operand_imm8(1), sample as u16);
            }
            Out => {
                let sample = self.get_reg(insn.get_operand_imm8(1));

                let idx = (insn.get_operand_imm8(0) % 4) as usize; // HACK for efficiency
                self.aregs[idx][self.areg_playheads[idx]] += sample as i16;
                self.advance_playhead(idx);

                if self.areg_playheads[idx] == 0 && insn.get_operand_imm8(2) & 1 == 1 {
                    res = VmRunResult::Stop;
                }
            }
            Tell => {
                let idx = (insn.get_operand_imm8(0) % NUM_REGISTERS) as usize;

                let pos = self.areg_playheads[idx];

                let (hi, lo) = if insn.get_operand_imm8(2) & 0x1 == 0x1 {
                    ((pos >> 16) as u16, pos as u16)
                } else {
                    let pos = (pos as f64 / self.aregs[idx].len() as f64 * u32::MAX as f64).floor()
                        as u32;
                    ((pos >> 16) as u16, pos as u16)
                };

                self.set_reg(REG_ACCUMULATOR, hi);
                self.set_reg(insn.get_operand_imm8(1), lo);
            }
            Seek => {
                let idx = (insn.get_operand_imm8(0) % NUM_REGISTERS) as usize;

                let val = self.get_reg(insn.get_operand_imm8(1)) as usize % (self.aregs[idx].len());

                let imm = insn.get_operand_imm8(2);

                if imm & 0x1 == 0 {
                    let q = (val as f64 / u16::MAX as f64 * self.aregs[idx].len() as f64).floor()
                        as usize;
                    self.areg_playheads[idx] = q;
                } else {
                    self.areg_playheads[idx] =
                        (self.areg_playheads[idx] + val) % self.aregs[idx].len();
                }
            }
            Die => {
                res = VmRunResult::Stop;
            }
            Filter => {
                let imm = insn.get_operand_imm8(0);
                let kernel_size = imm >> 1;
                let incr_playhead = (imm & 1) == 1;

                self.burn_gas(kernel_size as u64 * 2 / 3);

                let audio_idx = 3; // always out tbh // (insn.get_operand_imm8(2) % NUM_REGISTERS) as usize;
                let mut audio = Vec::new();
                std::mem::swap(&mut audio, &mut self.aregs[audio_idx]); // this will cause problems later

                let kernel_idx = 2; // always LUT tbh (insn.get_operand_imm8(1) % NUM_REGISTERS) as usize;
                let kernel = &self.aregs[kernel_idx];

                let mut kernel_playhead = self.areg_playheads[kernel_idx];
                let mut audio_playhead = self.areg_playheads[audio_idx];

                let scale = self.get_reg(insn.get_operand_imm8(REG_ACCUMULATOR)) as i16;
                for _ in 0..kernel_size {
                    kernel_playhead = (kernel_playhead + kernel.len() - 11) % kernel.len();
                    audio_playhead = (audio_playhead + 1) % audio.len();

                    audio[audio_playhead] +=
                        (((kernel[kernel_playhead] as i32) * (scale as i32)) >> 16) as i16;
                }

                std::mem::swap(&mut audio, &mut self.aregs[audio_idx]);

                if incr_playhead {
                    self.advance_playhead(audio_idx);
                }
            }
        }

        res
    }

    pub fn gas_remaining(&self) -> u64 {
        self.gas
    }
}

pub enum Flag {
    Equal = 0,
    SignedLessThan = 1,
    UnsignedLessThan = 2,
    Negative = 3,
}

#[repr(u8)]
#[derive(FromPrimitive, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Opcode {
    Nop,

    Jmp,
    JmpIf,

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

    Die, // stop program

    Filter, // set accumulator to the convolution of the next IMM8_A samples from AREG_B (at playhead) w/ next samples from AREG_C

    Maximum,
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize, Eq)]
pub struct Instruction(pub [u8; 4]);
impl PartialEq for Instruction {
    fn eq(&self, other: &Self) -> bool {
        match self.get_opcode() {
            // die and nop take no args, so just cmp opcode
            Some(Opcode::Die | Opcode::Nop) => self.get_opcode() == other.get_opcode(),
            _ => self.0 == other.0,
        }
    }
}
impl std::hash::Hash for Instruction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self.get_opcode() {
            // die and nop take no args, so just cmp opcode
            Some(Opcode::Die | Opcode::Nop) => self.get_opcode().hash(state),
            _ => self.0.hash(state),
        };
    }
}

impl Instruction {
    pub fn write(&self, writer: &mut impl Write, addr: u16) -> io::Result<()> {
        write!(writer, "{addr:04x}: ")?;

        let addr = addr.wrapping_add(1);

        // this can probably be a debug impl
        if let Some(op) = self.get_opcode() {
            use Opcode::*;
            match op {
                Nop => writeln!(writer, "nop"),
                //Jmp => writeln!(writer, "jmp\t\t{:04x}", addr.wrapping_add(self.get_operand_imm16(1)) % NUM_INSTRUCTIONS as u16),
                Call => writeln!(
                    writer,
                    "call\t\t{:04x}",
                    addr.wrapping_add(self.get_operand_imm16(1)) % NUM_INSTRUCTIONS as u16
                ),
                JmpIf => writeln!(
                    writer,
                    "jif\t\t{:02x},\t{:02x},\t{:02x}",
                    self.0[1],
                    self.0[2],
                    addr.wrapping_add(self.get_operand_imm8(2) as u16) % NUM_INSTRUCTIONS as u16
                ),

                Const16 => writeln!(
                    writer,
                    "const\t\t{},\t{:04x}",
                    self.get_operand_reg_name(0),
                    self.get_operand_imm16(1)
                ),

                Out => writeln!(
                    writer,
                    "out\t\t{},\tA{},\t{}",
                    self.get_operand_reg_name(1),
                    3, // HACK FIXED AUDIO OUT, AUDIO FUCKER DISASSEMBLE CORRECTLY
                    if self.get_operand_imm8(2) & 1 == 1 {
                        "stop"
                    } else {
                        "wrap"
                    }
                ),
                op => writeln!(
                    writer,
                    "{:?}\t\t{:02x},\t{:02x},\t{:02x}",
                    op, self.0[1], self.0[2], self.0[3]
                ),
            }?;
        } else {
            writeln!(
                writer,
                "ill_({:02x})\t\t{:02x}\t{:02x}\t{:02x}",
                self.0[0], self.0[1], self.0[2], self.0[3]
            )?;
        }

        Ok(())
    }

    pub fn get_operand_reg_name(&self, operand_idx: u8) -> String {
        // free perf: return an &'static str here
        let idx = self.get_operand_imm8(operand_idx);
        format!("r{:?}", idx % NUM_REGISTERS)
    }

    pub fn get_opcode(&self) -> Option<Opcode> {
        FromPrimitive::from_u8((self.0[0]) % Opcode::Maximum as u8)
    }

    pub fn get_operand_imm8(&self, idx: u8) -> u8 {
        self.0[idx as usize + 1]
    }

    pub fn get_operand_imm16(&self, idx: u8) -> u16 {
        let idx = idx as usize;
        u16::from_le_bytes(self.0[idx + 1..idx + 3].try_into().expect("infallible..."))
    }
    pub fn set_operand_imm16(&mut self, idx: u8, val: u16) {
        let i = idx as usize;
        let k: &mut [u8] = &mut self.0[i + 1..i + 3];
        k.copy_from_slice(&u16::to_le_bytes(val));
        assert_eq!(self.get_operand_imm16(idx), val);
    }
}
