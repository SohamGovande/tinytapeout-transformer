#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <verilated.h>

#include "Vtt_um_sohamgovande_transformer.h"

namespace {

constexpr int kCmdWriteA = 0b00;
constexpr int kCmdWriteB = 0b01;
constexpr int kCmdExec = 0b10;
constexpr int kCmdRead = 0b11;

constexpr int kBankA = 0;
constexpr int kBankB = 1;
constexpr int kBankC = 2;

constexpr int kOpMatmul = 0b000;
constexpr int kOpMatmulAcc = 0b001;
constexpr int kOpEwAdd = 0b010;
constexpr int kOpEwRelu = 0b011;
constexpr int kOpEwShift = 0b100;

constexpr int kIoWidth = 9;

int to_signed(uint32_t value, int width) {
    const uint32_t mask = (1u << width) - 1u;
    value &= mask;
    const uint32_t sign_bit = 1u << (width - 1);
    if (value & sign_bit)
        return static_cast<int>(value) - static_cast<int>(1u << width);
    return static_cast<int>(value);
}

uint32_t to_unsigned(int value, int width) {
    const uint32_t mask = (1u << width) - 1u;
    return static_cast<uint32_t>(value) & mask;
}

void tick(Vtt_um_sohamgovande_transformer* dut, VerilatedContext* context) {
    dut->clk = 0;
    dut->eval();
    context->timeInc(1);

    dut->clk = 1;
    dut->eval();
    context->timeInc(1);
}

void reset_dut(Vtt_um_sohamgovande_transformer* dut, VerilatedContext* context) {
    dut->ena = 1;
    dut->ui_in = 0;
    dut->uio_in = 0;
    dut->rst_n = 0;
    tick(dut, context);
    tick(dut, context);
    dut->rst_n = 1;
    tick(dut, context);
}

void pulse_cmd(
    Vtt_um_sohamgovande_transformer* dut,
    VerilatedContext* context,
    int cmd,
    int addr,
    int aux,
    int payload
) {
    dut->ui_in = to_unsigned(payload, 8);
    dut->uio_in = 0x1 | ((cmd & 0x3) << 1) | ((addr & 0x3) << 3) | ((aux & 0x1) << 6);
    tick(dut, context);
    dut->ui_in = 0;
    dut->uio_in = 0;
    tick(dut, context);
}

int get_busy(Vtt_um_sohamgovande_transformer* dut) {
    return (dut->uio_out >> 5) & 0x1;
}

int get_read_chunk(Vtt_um_sohamgovande_transformer* dut) {
    int raw = ((dut->uio_out >> 7) & 0x1) << 8;
    raw |= dut->uo_out;
    return raw;
}

void wait_while_busy(Vtt_um_sohamgovande_transformer* dut, VerilatedContext* context, int max_cycles = 32) {
    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        if (!get_busy(dut))
            return;
        tick(dut, context);
    }
    throw std::runtime_error("timed out waiting for busy to deassert");
}

void write_bank_entry(
    Vtt_um_sohamgovande_transformer* dut,
    VerilatedContext* context,
    int bank,
    int addr,
    int value
) {
    const int cmd = (bank == kBankB) ? kCmdWriteB : kCmdWriteA;
    const uint32_t encoded = to_unsigned(value, 5);
    pulse_cmd(dut, context, cmd, addr, (encoded >> 8) & 0x1, encoded & 0xFF);
}

int read_bank_entry(
    Vtt_um_sohamgovande_transformer* dut,
    VerilatedContext* context,
    int bank,
    int addr
) {
    const int width = (bank == kBankC) ? 11 : 5;
    const int chunks = (width + kIoWidth - 1) / kIoWidth;
    uint32_t raw = 0;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        pulse_cmd(dut, context, kCmdRead, addr, 0, bank | (chunk << 2));
        const int chunk_raw = get_read_chunk(dut);
        const int used_bits = std::min(kIoWidth, width - (chunk * kIoWidth));
        raw |= static_cast<uint32_t>(chunk_raw & ((1 << used_bits) - 1)) << (chunk * kIoWidth);
    }
    return to_signed(raw, width);
}

void exec_op(
    Vtt_um_sohamgovande_transformer* dut,
    VerilatedContext* context,
    int op,
    int addr,
    int shift = 0
) {
    const int payload = ((shift & 0xF) << 4) | (op & 0x7);
    pulse_cmd(dut, context, kCmdExec, addr, 0, payload);
    wait_while_busy(dut, context);
}

}  // namespace

int main(int argc, char** argv) {
    VerilatedContext context;
    context.commandArgs(argc, argv);
    Vtt_um_sohamgovande_transformer dut{&context};
    dut.clk = 0;
    reset_dut(&dut, &context);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string op;
        iss >> op;

        try {
            if (op == "reset") {
                reset_dut(&dut, &context);
                std::cout << "OK" << std::endl;
            } else if (op == "write_a" || op == "write_b") {
                int addr = 0;
                int value = 0;
                iss >> addr >> value;
                write_bank_entry(&dut, &context, op == "write_b" ? kBankB : kBankA, addr, value);
                std::cout << "OK" << std::endl;
            } else if (op == "read_a" || op == "read_b" || op == "read_c") {
                int addr = 0;
                iss >> addr;
                const int bank = (op == "read_a") ? kBankA : ((op == "read_b") ? kBankB : kBankC);
                std::cout << "VALUE " << read_bank_entry(&dut, &context, bank, addr) << std::endl;
            } else if (op == "matmul") {
                exec_op(&dut, &context, kOpMatmul, 0, 0);
                std::cout << "OK" << std::endl;
            } else if (op == "matmul_acc") {
                exec_op(&dut, &context, kOpMatmulAcc, 0, 0);
                std::cout << "OK" << std::endl;
            } else if (op == "ew_add") {
                int addr = 0;
                iss >> addr;
                exec_op(&dut, &context, kOpEwAdd, addr, 0);
                std::cout << "OK" << std::endl;
            } else if (op == "ew_relu") {
                int addr = 0;
                iss >> addr;
                exec_op(&dut, &context, kOpEwRelu, addr, 0);
                std::cout << "OK" << std::endl;
            } else if (op == "ew_shift") {
                int addr = 0;
                int shift = 0;
                iss >> addr >> shift;
                exec_op(&dut, &context, kOpEwShift, addr, shift);
                std::cout << "OK" << std::endl;
            } else if (op == "shutdown") {
                std::cout << "OK" << std::endl;
                break;
            } else if (op.empty()) {
                std::cout << "OK" << std::endl;
            } else {
                std::cout << "ERR unknown command: " << op << std::endl;
            }
        } catch (const std::exception& ex) {
            std::cout << "ERR " << ex.what() << std::endl;
        }
    }

    dut.final();
    return 0;
}
