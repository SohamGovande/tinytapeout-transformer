/*
 * Copyright (c) 2025 Soham Govande
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

`ifndef SA2X2_DATA_W
`define SA2X2_DATA_W 5
`endif

module tt_um_sohamgovande_transformer (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
`ifdef USE_POWER_PINS
    ,
    input wire VPWR,
    input wire VGND
`endif
);

    localparam int DATA_W = `SA2X2_DATA_W;
    localparam int ACC_W = (2 * DATA_W) + 1;

    wire [8:0] read_data;
    wire busy;

    sa2x2_controller #(
        .DATA_W(DATA_W),
        .ACC_W (ACC_W)
    ) u_controller (
        .clk     (clk),
        .rst_n   (rst_n),
        .cmd_stb (uio_in[0]),
        .cmd     (uio_in[2:1]),
        .addr    (uio_in[4:3]),
        .aux_in  (uio_in[6]),
        .payload (ui_in),
        .read_data(read_data),
        .busy    (busy)
    );

    assign uo_out = read_data[7:0];
    assign uio_out = {read_data[8], 1'b0, busy, 5'b0};
    assign uio_oe = 8'b1010_0000;

`ifdef USE_POWER_PINS
    wire _unused_power = &{VPWR, VGND, 1'b0};
`endif
    wire _unused = &{ena, uio_in[7], uio_in[5], 1'b0};

endmodule

`default_nettype wire
