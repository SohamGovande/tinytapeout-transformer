`default_nettype none
`timescale 1ns / 1ps

module systolic_array_2x2 #(
    parameter int DATA_W = 4,
    parameter int ACC_W = 9
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     clear,
    input  logic signed [DATA_W-1:0] a_left0,
    input  logic signed [DATA_W-1:0] a_left1,
    input  logic signed [DATA_W-1:0] b_top0,
    input  logic signed [DATA_W-1:0] b_top1,
    output logic signed [ACC_W-1:0]  c00,
    output logic signed [ACC_W-1:0]  c01,
    output logic signed [ACC_W-1:0]  c10,
    output logic signed [ACC_W-1:0]  c11
);

    logic signed [DATA_W-1:0] a_pipe_01;
    logic signed [DATA_W-1:0] a_pipe_11;
    logic signed [DATA_W-1:0] b_pipe_10;
    logic signed [DATA_W-1:0] b_pipe_11;
    logic signed [DATA_W-1:0] unused_a_out_01;
    logic signed [DATA_W-1:0] unused_b_out_10;
    logic signed [DATA_W-1:0] unused_a_out_11;
    logic signed [DATA_W-1:0] unused_b_out_11;

    pe_mac #(
        .DATA_W(DATA_W),
        .ACC_W (ACC_W)
    ) pe_00 (
        .clk    (clk),
        .rst_n  (rst_n),
        .clear  (clear),
        .a_in   (a_left0),
        .b_in   (b_top0),
        .a_out  (a_pipe_01),
        .b_out  (b_pipe_10),
        .acc_out(c00)
    );

    pe_mac #(
        .DATA_W(DATA_W),
        .ACC_W (ACC_W)
    ) pe_01 (
        .clk    (clk),
        .rst_n  (rst_n),
        .clear  (clear),
        .a_in   (a_pipe_01),
        .b_in   (b_top1),
        .a_out  (unused_a_out_01),
        .b_out  (b_pipe_11),
        .acc_out(c01)
    );

    pe_mac #(
        .DATA_W(DATA_W),
        .ACC_W (ACC_W)
    ) pe_10 (
        .clk    (clk),
        .rst_n  (rst_n),
        .clear  (clear),
        .a_in   (a_left1),
        .b_in   (b_pipe_10),
        .a_out  (a_pipe_11),
        .b_out  (unused_b_out_10),
        .acc_out(c10)
    );

    pe_mac #(
        .DATA_W(DATA_W),
        .ACC_W (ACC_W)
    ) pe_11 (
        .clk    (clk),
        .rst_n  (rst_n),
        .clear  (clear),
        .a_in   (a_pipe_11),
        .b_in   (b_pipe_11),
        .a_out  (unused_a_out_11),
        .b_out  (unused_b_out_11),
        .acc_out(c11)
    );

endmodule

`default_nettype wire
