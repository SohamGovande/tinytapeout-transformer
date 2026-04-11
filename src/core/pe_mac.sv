`default_nettype none
`timescale 1ns / 1ps

module pe_mac #(
    parameter int DATA_W = 4,
    parameter int ACC_W = 9
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     clear,
    input  logic signed [DATA_W-1:0] a_in,
    input  logic signed [DATA_W-1:0] b_in,
    output logic signed [DATA_W-1:0] a_out,
    output logic signed [DATA_W-1:0] b_out,
    output logic signed [ACC_W-1:0]  acc_out
);

    localparam int PROD_W = DATA_W * 2;

    logic signed [PROD_W-1:0] product;
    logic signed [ACC_W-1:0] product_ext;

    assign product = a_in * b_in;
    assign product_ext = {{(ACC_W-PROD_W){product[PROD_W-1]}}, product};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= '0;
            b_out <= '0;
            acc_out <= '0;
        end else begin
            a_out <= a_in;
            b_out <= b_in;

            if (clear)
                acc_out <= '0;
            else
                acc_out <= acc_out + product_ext;
        end
    end

endmodule

`default_nettype wire
