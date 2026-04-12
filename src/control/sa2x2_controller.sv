`default_nettype none
`timescale 1ns / 1ps

module sa2x2_controller #(
    parameter int DATA_W = 4,
    parameter int ACC_W = (2 * DATA_W) + 1
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        cmd_stb,
    input  logic [1:0]  cmd,
    input  logic [1:0]  addr,
    input  logic        aux_in,
    input  logic [7:0]  payload,
    output logic [8:0]  read_data,
    output logic        busy
);

    localparam logic [1:0] CMD_WRITE_A = 2'b00;
    localparam logic [1:0] CMD_WRITE_B = 2'b01;
    localparam logic [1:0] CMD_EXEC    = 2'b10;
    localparam logic [1:0] CMD_READ    = 2'b11;

    localparam logic [2:0] OP_MATMUL      = 3'b000;
    localparam logic [2:0] OP_MATMUL_ACC  = 3'b001;
    localparam logic [2:0] OP_EW_ADD      = 3'b010;
    localparam logic [2:0] OP_EW_RELU     = 3'b011;
    localparam logic [2:0] OP_EW_SHIFT    = 3'b100;

    localparam int IO_W = 9;
    localparam logic [5:0] ACC_W_U = 6'(ACC_W);

    typedef enum logic [2:0] {
        STATE_IDLE   = 3'd0,
        STATE_CLEAR  = 3'd1,
        STATE_FEED0  = 3'd2,
        STATE_FEED1  = 3'd3,
        STATE_FLUSH0 = 3'd4,
        STATE_FLUSH1 = 3'd5,
        STATE_LATCH  = 3'd6,
        STATE_EXEC   = 3'd7
    } state_t;

    state_t state_q;

    logic signed [DATA_W-1:0] bank_a_q [0:3];
    logic signed [DATA_W-1:0] bank_b_q [0:3];
    logic signed [ACC_W-1:0]  bank_c_q [0:3];

    logic [1:0] read_bank_q;
    logic [1:0] read_addr_q;
    logic [5:0] read_chunk_q;

    logic [2:0] exec_op_q;
    logic [1:0] exec_addr_q;
    logic [3:0] exec_shift_q;

    logic                     array_clear;
    logic signed [DATA_W-1:0] array_a_left0;
    logic signed [DATA_W-1:0] array_a_left1;
    logic signed [DATA_W-1:0] array_b_top0;
    logic signed [DATA_W-1:0] array_b_top1;
    logic signed [ACC_W-1:0]  array_c00;
    logic signed [ACC_W-1:0]  array_c01;
    logic signed [ACC_W-1:0]  array_c10;
    logic signed [ACC_W-1:0]  array_c11;

    logic signed [DATA_W-1:0] write_data_word;
    logic [ACC_W-1:0]         read_word_bits;

    integer word_idx;

    function automatic logic signed [ACC_W-1:0] relu_word(
        input logic signed [ACC_W-1:0] value
    );
        begin
            relu_word = value[ACC_W-1] ? '0 : value;
        end
    endfunction

    function automatic logic signed [ACC_W-1:0] shift_word(
        input logic signed [ACC_W-1:0] value,
        input logic [3:0]              amount
    );
        begin
            if ({2'b0, amount} >= ACC_W_U)
                shift_word = value[ACC_W-1] ? {ACC_W{1'b1}} : '0;
            else
                shift_word = value >>> amount;
        end
    endfunction

    function automatic logic signed [ACC_W-1:0] extend_data_word(
        input logic signed [DATA_W-1:0] value
    );
        begin
            extend_data_word = {{(ACC_W-DATA_W){value[DATA_W-1]}}, value};
        end
    endfunction

    function automatic logic [IO_W-1:0] read_chunk_word(
        input logic [ACC_W-1:0] value,
        input logic [5:0]       chunk
    );
        integer bit_idx;
        begin
            read_chunk_word = '0;
            for (bit_idx = 0; bit_idx < IO_W; bit_idx = bit_idx + 1) begin
                if ((chunk * IO_W) + bit_idx < ACC_W)
                    read_chunk_word[bit_idx] = value[(chunk * IO_W) + bit_idx];
            end
        end
    endfunction

    function automatic logic signed [DATA_W-1:0] decode_write_word(
        input logic       aux,
        input logic [7:0] data
    );
        integer bit_idx;
        begin
            decode_write_word = '0;
            for (bit_idx = 0; bit_idx < DATA_W; bit_idx = bit_idx + 1) begin
                if (bit_idx < 8)
                    decode_write_word[bit_idx] = data[bit_idx];
                else
                    decode_write_word[bit_idx] = aux;
            end
        end
    endfunction

    systolic_array_2x2 #(
        .DATA_W(DATA_W),
        .ACC_W (ACC_W)
    ) u_array (
        .clk    (clk),
        .rst_n  (rst_n),
        .clear  (array_clear),
        .a_left0(array_a_left0),
        .a_left1(array_a_left1),
        .b_top0 (array_b_top0),
        .b_top1 (array_b_top1),
        .c00    (array_c00),
        .c01    (array_c01),
        .c10    (array_c10),
        .c11    (array_c11)
    );

    always_comb begin
        array_clear = 1'b0;
        array_a_left0 = '0;
        array_a_left1 = '0;
        array_b_top0 = '0;
        array_b_top1 = '0;

        case (state_q)
            STATE_CLEAR: begin
                array_clear = 1'b1;
            end
            STATE_FEED0: begin
                array_a_left0 = bank_a_q[0];
                array_b_top0 = bank_b_q[0];
            end
            STATE_FEED1: begin
                array_a_left0 = bank_a_q[1];
                array_a_left1 = bank_a_q[2];
                array_b_top0 = bank_b_q[2];
                array_b_top1 = bank_b_q[1];
            end
            STATE_FLUSH0: begin
                array_a_left1 = bank_a_q[3];
                array_b_top1 = bank_b_q[3];
            end
            default: begin
            end
        endcase
    end

    always_comb begin
        write_data_word = decode_write_word(aux_in, payload);

        read_word_bits = '0;
        if (read_bank_q == 2'd0) begin
            case (read_addr_q)
                2'd0: read_word_bits[DATA_W-1:0] = bank_a_q[0];
                2'd1: read_word_bits[DATA_W-1:0] = bank_a_q[1];
                2'd2: read_word_bits[DATA_W-1:0] = bank_a_q[2];
                default: read_word_bits[DATA_W-1:0] = bank_a_q[3];
            endcase
        end else if (read_bank_q == 2'd1) begin
            case (read_addr_q)
                2'd0: read_word_bits[DATA_W-1:0] = bank_b_q[0];
                2'd1: read_word_bits[DATA_W-1:0] = bank_b_q[1];
                2'd2: read_word_bits[DATA_W-1:0] = bank_b_q[2];
                default: read_word_bits[DATA_W-1:0] = bank_b_q[3];
            endcase
        end else begin
            case (read_addr_q)
                2'd0: read_word_bits = bank_c_q[0];
                2'd1: read_word_bits = bank_c_q[1];
                2'd2: read_word_bits = bank_c_q[2];
                default: read_word_bits = bank_c_q[3];
            endcase
        end

        read_data = read_chunk_word(read_word_bits, read_chunk_q);
    end

    assign busy = (state_q != STATE_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= STATE_IDLE;
            read_bank_q <= '0;
            read_addr_q <= '0;
            read_chunk_q <= '0;
            exec_op_q <= OP_MATMUL;
            exec_addr_q <= '0;
            exec_shift_q <= '0;
            for (word_idx = 0; word_idx < 4; word_idx = word_idx + 1) begin
                bank_a_q[word_idx] <= '0;
                bank_b_q[word_idx] <= '0;
                bank_c_q[word_idx] <= '0;
            end
        end else begin
            case (state_q)
                STATE_IDLE: begin
                    if (cmd_stb) begin
                        case (cmd)
                            CMD_WRITE_A: bank_a_q[addr] <= write_data_word;
                            CMD_WRITE_B: bank_b_q[addr] <= write_data_word;
                            CMD_READ: begin
                                read_bank_q <= payload[1:0];
                                read_addr_q <= addr;
                                read_chunk_q <= payload[7:2];
                            end
                            CMD_EXEC: begin
                                exec_op_q <= payload[2:0];
                                exec_addr_q <= addr;
                                exec_shift_q <= payload[7:4];

                                if ((payload[2:0] == OP_MATMUL) || (payload[2:0] == OP_MATMUL_ACC))
                                    state_q <= STATE_CLEAR;
                                else
                                    state_q <= STATE_EXEC;
                            end
                            default: begin
                            end
                        endcase
                    end
                end

                STATE_CLEAR: begin
                    state_q <= STATE_FEED0;
                end

                STATE_FEED0: begin
                    state_q <= STATE_FEED1;
                end

                STATE_FEED1: begin
                    state_q <= STATE_FLUSH0;
                end

                STATE_FLUSH0: begin
                    state_q <= STATE_FLUSH1;
                end

                STATE_FLUSH1: begin
                    state_q <= STATE_LATCH;
                end

                STATE_LATCH: begin
                    if (exec_op_q == OP_MATMUL_ACC) begin
                        bank_c_q[0] <= bank_c_q[0] + array_c00;
                        bank_c_q[1] <= bank_c_q[1] + array_c01;
                        bank_c_q[2] <= bank_c_q[2] + array_c10;
                        bank_c_q[3] <= bank_c_q[3] + array_c11;
                    end else begin
                        bank_c_q[0] <= array_c00;
                        bank_c_q[1] <= array_c01;
                        bank_c_q[2] <= array_c10;
                        bank_c_q[3] <= array_c11;
                    end

                    state_q <= STATE_IDLE;
                end

                STATE_EXEC: begin
                    case (exec_op_q)
                        OP_EW_ADD: bank_c_q[exec_addr_q] <=
                            extend_data_word(bank_a_q[exec_addr_q]) +
                            extend_data_word(bank_b_q[exec_addr_q]);
                        OP_EW_RELU: bank_c_q[exec_addr_q] <= relu_word(bank_c_q[exec_addr_q]);
                        OP_EW_SHIFT: bank_c_q[exec_addr_q] <= shift_word(bank_c_q[exec_addr_q], exec_shift_q);
                        default: begin
                        end
                    endcase

                    state_q <= STATE_IDLE;
                end

                default: begin
                    state_q <= STATE_IDLE;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
