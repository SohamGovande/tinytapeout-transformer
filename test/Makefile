# Makefile
# See https://docs.cocotb.org/en/stable/quickstart.html for more info

SIM ?= verilator
TOPLEVEL_LANG ?= verilog
FST ?= -fst

SRC_DIR = $(PWD)/../src
PROJECT_SOURCES = tt_um_sohamgovande_sa2x2.sv control/sa2x2_controller.sv core/pe_mac.sv core/systolic_array_2x2.sv

ifneq ($(GATES),yes)

SIM_BUILD = sim_build/rtl
VERILOG_SOURCES += $(addprefix $(SRC_DIR)/,$(PROJECT_SOURCES))

else

SIM_BUILD = sim_build/gl
COMPILE_ARGS += -DGL_TEST
COMPILE_ARGS += -DFUNCTIONAL
COMPILE_ARGS += -DUSE_POWER_PINS
COMPILE_ARGS += -DSIM
COMPILE_ARGS += -DUNIT_DELAY=\#1
VERILOG_SOURCES += $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/primitives.v
VERILOG_SOURCES += $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/sky130_fd_sc_hd.v
VERILOG_SOURCES += $(PWD)/gate_level_netlist.v

endif

COMPILE_ARGS += -I$(SRC_DIR)
EXTRA_ARGS += --timing
EXTRA_ARGS += --trace-fst
EXTRA_ARGS += -Wno-fatal

VERILOG_SOURCES += $(PWD)/tb.v
TOPLEVEL = tb
COCOTB_TEST_MODULES = test

include $(shell python3 -m cocotb_tools.config --makefiles)/Makefile.sim
