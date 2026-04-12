# Makefile
# See https://docs.cocotb.org/en/stable/quickstart.html for more info

DATA_W ?= 5
ACC_W ?= $(shell expr $(DATA_W) \* 2 + 1)
export SA2X2_DATA_W := $(DATA_W)
export SA2X2_ACC_W := $(ACC_W)
PYTHON ?= python

SIM ?= icarus
TOPLEVEL_LANG ?= verilog
FST ?= -fst

SRC_DIR = ../src
PROJECT_SOURCES = tt_um_sohamgovande_transformer.sv control/sa2x2_controller.sv core/pe_mac.sv core/systolic_array_2x2.sv

ifneq ($(GATES),yes)

SIM_BUILD = sim_build/rtl_w$(DATA_W)
VERILOG_SOURCES += $(addprefix $(SRC_DIR)/,$(PROJECT_SOURCES))

else

SIM_BUILD = sim_build/gl_w$(DATA_W)
COMPILE_ARGS += -DGL_TEST
COMPILE_ARGS += -DFUNCTIONAL
COMPILE_ARGS += -DUSE_POWER_PINS
COMPILE_ARGS += -DSIM
COMPILE_ARGS += -DUNIT_DELAY=\#1
VERILOG_SOURCES += $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/primitives.v
VERILOG_SOURCES += $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/sky130_fd_sc_hd.v
VERILOG_SOURCES += gate_level_netlist.v

endif

COMPILE_ARGS += -I$(SRC_DIR)
COMPILE_ARGS += -DSA2X2_DATA_W=$(DATA_W)

ifeq ($(SIM),verilator)
EXTRA_ARGS += --timing
EXTRA_ARGS += --trace-fst
EXTRA_ARGS += -Wno-fatal
endif

VERILOG_SOURCES += tb.v
TOPLEVEL = tb
COCOTB_TEST_MODULES = test
COCOTB_RESULTS_FILE ?= results.xml

include $(shell $(PYTHON) -m cocotb_tools.config --makefiles)/Makefile.sim
