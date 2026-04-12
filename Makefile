.PHONY: test test-gl lint synth tt-config harden warnings png clean

TOP_MODULE := tt_um_sohamgovande_transformer
SRC_FILES := $(shell find src -name '*.sv' | sort)
TT_TOOL := ./tt/tt_tool.py
SYNTH_SCRIPT := scripts/synth.ys

define require_tt_tool
	@if [ ! -f "$(TT_TOOL)" ]; then \
		echo "error: $(TT_TOOL) not found. Initialize the tt-support-tools submodule at ./tt first."; \
		exit 1; \
	fi
endef

test:
	$(MAKE) -C test -B

test-gl:
	$(MAKE) -C test -B GATES=yes

lint:
	verilator --lint-only --sv -Wall -Isrc --top-module $(TOP_MODULE) $(SRC_FILES)

synth:
	mkdir -p build
	yosys -s $(SYNTH_SCRIPT) | tee build/synth.log

tt-config:
	$(call require_tt_tool)
	python3 $(TT_TOOL) --create-user-config

harden:
	$(call require_tt_tool)
	python3 $(TT_TOOL) --harden

warnings:
	$(call require_tt_tool)
	python3 $(TT_TOOL) --print-warnings

png:
	$(call require_tt_tool)
	python3 $(TT_TOOL) --create-png

clean:
	rm -rf build test/sim_build test/results*.xml test/tb.fst test/tb.vcd test/output test/__pycache__ test/.pytest_cache test/gate_level_netlist.v
