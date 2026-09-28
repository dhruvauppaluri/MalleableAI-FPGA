BUILD_DIR := build
IVERILOG ?= iverilog
VVP ?= vvp
VERILATOR ?= verilator
YOSYS ?= yosys
PYTHON ?= python3

ACCELERATOR_RTL := \
	rtl/int8_dot_product.sv \
	rtl/int8_tiled_accumulator.sv \
	rtl/int8_postprocess.sv \
	rtl/malleable_accelerator_top.sv

.PHONY: all test test-rtl test-host lint synth verify clean

all: verify

test: test-rtl test-host

test-host:
	$(PYTHON) -m unittest discover -s tests -v

test-rtl: \
	$(BUILD_DIR)/int8_mac_tb.out \
	$(BUILD_DIR)/int8_dot_product_tb.out \
	$(BUILD_DIR)/int8_tiled_accumulator_tb.out \
	$(BUILD_DIR)/int8_postprocess_tb.out \
	$(BUILD_DIR)/malleable_accelerator_top_tb.out
	$(VVP) $(BUILD_DIR)/int8_mac_tb.out
	$(VVP) $(BUILD_DIR)/int8_dot_product_tb.out
	$(VVP) $(BUILD_DIR)/int8_tiled_accumulator_tb.out
	$(VVP) $(BUILD_DIR)/int8_postprocess_tb.out
	$(VVP) $(BUILD_DIR)/malleable_accelerator_top_tb.out

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/int8_mac_tb.out: rtl/int8_mac.sv sim/int8_mac_tb.sv | $(BUILD_DIR)
	$(IVERILOG) -g2012 -Wall -s int8_mac_tb -o $@ $^

$(BUILD_DIR)/int8_dot_product_tb.out: rtl/int8_dot_product.sv sim/int8_dot_product_tb.sv | $(BUILD_DIR)
	$(IVERILOG) -g2012 -Wall -s int8_dot_product_tb -o $@ $^

$(BUILD_DIR)/int8_tiled_accumulator_tb.out: rtl/int8_dot_product.sv rtl/int8_tiled_accumulator.sv sim/int8_tiled_accumulator_tb.sv | $(BUILD_DIR)
	$(IVERILOG) -g2012 -Wall -s int8_tiled_accumulator_tb -o $@ $^

$(BUILD_DIR)/int8_postprocess_tb.out: rtl/int8_postprocess.sv sim/int8_postprocess_tb.sv | $(BUILD_DIR)
	$(IVERILOG) -g2012 -Wall -s int8_postprocess_tb -o $@ $^

$(BUILD_DIR)/malleable_accelerator_top_tb.out: $(ACCELERATOR_RTL) sim/malleable_accelerator_top_tb.sv | $(BUILD_DIR)
	$(IVERILOG) -g2012 -Wall -s malleable_accelerator_top_tb -o $@ $^


lint:
	$(VERILATOR) --lint-only --Wall -Wno-fatal --top-module int8_mac rtl/int8_mac.sv
	@for lanes in 1 2 4 8; do \
		$(VERILATOR) --lint-only --Wall -Wno-fatal \
		-Wno-WIDTH -Wno-BLKSEQ -Wno-UNUSEDSIGNAL -GLANES=$$lanes \
		--top-module malleable_accelerator_top $(ACCELERATOR_RTL) || exit 1; \
	done

synth: | $(BUILD_DIR)
	@for lanes in 1 2 4 8; do \
		$(YOSYS) -q -l $(BUILD_DIR)/synthesis-$$lanes.log -p \
		"read_verilog -sv $(ACCELERATOR_RTL); chparam -set LANES $$lanes malleable_accelerator_top; synth -top malleable_accelerator_top -run begin:fine; select -assert-none t:\$$dlatch p:*; check -assert; stat" || exit 1; \
	done

verify: test lint synth


clean:
	rm -rf $(BUILD_DIR)
