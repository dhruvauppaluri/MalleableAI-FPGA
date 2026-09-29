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

.PHONY: all test test-rtl test-host lint synth verify verify-llm test-upstream ui clean release-tiny release-check full-release-check verify-evidence test-f2 lint-f2 synth-f2 verify-f2 replay-f2 lint-f2-hdk check-f2-scripts synth-f2-xilinx

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

test-upstream:
	PYTHONPATH=third_party/opentpu $(PYTHON) -m pytest -q third_party/opentpu/tests/test_isa.py third_party/opentpu/tests/test_fp.py third_party/opentpu/tests/test_rtl.py third_party/opentpu/tests/test_quant.py third_party/opentpu/tests/test_compiler.py
	PYTHONPATH=third_party/opentpu $(PYTHON) -m pytest -q third_party/opentpu/tests/test_qwen3.py third_party/opentpu/tests/test_qwen35.py third_party/opentpu/tests/test_lfm2.py -k 'tiny or plan or one_sequence'
	$(PYTHON) -m pytest -q tests/test_llm_rtl.py

ui:
	npm --prefix frontend ci
	npm --prefix frontend run build

F2_RTL := \
	f2/rtl/f2_fifo.sv \
	f2/rtl/f2_cdc.sv \
	f2/rtl/f2_ocl.sv \
	f2/rtl/f2_hbm_router.sv \
	f2/rtl/f2_hbm_pc_bridge.sv \
	f2/rtl/f2_hbm_adapter.sv

# F2 platform (docs/adr/0008-*.md, docs/f2-status.md): simulation and structural checks only.
lint-f2:
	$(VERILATOR) --lint-only --Wall -Wno-fatal -Wno-DECLFILENAME -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
		--top-module f2_hbm_adapter $(F2_RTL)
	PYTHONPATH=. $(PYTHON) -m malleable.f2.sim --sources --rtl-only | xargs $(VERILATOR) --lint-only -Wno-fatal \
		-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-DECLFILENAME -Wno-TIMESCALEMOD \
		--top-module cl_otpu_core

synth-f2: | $(BUILD_DIR)
	$(YOSYS) -q -l $(BUILD_DIR)/synthesis-f2.log -p \
		"read_verilog -sv $(F2_RTL); hierarchy -top f2_hbm_adapter; synth -top f2_hbm_adapter -run begin:fine; select -assert-none t:\$$dlatch p:*; check -assert; stat"

test-f2:
	PYTHONPATH=. $(PYTHON) -m pytest -q tests/test_f2_placement.py tests/test_f2_host.py tests/test_f2_rtl.py tests/test_f2_replay.py

verify-f2: lint-f2 synth-f2 test-f2 check-f2-scripts

# Yosys cell counts (generic UltraScale+ mapping, not Vivado) for the F2 wrapper blocks.
synth-f2-xilinx:
	$(PYTHON) f2/tools/yosys_resources.py --json $(BUILD_DIR)/f2/yosys-resources.json

# Vivado/HDK scripts are UNTESTED (no Vivado here): only shell syntax and Tcl bracket balance.
check-f2-scripts:
	bash -n f2/vivado/setup_cl.sh
	bash -n f2/vivado/run_ooc.sh
	@for f in f2/vivado/ooc_synth.tcl f2/vivado/synth_cl_otpu.tcl; do \
		printf 'set fh [open %s]; set c [read $$fh]; close $$fh; if {![info complete $$c]} {puts "unbalanced: %s"; exit 1}\n' $$f $$f | tclsh || exit 1; \
	done

# Port-connection lint of the HDK-facing top against cl_ports.vh, with stub HDK modules.
# Needs a clone of aws/aws-fpga: make lint-f2-hdk AWS_FPGA_REPO_DIR=/path/to/aws-fpga
lint-f2-hdk:
	@test -n "$(AWS_FPGA_REPO_DIR)" || (echo 'Set AWS_FPGA_REPO_DIR=/path/to/aws-fpga (only cl_ports.vh and cl_id_defines.vh are read)' && exit 2)
	PYTHONPATH=. $(PYTHON) -m malleable.f2.sim --sources --rtl-only | xargs $(VERILATOR) --lint-only -Wno-fatal \
		-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-DECLFILENAME -Wno-TIMESCALEMOD \
		-Wno-PINCONNECTEMPTY -Wno-MULTIDRIVEN -Wno-UNDRIVEN -Wno-UNOPTFLAT \
		-I$(AWS_FPGA_REPO_DIR)/hdk/common/shell_stable/design/interfaces -I$(AWS_FPGA_REPO_DIR)/hdk/cl/examples/CL_TEMPLATE/design \
		--top-module cl_otpu f2/vivado/lint_stubs.sv f2/hdk/cl_otpu.sv

# Durable 100-token replay evidence (destination must not exist), as tools/export_tiny_rtl_evidence.py does for RTL.
replay-f2:
	PYTHONPATH=. $(PYTHON) -m malleable.f2.replay --output $(BUILD_DIR)/release-evidence/f2-replay

verify-llm:
	$(PYTHON) -c "import torch, transformers, safetensors, fastapi, httpx, pytest"
	$(MAKE) verify PYTHON=$(PYTHON)
	$(PYTHON) -m pytest -q tests/test_diagnostics.py tests/test_precision.py tests/test_int8_candidates.py tests/test_candidate_integration.py tests/test_release_implementation.py tests/test_cuda_cache.py
	$(MAKE) test-upstream PYTHON=$(PYTHON)
	$(MAKE) test-f2 PYTHON=$(PYTHON)
	npm --prefix frontend run check
	npm --prefix frontend run build

release-tiny:
	PYTHONPATH=. $(PYTHON) tools/export_tiny_rtl_evidence.py --output $(BUILD_DIR)/release-evidence/tiny

release-check:
	@test -n "$(RELEASE_MANIFEST)" || (echo 'Set RELEASE_MANIFEST=path/to/standalone.json' && exit 2)
	$(PYTHON) -m malleable.llm.cli release-check --manifest $(RELEASE_MANIFEST)

full-release-check:
	@test -n "$(FULL_RELEASE_MANIFEST)" || (echo 'Set FULL_RELEASE_MANIFEST=path/to/full-release.json' && exit 2)
	$(PYTHON) -m malleable.llm.cli full-release-check --manifest $(FULL_RELEASE_MANIFEST)

verify-evidence:
	PYTHONPATH=. $(PYTHON) tools/verify_local_release.py --output $(BUILD_DIR)/release-evidence/verification

clean:
	rm -f $(BUILD_DIR)/int8_mac_tb.out $(BUILD_DIR)/int8_dot_product_tb.out $(BUILD_DIR)/int8_tiled_accumulator_tb.out $(BUILD_DIR)/int8_postprocess_tb.out $(BUILD_DIR)/malleable_accelerator_top_tb.out
	@echo 'Checkpoint folders, experiments, traces and unrelated build data are preserved.'
