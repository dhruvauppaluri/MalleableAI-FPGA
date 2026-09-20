BUILD_DIR := build
IVERILOG ?= iverilog
VVP ?= vvp

.PHONY: all test test-rtl clean

all: test

test: test-rtl

test-rtl: $(BUILD_DIR)/int8_mac_tb.out $(BUILD_DIR)/int8_dot_product_tb.out
	$(VVP) $(BUILD_DIR)/int8_mac_tb.out
	$(VVP) $(BUILD_DIR)/int8_dot_product_tb.out

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/int8_mac_tb.out: rtl/int8_mac.sv sim/int8_mac_tb.sv | $(BUILD_DIR)
	$(IVERILOG) -g2012 -Wall -s int8_mac_tb -o $@ $^

$(BUILD_DIR)/int8_dot_product_tb.out: rtl/int8_dot_product.sv sim/int8_dot_product_tb.sv | $(BUILD_DIR)
	$(IVERILOG) -g2012 -Wall -s int8_dot_product_tb -o $@ $^

clean:
	rm -rf $(BUILD_DIR)
