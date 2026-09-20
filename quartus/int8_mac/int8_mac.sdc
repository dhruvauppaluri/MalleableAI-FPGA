create_clock -name clk -period 10.000 [get_ports {clk}]
derive_clock_uncertainty

set_input_delay -clock clk 0.000 [get_ports {reset valid_in multiplicand[*] multiplier[*] accumulator_in[*]}]
set_output_delay -clock clk 0.000 [get_ports {result[*] valid_out overflow}]
