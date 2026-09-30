// Behavioural stand-ins for AWS HDK modules, only so that `make lint-f2-hdk` can check the port
// connections of f2/hdk/cl_otpu.sv with Verilator. They are written from the port lists in
// the HDK sources (cl_mem_perf, CL_TEMPLATE, aws_clk_gen) and are NOT the AWS modules; they
// are never used in a build.
interface cfg_bus_t #(parameter ADDR_WIDTH = 32, parameter DATA_WIDTH = 32, parameter USER_WIDTH = 3);
  logic [ADDR_WIDTH-1:0] addr;
  logic [DATA_WIDTH-1:0] wdata;
  logic                  wr, rd;
  logic [USER_WIDTH-1:0] user;
  logic                  ack;
  logic [DATA_WIDTH-1:0] rdata;
  modport slave (input addr, wdata, wr, rd, user, output ack, rdata);
  modport master (output addr, wdata, wr, rd, user, input ack, rdata);
endinterface

module sh_ddr #(parameter DDR_PRESENT = 1) (
  input  logic clk, output logic rst_n, input logic stat_clk, output logic stat_rst_n,
  input  logic CLK_DIMM_DP, input logic CLK_DIMM_DN, output logic M_ACT_N, output logic [17:0] M_MA,
  output logic [1:0] M_BA, output logic [1:0] M_BG, output logic [1:0] M_CKE, output logic [1:0] M_ODT,
  output logic [1:0] M_CS_N, output logic [1:0] M_CLK_DN, output logic [1:0] M_CLK_DP, output logic M_PAR,
  inout  wire [63:0] M_DQ, inout wire [7:0] M_ECC, inout wire [17:0] M_DQS_DP, inout wire [17:0] M_DQS_DN,
  output logic cl_RST_DIMM_N,
  input  logic [15:0] cl_sh_ddr_axi_awid, input logic [63:0] cl_sh_ddr_axi_awaddr, input logic [7:0] cl_sh_ddr_axi_awlen,
  input  logic [2:0] cl_sh_ddr_axi_awsize, input logic cl_sh_ddr_axi_awvalid, input logic [1:0] cl_sh_ddr_axi_awburst,
  input  logic cl_sh_ddr_axi_awuser, output logic cl_sh_ddr_axi_awready,
  input  logic [511:0] cl_sh_ddr_axi_wdata, input logic [63:0] cl_sh_ddr_axi_wstrb, input logic cl_sh_ddr_axi_wlast,
  input  logic cl_sh_ddr_axi_wvalid, output logic cl_sh_ddr_axi_wready,
  output logic [15:0] cl_sh_ddr_axi_bid, output logic [1:0] cl_sh_ddr_axi_bresp, output logic cl_sh_ddr_axi_bvalid,
  input  logic cl_sh_ddr_axi_bready,
  input  logic [15:0] cl_sh_ddr_axi_arid, input logic [63:0] cl_sh_ddr_axi_araddr, input logic [7:0] cl_sh_ddr_axi_arlen,
  input  logic [2:0] cl_sh_ddr_axi_arsize, input logic cl_sh_ddr_axi_arvalid, input logic [1:0] cl_sh_ddr_axi_arburst,
  input  logic cl_sh_ddr_axi_aruser, output logic cl_sh_ddr_axi_arready,
  output logic [15:0] cl_sh_ddr_axi_rid, output logic [511:0] cl_sh_ddr_axi_rdata, output logic [1:0] cl_sh_ddr_axi_rresp,
  output logic cl_sh_ddr_axi_rlast, output logic cl_sh_ddr_axi_rvalid, input logic cl_sh_ddr_axi_rready,
  input  logic [7:0] sh_ddr_stat_bus_addr, input logic [31:0] sh_ddr_stat_bus_wdata, input logic sh_ddr_stat_bus_wr,
  input  logic sh_ddr_stat_bus_rd, output logic sh_ddr_stat_bus_ack, output logic [31:0] sh_ddr_stat_bus_rdata,
  output logic [7:0] ddr_sh_stat_int, output logic sh_cl_ddr_is_ready);
  assign {rst_n, stat_rst_n, M_ACT_N, M_MA, M_BA, M_BG, M_CKE, M_ODT, M_CS_N, M_CLK_DN, M_CLK_DP, M_PAR,
          cl_RST_DIMM_N, cl_sh_ddr_axi_awready, cl_sh_ddr_axi_wready, cl_sh_ddr_axi_bid, cl_sh_ddr_axi_bresp,
          cl_sh_ddr_axi_bvalid, cl_sh_ddr_axi_arready, cl_sh_ddr_axi_rid, cl_sh_ddr_axi_rdata, cl_sh_ddr_axi_rresp,
          cl_sh_ddr_axi_rlast, cl_sh_ddr_axi_rvalid, sh_ddr_stat_bus_ack, sh_ddr_stat_bus_rdata, ddr_sh_stat_int,
          sh_cl_ddr_is_ready} = '0;
endmodule

module aws_clk_gen #(parameter CLK_GRP_A_EN = 1, CLK_GRP_B_EN = 1, CLK_GRP_C_EN = 1, CLK_HBM_EN = 1) (
  input  logic i_clk_main_a0, input logic i_rst_main_n, input logic i_clk_hbm_ref,
  input  logic [31:0] s_axil_ctrl_awaddr, input logic s_axil_ctrl_awvalid, output logic s_axil_ctrl_awready,
  input  logic [31:0] s_axil_ctrl_wdata, input logic [3:0] s_axil_ctrl_wstrb, input logic s_axil_ctrl_wvalid,
  output logic s_axil_ctrl_wready, output logic [1:0] s_axil_ctrl_bresp, output logic s_axil_ctrl_bvalid,
  input  logic s_axil_ctrl_bready, input logic [31:0] s_axil_ctrl_araddr, input logic s_axil_ctrl_arvalid,
  output logic s_axil_ctrl_arready, output logic [31:0] s_axil_ctrl_rdata, output logic [1:0] s_axil_ctrl_rresp,
  output logic s_axil_ctrl_rvalid, input logic s_axil_ctrl_rready,
  output logic o_clk_hbm_ref, output logic o_clk_main_a0, output logic o_clk_extra_a1, output logic o_clk_extra_a2,
  output logic o_clk_extra_a3, output logic o_clk_extra_b0, output logic o_clk_extra_b1, output logic o_clk_extra_c0,
  output logic o_clk_extra_c1, output logic o_clk_hbm_axi,
  output logic o_cl_rst_hbm_axi_n, output logic o_cl_rst_hbm_ref_n, output logic o_cl_rst_c1_n, output logic o_cl_rst_c0_n,
  output logic o_cl_rst_b1_n, output logic o_cl_rst_b0_n, output logic o_cl_rst_a3_n, output logic o_cl_rst_a2_n,
  output logic o_cl_rst_a1_n, output logic o_cl_rst_main_n);
  assign {s_axil_ctrl_awready, s_axil_ctrl_wready, s_axil_ctrl_bresp, s_axil_ctrl_bvalid, s_axil_ctrl_arready,
          s_axil_ctrl_rdata, s_axil_ctrl_rresp, s_axil_ctrl_rvalid, o_clk_hbm_ref, o_clk_main_a0, o_clk_extra_a1,
          o_clk_extra_a2, o_clk_extra_a3, o_clk_extra_b0, o_clk_extra_b1, o_clk_extra_c0, o_clk_extra_c1,
          o_clk_hbm_axi, o_cl_rst_hbm_axi_n, o_cl_rst_hbm_ref_n, o_cl_rst_c1_n, o_cl_rst_c0_n, o_cl_rst_b1_n,
          o_cl_rst_b0_n, o_cl_rst_a3_n, o_cl_rst_a2_n, o_cl_rst_a1_n, o_cl_rst_main_n} = '0;
endmodule

module cl_mem_hbm_wrapper #(parameter NUM_OF_AXI_PORTS = 1, parameter AXI4_INTERFACE = 0,
                            parameter AXLEN_WIDTH = AXI4_INTERFACE ? 8 : 4) (
  input logic apb_clk, input logic i_clk_250m, input logic i_rst_250m_n, input logic i_clk_450m, input logic i_rst_450m_n,
  input logic [33:0] i_axi_araddr [0:NUM_OF_AXI_PORTS-1], input logic [1:0] i_axi_arburst [0:NUM_OF_AXI_PORTS-1],
  input logic [5:0] i_axi_arid [0:NUM_OF_AXI_PORTS-1], input logic [AXLEN_WIDTH-1:0] i_axi_arlen [0:NUM_OF_AXI_PORTS-1],
  input logic [2:0] i_axi_arsize [0:NUM_OF_AXI_PORTS-1], input logic i_axi_arvalid [0:NUM_OF_AXI_PORTS-1],
  input logic [33:0] i_axi_awaddr [0:NUM_OF_AXI_PORTS-1], input logic [1:0] i_axi_awburst [0:NUM_OF_AXI_PORTS-1],
  input logic [5:0] i_axi_awid [0:NUM_OF_AXI_PORTS-1], input logic [AXLEN_WIDTH-1:0] i_axi_awlen [0:NUM_OF_AXI_PORTS-1],
  input logic [2:0] i_axi_awsize [0:NUM_OF_AXI_PORTS-1], input logic i_axi_awvalid [0:NUM_OF_AXI_PORTS-1],
  input logic i_axi_rready [0:NUM_OF_AXI_PORTS-1], input logic i_axi_bready [0:NUM_OF_AXI_PORTS-1],
  input logic [255:0] i_axi_wdata [0:NUM_OF_AXI_PORTS-1], input logic i_axi_wlast [0:NUM_OF_AXI_PORTS-1],
  input logic [31:0] i_axi_wstrb [0:NUM_OF_AXI_PORTS-1], input logic i_axi_wvalid [0:NUM_OF_AXI_PORTS-1],
  output logic o_axi_arready [0:NUM_OF_AXI_PORTS-1], output logic o_axi_awready [0:NUM_OF_AXI_PORTS-1],
  output logic [255:0] o_axi_rdata [0:NUM_OF_AXI_PORTS-1], output logic [5:0] o_axi_rid [0:NUM_OF_AXI_PORTS-1],
  output logic o_axi_rlast [0:NUM_OF_AXI_PORTS-1], output logic [1:0] o_axi_rresp [0:NUM_OF_AXI_PORTS-1],
  output logic o_axi_rvalid [0:NUM_OF_AXI_PORTS-1], output logic o_axi_wready [0:NUM_OF_AXI_PORTS-1],
  output logic [5:0] o_axi_bid [0:NUM_OF_AXI_PORTS-1], output logic [1:0] o_axi_bresp [0:NUM_OF_AXI_PORTS-1],
  output logic o_axi_bvalid [0:NUM_OF_AXI_PORTS-1],
  input logic i_hbm_apb_preset_n_1, output logic [21:0] o_hbm_apb_paddr_1, output logic [2:0] o_hbm_apb_pprot_1,
  output logic o_hbm_apb_psel_1, output logic o_hbm_apb_penable_1, output logic o_hbm_apb_pwrite_1,
  output logic [31:0] o_hbm_apb_pwdata_1, output logic [3:0] o_hbm_apb_pstrb_1, output logic o_hbm_apb_pready_1,
  output logic [31:0] o_hbm_apb_prdata_1, output logic o_hbm_apb_pslverr_1,
  input logic i_hbm_apb_preset_n_0, output logic [21:0] o_hbm_apb_paddr_0, output logic [2:0] o_hbm_apb_pprot_0,
  output logic o_hbm_apb_psel_0, output logic o_hbm_apb_penable_0, output logic o_hbm_apb_pwrite_0,
  output logic [31:0] o_hbm_apb_pwdata_0, output logic [3:0] o_hbm_apb_pstrb_0, output logic o_hbm_apb_pready_0,
  output logic [31:0] o_hbm_apb_prdata_0, output logic o_hbm_apb_pslverr_0,
  cfg_bus_t.slave hbm_stat_bus, output logic [7:0] o_cl_sh_hbm_stat_int, output logic o_hbm_ready);
  assign {o_hbm_apb_paddr_1, o_hbm_apb_pprot_1, o_hbm_apb_psel_1, o_hbm_apb_penable_1, o_hbm_apb_pwrite_1,
          o_hbm_apb_pwdata_1, o_hbm_apb_pstrb_1, o_hbm_apb_pready_1, o_hbm_apb_prdata_1, o_hbm_apb_pslverr_1,
          o_hbm_apb_paddr_0, o_hbm_apb_pprot_0, o_hbm_apb_psel_0, o_hbm_apb_penable_0, o_hbm_apb_pwrite_0,
          o_hbm_apb_pwdata_0, o_hbm_apb_pstrb_0, o_hbm_apb_pready_0, o_hbm_apb_prdata_0, o_hbm_apb_pslverr_0,
          o_cl_sh_hbm_stat_int, o_hbm_ready} = '0;
  assign hbm_stat_bus.ack = 1'b0;
  assign hbm_stat_bus.rdata = '0;
  for (genvar i = 0; i < NUM_OF_AXI_PORTS; i++) begin : g
    assign o_axi_arready[i] = 1'b0;  assign o_axi_awready[i] = 1'b0;  assign o_axi_rdata[i] = '0;
    assign o_axi_rid[i] = '0;  assign o_axi_rlast[i] = 1'b0;  assign o_axi_rresp[i] = '0;  assign o_axi_rvalid[i] = 1'b0;
    assign o_axi_wready[i] = 1'b0;  assign o_axi_bid[i] = '0;  assign o_axi_bresp[i] = '0;  assign o_axi_bvalid[i] = 1'b0;
  end
endmodule
