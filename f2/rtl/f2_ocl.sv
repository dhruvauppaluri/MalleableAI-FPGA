// F2 platform: OCL AXI-Lite front end (shell main clock, 32-bit) for the CL.
//
//   OCL byte offset       target
//   0x0000 - 0x0FFF       the frozen board control block (otpu_ctrl register map, 12-bit;
//                         SLVERR/0xDEADBEEF until HBM is ready,
//                         address), reached through an asynchronous crossing into the core
//                         clock domain. Same offsets as the original board's BAR0, so the
//                         unmodified upstream host driver (opentpu/host/board.py) works.
//   0x1000 - 0x1FFF       the F2 register block below (wrapper-side additions only)
//   anything else         DECERR; reads return 0xDEADBEEF
//
//   F2 block (offsets from 0x1000):
//     0x000 F2_ID          0x46324F54 ("F2OT")
//     0x004 F2_VERSION     1
//     0x008 F2_CAPS        [7:0] PCs per core channel, [15:8] PC-local address bits,
//                          [23:16] log2(stripe bytes), [31:24] 2 (core channels)
//     0x00C F2_STATUS      bit0 HBM ready, bit1 core-path error seen, bit2 PCIS-path error seen
//     0x010 F2_CTRL        write bit0 = 1: clear the sticky error bits and error counts
//     0x014 F2_CORE_KHZ    core clock in kHz (build parameter)
//     0x018 F2_BUILD_ID    build parameter
//     0x01C F2_HBM_BASE_LO PCIS address of the HBM window, low word (0x00000000)
//     0x020 F2_HBM_BASE_HI high word (0x10 = 1 << 36)
//     0x024 F2_CORE_ERRS   error pieces/responses on the core channels
//     0x028 F2_PCIS_ERRS   error pieces/responses on PCIS
//     0x02C F2_SCRATCH     read/write scratch
//
// One OCL transaction is handled at a time. UNTESTED on hardware; not run through Vivado.
module f2_ocl #(
  parameter int PCS_PER_CH = 2,
  parameter int PC_AW      = 29,
  parameter int STRIPE     = 9,
  parameter int CORE_KHZ   = 125000,
  parameter logic [31:0] BUILD_ID = 32'h0
) (
  input  logic         main_clk,
  input  logic         main_rst,
  input  logic         core_clk,
  input  logic         core_rst,

  // ---- OCL AXI-Lite slave (main clock)
  input  logic [31:0]  ocl_awaddr,
  input  logic         ocl_awvalid,
  output logic         ocl_awready,
  input  logic [31:0]  ocl_wdata,
  input  logic [3:0]   ocl_wstrb,
  input  logic         ocl_wvalid,
  output logic         ocl_wready,
  output logic [1:0]   ocl_bresp,
  output logic         ocl_bvalid,
  input  logic         ocl_bready,
  input  logic [31:0]  ocl_araddr,
  input  logic         ocl_arvalid,
  output logic         ocl_arready,
  output logic [31:0]  ocl_rdata,
  output logic [1:0]   ocl_rresp,
  output logic         ocl_rvalid,
  input  logic         ocl_rready,

  // ---- board control master (core clock), 12-bit address
  output logic [11:0]  b_awaddr,
  output logic         b_awvalid,
  input  logic         b_awready,
  output logic [31:0]  b_wdata,
  output logic [3:0]   b_wstrb,
  output logic         b_wvalid,
  input  logic         b_wready,
  input  logic [1:0]   b_bresp,
  input  logic         b_bvalid,
  output logic         b_bready,
  output logic [11:0]  b_araddr,
  output logic         b_arvalid,
  input  logic         b_arready,
  input  logic [31:0]  b_rdata,
  input  logic [1:0]   b_rresp,
  input  logic         b_rvalid,
  output logic         b_rready,

  // ---- status (main clock)
  input  logic         hbm_ready,
  input  logic         core_err_pulse,      // already in the main clock domain
  input  logic         pcis_err_pulse
);
  // ============================================================ F2 register block
  logic [31:0] scratch, core_errs, pcis_errs;
  logic        core_err_seen, pcis_err_seen;
  logic        clr_req;

  always_ff @(posedge main_clk) begin
    if (main_rst) begin
      core_errs <= '0;
      pcis_errs <= '0;
      core_err_seen <= 1'b0;
      pcis_err_seen <= 1'b0;
    end else begin
      if (core_err_pulse) begin core_errs <= core_errs + 1'b1; core_err_seen <= 1'b1; end
      if (pcis_err_pulse) begin pcis_errs <= pcis_errs + 1'b1; pcis_err_seen <= 1'b1; end
      if (clr_req) begin
        core_errs <= '0; pcis_errs <= '0; core_err_seen <= 1'b0; pcis_err_seen <= 1'b0;
      end
    end
  end

  function automatic logic [31:0] f2_read(input logic [11:0] a);
    case (a[11:2])
      10'h000: f2_read = 32'h4632_4F54;
      10'h001: f2_read = 32'd1;
      10'h002: f2_read = {8'd2, 8'(STRIPE), 8'(PC_AW), 8'(PCS_PER_CH)};
      10'h003: f2_read = {29'd0, pcis_err_seen, core_err_seen, hbm_ready};
      10'h004: f2_read = 32'd0;
      10'h005: f2_read = 32'(CORE_KHZ);
      10'h006: f2_read = BUILD_ID;
      10'h007: f2_read = 32'h0000_0000;
      10'h008: f2_read = 32'h0000_0010;
      10'h009: f2_read = core_errs;
      10'h00A: f2_read = pcis_errs;
      10'h00B: f2_read = scratch;
      default: f2_read = 32'hDEAD_BEEF;
    endcase
  endfunction

  // ============================================================ main-domain FSM
  typedef enum logic [2:0] {S_IDLE, S_DISPATCH, S_PUSH, S_WAIT, S_RESP} state_t;
  state_t     st;
  logic       op_wr;
  logic [15:0] op_addr;
  logic [31:0] op_data;
  logic [3:0]  op_strb;
  logic [31:0] res_data;
  logic [1:0]  res_resp;

  logic        rq_wvalid, rq_wready, rs_rvalid, rs_rready;
  logic [48:0] rq_wdata;
  logic [33:0] rs_rdata;
  assign rq_wdata = {op_wr, op_addr[11:0], op_data, op_strb};
  assign rq_wvalid = st == S_PUSH;
  assign rs_rready = st == S_WAIT;

  wire take_w = st == S_IDLE && ocl_awvalid && ocl_wvalid;
  wire take_r = st == S_IDLE && !(ocl_awvalid && ocl_wvalid) && ocl_arvalid;
  assign ocl_awready = take_w;
  assign ocl_wready  = take_w;
  assign ocl_arready = take_r;
  assign ocl_bvalid  = st == S_RESP && op_wr;
  assign ocl_bresp   = res_resp;
  assign ocl_rvalid  = st == S_RESP && !op_wr;
  assign ocl_rresp   = res_resp;
  assign ocl_rdata   = res_data;

  always_ff @(posedge main_clk) begin
    clr_req <= 1'b0;
    if (main_rst) begin
      st <= S_IDLE;
      scratch <= '0;
    end else begin
      case (st)
        S_IDLE: begin
          if (take_w) begin
            op_wr <= 1'b1; op_addr <= ocl_awaddr[15:0]; op_data <= ocl_wdata; op_strb <= ocl_wstrb;
            st <= S_DISPATCH;
          end else if (take_r) begin
            op_wr <= 1'b0; op_addr <= ocl_araddr[15:0]; op_data <= '0; op_strb <= '0;
            st <= S_DISPATCH;
          end
        end
        S_DISPATCH: begin
          // upper address bits beyond the 64 MiB BAR are the shell's business; only [15:12] decode here
          if (op_addr[15:12] == 4'h0 && hbm_ready) begin
            st <= S_PUSH;
          end else if (op_addr[15:12] == 4'h0) begin
            // the board block is held in reset until HBM is initialized: answer SLVERR
            // instead of stalling the shell's OCL port
            res_resp <= 2'b10;
            res_data <= 32'hDEAD_BEEF;
            st <= S_RESP;
          end else if (op_addr[15:12] == 4'h1) begin
            res_resp <= 2'b00;
            res_data <= op_wr ? 32'd0 : f2_read(op_addr[11:0]);
            if (op_wr) begin
              if (op_addr[11:2] == 10'h004 && op_strb[0] && op_data[0]) clr_req <= 1'b1;
              if (op_addr[11:2] == 10'h00B) begin
                for (int b = 0; b < 4; b++) if (op_strb[b]) scratch[b*8 +: 8] <= op_data[b*8 +: 8];
              end
            end
            st <= S_RESP;
          end else begin
            res_resp <= 2'b11;
            res_data <= 32'hDEAD_BEEF;
            st <= S_RESP;
          end
        end
        S_PUSH: if (rq_wready) st <= S_WAIT;
        S_WAIT: if (rs_rvalid) begin
          res_data <= rs_rdata[33:2];
          res_resp <= rs_rdata[1:0];
          st <= S_RESP;
        end
        S_RESP: if ((op_wr && ocl_bready) || (!op_wr && ocl_rready)) st <= S_IDLE;
        default: st <= S_IDLE;
      endcase
    end
  end

  // ============================================================ crossings
  logic        rq_rvalid, rq_rready;
  logic [48:0] rq_rdata;
  f2_async_fifo #(.W(49), .AW(2)) u_req (
    .wclk(main_clk), .wrst(main_rst), .wvalid(rq_wvalid), .wready(rq_wready), .wdata(rq_wdata),
    .rclk(core_clk), .rrst(core_rst), .rvalid(rq_rvalid), .rready(rq_rready), .rdata(rq_rdata));

  logic        rs_wvalid, rs_wready;
  logic [33:0] rs_wdata;
  f2_async_fifo #(.W(34), .AW(2)) u_rsp (
    .wclk(core_clk), .wrst(core_rst), .wvalid(rs_wvalid), .wready(rs_wready), .wdata(rs_wdata),
    .rclk(main_clk), .rrst(main_rst), .rvalid(rs_rvalid), .rready(rs_rready), .rdata(rs_rdata));

  // ============================================================ core-domain master
  typedef enum logic [2:0] {C_IDLE, C_WR, C_WRESP, C_RD, C_RRESP, C_PUSH} cstate_t;
  cstate_t     cs;
  logic        c_wr, aw_done, w_done;
  logic [11:0] c_addr;
  logic [31:0] c_data;
  logic [3:0]  c_strb;
  logic [31:0] c_rd_data;
  logic [1:0]  c_resp;

  assign rq_rready = cs == C_IDLE && rq_rvalid;
  assign b_awaddr  = c_addr;
  assign b_awvalid = cs == C_WR && !aw_done;
  assign b_wdata   = c_data;
  assign b_wstrb   = c_strb;
  assign b_wvalid  = cs == C_WR && !w_done;
  assign b_bready  = cs == C_WRESP;
  assign b_araddr  = c_addr;
  assign b_arvalid = cs == C_RD;
  assign b_rready  = cs == C_RRESP;
  assign rs_wvalid = cs == C_PUSH;
  assign rs_wdata  = {c_wr ? 32'd0 : c_rd_data, c_resp};

  always_ff @(posedge core_clk) begin
    if (core_rst) begin
      cs <= C_IDLE;
    end else begin
      case (cs)
        C_IDLE: if (rq_rvalid) begin
          {c_wr, c_addr, c_data, c_strb} <= rq_rdata;
          aw_done <= 1'b0; w_done <= 1'b0;
          cs <= rq_rdata[48] ? C_WR : C_RD;
        end
        C_WR: begin
          if (b_awvalid && b_awready) aw_done <= 1'b1;
          if (b_wvalid && b_wready) w_done <= 1'b1;
          if ((aw_done || (b_awvalid && b_awready)) && (w_done || (b_wvalid && b_wready))) cs <= C_WRESP;
        end
        C_WRESP: if (b_bvalid) begin c_resp <= b_bresp; cs <= C_PUSH; end
        C_RD: if (b_arready) cs <= C_RRESP;
        C_RRESP: if (b_rvalid) begin c_rd_data <= b_rdata; c_resp <= b_rresp; cs <= C_PUSH; end
        C_PUSH: if (rs_wready) cs <= C_IDLE;
        default: cs <= C_IDLE;
      endcase
    end
  end
endmodule
