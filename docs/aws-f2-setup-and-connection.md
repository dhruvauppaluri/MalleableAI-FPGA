# AWS F2 setup for MalleableAI-FPGA

This is the hardware path after local RTL simulation. It does not turn the
current simulator into an F2 bitstream. The repository has no F2 host driver,
AWS Small Shell integration, PCIe transfer path, HBM path, DCP, or AFI.

## 1. Prepare an AWS account

Choose an AWS Region and Availability Zone offering EC2 F2, and check current
F2 On-Demand quota and hourly price in that Region. An F2 quota may start at
zero; request an increase before planning a run. Put a spending alarm/budget
in place and plan to stop or terminate development instances when idle. Create
an EC2 key pair and a security group allowing SSH only from your address, or
use a private SSM connection. Keep checkpoint weights and private datasets on
your own account storage; do not put credentials in this repository or the AFI.

Create an instance role with the narrow S3/EC2 FPGA permissions required for
the tutorial and AFI build. Confirm the role and chosen bucket are in the same
intended Region. Follow AWS's current F2 HDK instructions for the exact role,
AMI, driver and tool versions; hardcoded AMI IDs and prices go stale.

## 2. Prove the AWS reference example first

Launch the current AWS FPGA Developer AMI on an F2 instance (the HDK describes
the required build/runtime setup). Connect over SSH or SSM. On the instance,
clone AWS's `aws-fpga` repository and use its `hdk_setup.sh` and `sdk_setup.sh`
scripts as instructed by the current F2 guide. Run the `cl_sde` or simple
register-access example end to end: build a DCP, submit AFI creation using S3,
wait for the AGFI, load it in slot 0, and run its host sample. Use
`fpga-describe-local-image -S 0` to verify a loaded image. Record AMI, HDK/SDK
commit, Vivado version, Region, instance type, shell version, AGFI and logs.
An AFI ID and a global AGFI ID have different roles; the load command uses AGFI.

## 3. Port this accelerator

Wrap OpenTPU RTL as an F2 Custom Logic (CL) design under AWS Small Shell.
Implement and verify shell clock/reset, AXI-Lite controls, PCIe data transfer
(SDE or explicit DMA), HBM/DDR addressing, interrupt/completion and safe
drain/reset. The current shell has no built-in DMA engine. Ensure actual model
images and KV state fit board memory. Simulate the CL with the AWS HDK testbench,
then run timing, synthesis and DCP build in the compatible Vivado flow. Publish
source/tool/shell identity and resource/timing reports. Start with one static
balanced INT8 image and a tiny deterministic test against the ISA/RTL output;
then test a supported pretrained model and personality variants. Do not assume
the Quartus test projects compile an F2 image.

## 4. Measure switching on board

The host driver should write board-clock start/end timestamps for drain,
program, reload, initialization, KV reset or reprefill, warmup and controller
overhead. Record source/destination image hashes, platform/shell, clock and
context compatibility on every trace. Capture at least three independent
transitions per pair as a first diagnostic, including cold and warm cases.
`transition_measurement.aggregate` can validate these records and produce an
observed upper cost with a reserve. Its reserve is heuristic, not a tail
guarantee; collect more traces before promotion. A first static-image F2 run
does not measure a real personality switch.

## 5. Connect the machine to Codex

For a **local Codex workspace**, open this repository from a terminal that can
SSH into your EC2 instance. An SSH config host alias keeps commands simple:

```sshconfig
Host malleable-f2
  HostName <instance-public-dns-or-private-vpn-address>
  User <developer-ami-user>
  IdentityFile <local-private-key-path>
  IdentitiesOnly yes
```

Use `ssh malleable-f2` to verify access. Keep the checkout and untracked model
files on the instance; sync only reviewed source, manifests, logs and small
results. A remote SSH workspace or an agent started on the instance can run
the build and measurement commands there. Do not expose the workbench port to
the public internet; use an SSH tunnel if needed.

For **this managed cloud Codex environment**, network reachability and AWS
credentials are separate setup items. The currently selected environment has
no AWS identity binding and no direct TCP destination grants. It cannot SSH
to an F2 instance or operate an AWS account as configured. Use the Codex cloud
environment configuration workflow to attach a scoped AWS identity and allow
the instance's SSH endpoint or a configured Tailscale route. After setup,
verify readiness through `cloud_environment.environment_status`, then test
`aws sts get-caller-identity` and `ssh malleable-f2`. Alternatively, work from
a local Codex session with SSH access. Never paste private keys or AWS secret
keys into chat.

## References

- [AWS F2 Developer Kit](https://awsdocs-fpga-f2.readthedocs-hosted.com/latest/)
- [AWS HDK getting started](https://awsdocs-fpga-f2.readthedocs-hosted.com/latest/hdk/README.html)
- [AWS AFI guide](https://awsdocs-fpga-f2.readthedocs-hosted.com/latest/hdk/docs/Amazon-FPGA-Images-Afis-Guide.html)
- [AWS FPGA SDK](https://awsdocs-fpga-f2.readthedocs-hosted.com/latest/sdk/README.html)
