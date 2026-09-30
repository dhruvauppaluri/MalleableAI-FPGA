# CL-specific synthesis constraints for the F2 build of cl_otpu.
# UNTESTED. Intentionally empty apart from the reset fan-out hint: add attributes only after the
# first synthesis shows where they are needed.
set_property MAX_FANOUT 50 [get_nets -hierarchical -filter {NAME =~ *u_rm/q[2]}]
