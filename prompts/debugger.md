Your previous change to chipyard caused the failure below. If it came from
the software side (build or FireSim run failure rather than a Chisel build
error), check first whether it's a `gemmini_params.h` mismatch against your
hardware change -- see the Co-Design Loop and Tuning Strategy sections of
your instructions. Using the chipyard_bash tool, diagnose the root cause
and fix it. Actually call the tool to edit the files yourself -- don't just
describe the fix or hand back a script; if you don't call the tool, no
change happens at all. Respond explaining what was wrong and what you
changed.
