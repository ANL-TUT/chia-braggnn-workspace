Your previous hardware change succeeded and was validated end-to-end on
real FPGA hardware via FireSim, with the results below. These numbers are
your previous hardware paired with the best schedule the automated SW
search (Exo + AlphaEvolve) could find for it, not a fixed kernel -- read
them as feedback on the hardware, per the Co-Design Loop section of your
instructions. A Hammer PPA (Genus synthesis, sky130 + SRAM macros) area/slack/power
report for the same hardware may also be included below; when present,
treat it as a secondary objective alongside cycles/accuracy -- do not trade
away correctness or a large amount of cycles for a small area win, and
remember the only lever for it is the same Configs.scala hardware
parameters (never edit anything under vlsi/, per the Restriction section).
The sky130 slack is negative even for the unedited design: synthesized
without retiming, the path from `norm_unit_module` into `acc_scale_unit`
misses the 20 ns clock by about 23.6 ns. Judge slack by how it moves from
one attempt to the next, not by its sign, and do not try to fix that path.
Using the chipyard_bash tool, try to further improve the accelerator:
reduce the cycle count, reduce area, and/or improve accuracy through
hardware parameter changes. Actually call the tool to edit the files
yourself -- don't just describe the change or hand back a script; if you
don't call the tool, no change happens at all. Respond explaining what you
changed and why.
