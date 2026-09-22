Your previous hardware change succeeded and was validated end-to-end on
real FPGA hardware via FireSim, with the results below. These numbers are
your previous hardware paired with the best schedule the automated SW
search (Exo + AlphaEvolve) could find for it, not a fixed kernel -- read
them as feedback on the hardware, per the Co-Design Loop section of your
instructions. Using the chipyard_bash tool, try to further improve the
accelerator: reduce the cycle count and/or improve accuracy through
hardware parameter changes. Actually call the tool to edit the files
yourself -- don't just describe the change or hand back a script; if you
don't call the tool, no change happens at all. Respond explaining what you
changed and why.
