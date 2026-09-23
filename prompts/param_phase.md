## This step: parameters

The RTL changes described below are in place and accepted. In this step, tune
Gemmini's parameters in `generators/gemmini/src/main/scala/gemmini/Configs.scala`
(the ones listed under Tuning Strategy) to get the most out of that design.
Do not edit other RTL here. Keep the arithmetic unchanged: leave the datatypes
and widths (`inputType`, `weightType`, `accType`, `spatialArrayOutputType`) and
the scaling (`acc_scale_args`, `mvin_scale_args`) as they are, and keep `DIM`
at 16.

The change is judged the same way as an RTL change: a Verilator simulation of
the fixed, known-good schedules must reproduce the seed's predictions bit for
bit, and the mean cycles per patch must go down. A rejected change is reverted
automatically before your next step, and you get the reason.
