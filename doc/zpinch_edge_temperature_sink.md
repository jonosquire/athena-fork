# Edge-temperature sink in `z_pinch_heated`

The heated Z-pinch pgen supports two runtime-selectable outer thermal-energy
sinks:

```text
problem/sink_type = fractional   # default, legacy behavior
problem/sink_type = thermostat
```

The thermostat uses

```text
chi(r) = exp((r-rout)/sink_width)
dEth/dV = -dt*chi*rho*max(p/rho-t_set,0)/((gamma-1)*tau_T).
```

Here `chi(rout)=1`: it is the fractional sink's normalized weight multiplied
by the precomputed sink volume integral. Consequently `tau_T` is the literal
relaxation time of the temperature excess at the boundary, independent of box
volume and resolution. Only conserved total energy is changed. Density,
momentum, magnetic field, passive scalars, sources, mass control, and boundary
conditions are untouched.

Thermostat inputs are:

```text
problem/t_set = 1.6e-3
problem/tau_T = 5.0e-3
```

The explicit update requires every source-step `dt <= tau_T`; violation causes
a fatal error. A practical production target is `dt/tau_T <= 0.5`, followed by
a `tau_T` convergence pair. The example input
`inputs/mhd/athinput.z_pinch_heated_thermostat` uses a value just large enough
for its inexpensive local mesh. Higher-resolution runs can use a smaller
value after checking their actual timestep history.

History channels:

- `therm_power`: instantaneous integrated thermostat extraction;
- `edge_T_w`: `sum(chi*T*dV)/sum(chi*dV)`;
- `Eth_sink` and `Etot_sink`: include the selected sink as before.

Local verification on the `edge-temp-sink` branch found:

- the default fractional path is bitwise identical to its parent-branch
  executable in final fluid datasets;
- a thermostat with a set point above all temperatures agrees with a
  sink-free run to `2.1e-15`;
- reducing `tau_T` from `0.005` to `0.00125` reduces the measured excess
  edge temperature by a factor 3.96;
- matched Q=4.56 thermostat and fractional total-energy budgets close to
  0.88% and 0.70%, respectively.

The set point `1.6e-3` is the final-100 sink-weighted edge temperature of the
existing Q=0.114 fractional baseline. Values `tau_T=1-5` are numerically
stable but do not clamp this low temperature: useful values in the tested
normalization are of order `1e-3` to `1e-2`.
