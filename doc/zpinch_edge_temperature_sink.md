# Edge-temperature sink in `z_pinch_heated`

The heated Z-pinch pgen supports two runtime-selectable outer thermal-energy
sinks:

```text
problem/sink_type = fractional   # default, legacy behavior
problem/sink_type = thermostat
```

The thermostat uses a compact limiter/SOL mask,

```text
chi(r) = 0                                      r <= r_start
       = [1-cos(pi*(r-r_start)/Delta_r)]/2      r_start < r < r_start+Delta_r
       = 1                                      r >= r_start+Delta_r
dEth/dV = -dt*chi*rho*max(p/rho-t_set,0)/((gamma-1)*tau_T).
```

Thus the thermostat is exactly inactive in the confined region, has a smooth
half-cosine onset, and is full strength in the outer SOL. `tau_T` is the
literal relaxation time of the temperature excess where `chi=1`, independent
of box volume and resolution. Only conserved total energy is changed.
Density, momentum, magnetic field, passive scalars, sources, mass control,
and boundary conditions are untouched.

Thermostat inputs are:

```text
problem/t_set = 2.0e-2
problem/tau_T = 1.0e-2
problem/thermostat_r_start = 2.75
problem/thermostat_ramp_width = 0.05
```

The explicit update requires every source-step `dt <= tau_T`; violation causes
a fatal error. A practical production target is `dt/tau_T <= 0.5`, followed
by a `tau_T` convergence pair if the pilot succeeds.

History channels:

- `therm_power`: instantaneous integrated thermostat extraction;
- `edge_T_w`: `sum(chi*T*dV)/sum(chi*dV)`;
- `therm_P_conf`, `therm_P_ramp`, `therm_P_sol_in`, and
  `therm_P_sol_out`: extraction split into the confined region, transition
  ramp, and two equal-width portions of the full-strength SOL;
- `Eth_sink` and `Etot_sink`: include the selected sink as before.
- `mass_bnd_inner`/`mass_bnd_outer`, `Etot_bnd_inner`/`Etot_bnd_outer`, and
  corresponding `ash`/`imp` channels: separate inner and outer face fluxes.
  Positive values mean loss out of the computational domain. Their sum
  reproduces each legacy combined `*_bnd_out` channel.

The default `sink_type=fractional` path retains the historical exponential,
volume-normalized energy-proportional sink and ignores the four compact-mask
parameters. Earlier branch verification found:

- the default fractional path is bitwise identical to its parent-branch
  executable in final fluid datasets;
- a thermostat with a set point above all temperatures agrees with a
  sink-free run to `2.1e-15`;
- matched Q=4.56 thermostat and fractional total-energy budgets close to
  0.88% and 0.70%, respectively.

The revised set point `0.02` comes from extrapolating the smooth
`q_5em03_edge` envelope to the outer edge: with thermal-energy density about
`0.007`, density about `0.2`, and `gamma=5/3`,
`T=(gamma-1)e_th/rho ~= 0.023`. It intentionally ignores the sharp,
sink-dominated edge cells in that baseline.
