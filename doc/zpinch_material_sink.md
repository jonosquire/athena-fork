# Independent material sink for `z_pinch_heated`

The heated Z-pinch problem generator can remove material independently of the
mass controller and of either energy-loss model.  The sink has the same
normalized outer-edge exponential shape used by the fractional energy sink,
so it is useful for comparing central fueling and edge exhaust with the
standard edge-fueled calculations.

Enable it in the `<problem>` block with

```ini
mass_sink_flag          = 1
sink_rate_dens          = 5.2
mass_sink_density_floor = 0.03
mass_sink_max_fraction  = 0.1
```

`sink_rate_dens` is the coefficient multiplying the normalized sink shape.
`mass_sink_density_floor` is a cutoff for this explicit removal operation: the
sink will not lower a cell below this density.  It is not an equation-of-state
floor.  `mass_sink_max_fraction` limits the fraction removed in one source
update.  All options default to values that leave the material sink disabled,
so existing inputs are unchanged.

The update removes density, momentum, passive-scalar mass, and the associated
kinetic and thermal energy in the same fraction.  It therefore preserves the
local velocity, temperature, and passive-scalar concentrations while leaving
the magnetic energy unchanged.  It can be combined with core fueling, either
energy sink, or no explicit energy sink (`sink_type = fractional` and
`sink_rate_energy = 0`).

The following history variables audit the requested and realized removal:

- `mass_sink_req`, `mass_sink`, and `mass_sink_clip_vol`;
- `ash_mass_sink` and `imp_mass_sink`;
- `Etot_mass_sink`;
- `mass_bnd_inner`, `mass_bnd_outer`, `Etot_bnd_inner`, and
  `Etot_bnd_outer` (positive means loss from the domain).

The legacy combined boundary-loss histories remain available.  The realized
material-sink rate should be used when comparing runs because the density
cutoff can make a nominally strong sink supply limited.

Local regression tests verified mass and energy closure in serial and MPI,
agreement of the separated and combined boundary channels, passive-scalar
removal, the density cutoff, the no-energy-sink limit, and unchanged behavior
when the new sink is disabled.  A short unsaturated scan found that a cutoff
near `0.03` prevented the most severe edge-density depletion with little
effect away from the sink layer, whereas `0.05` already clipped much of the
requested removal.  Production runs should recheck this at saturation.
