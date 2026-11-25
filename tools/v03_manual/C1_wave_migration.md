## Wave-equation depth migration

These steps continue the recorded wavefield down through the depth model, one depth step at a time. They follow
strong lateral velocity changes (salt, basalt, karst, steep faults, channels) better than Kirchhoff migration with
straight rays or first-arrival traveltimes. Four methods migrate a stack (post-stack); one migrates the shot gathers
before stack (prestack). All of them are in the **Depth migration** category of the Flow editor and of
**Processing > Apply one step to the dataset...**.

| Method | Input | Dips | Lateral velocity changes | Cost |
|---|---|---|---|---|
| Split-step (since version 0.1) | stack | all, exact where the velocity is laterally constant | moderate | lowest |
| Finite difference, 45 or 65 degrees | stack | up to 45 or 65 degrees | strong | low |
| Fourier finite difference | stack | steep | strong | moderate |
| Phase shift plus interpolation (PSPI) | stack | all | strong, with enough reference velocities | grows with the velocity range |
| Gaussian beams | stack | up to the maximum ray angle | smooth model; caustics and several arrivals | grows with the number of beams |
| Prestack shot profile | shot gathers | steep | strong | highest (every shot is migrated) |

### Post-stack: finite difference, Fourier finite difference, PSPI and Gaussian beams

1. Build the depth model first (**Build depth model**, see *Depth model and depth migration*). It is the *project
   model*. Without one, the step builds a Dix model from the velocity picks.
2. Select the stacked section in the Project panel. It needs one trace per CMP and the 2D geometry (CMP x or the bin
   size).
3. Choose the step and set the parameters (table below). Run it on the whole section: these steps work on all traces
   together, so *displayed gather only* gives no useful test.
4. Compare the result with the split-step migration of the same stack. Check reflector depths against wells or
   refraction depths.

| Parameter | Meaning and choice |
|---|---|
| Velocity model | *project model* (recommended), the velocity picks (Dix) or a file |
| Depth step | 5-10 m. At most a quarter of the shortest wavelength, v_min / (4 f_max) |
| Maximum depth | Below the deepest reflector of interest. Run time grows with the number of depth steps |
| Highest frequency | The top of the useful band (spectral analysis), e.g. 40-60 Hz. Run time grows with it |
| Smoothing of a Dix model | 100-300 m; removes Dix noise |
| Dip accuracy (finite difference) | 65 degrees (default) or 45 degrees. Both cost the same; steeper dips are under-migrated |
| Maximum reference velocities (PSPI) | 8. More for strong velocity contrasts within one depth step |
| Lowest frequency (Gaussian beams) | 5 Hz. The band's centre frequency sets the beam width |
| Initial beam width (Gaussian beams) | 0 = automatic, about one wavelength at the centre frequency. Wider beams stay narrow deeper down |
| Maximum ray angle (Gaussian beams) | 60 degrees; up to 75 degrees for steep flanks |
| Model smoothing for the rays (Gaussian beams) | 300 m; at least one wavelength. Rays scatter in a rough model |

How the methods differ:
- **Finite difference** solves the 45-degree (Claerbout) or 65-degree (Lee and Suh) one-way wave equation in
  frequency and space. The velocity can change from trace to trace.
- **Fourier finite difference** (Ristow and Rühl) shifts the phase with the lowest velocity of each depth step,
  corrects the time shift for the true velocity, and adds a finite-difference correction for the rest. It is accurate
  for steeper dips than the finite-difference and split-step methods.
- **PSPI** (Gazdag and Sguazzero) continues the wavefield with a few reference velocities about 10 % apart and
  interpolates between them with the true velocity at each trace.
- **Gaussian beams** (Hill) split the stack into local plane waves under overlapping Gaussian windows. Each plane wave
  is sent down along a ray traced in the smoothed model and spread as a Gaussian beam. Beams handle caustics and
  several arrivals, which single-arrival Kirchhoff migration does not.

![One stack migrated by finite difference (45 degrees), Fourier finite difference and PSPI.](images/m42_wave_migration_poststack.png)

![Finite difference at 45 and 65 degrees against split-step on three point diffractors. The 45-degree equation leaves
the steep flanks of the deeper diffractions as smiles; the 65-degree equation collapses more of them.](images/m45_fd_dip_limit.png)

![Gaussian-beam migration compared with split-step on the same stack.](images/m44_gaussian_beam.png)

### Prestack: shot-profile wave-equation migration

**Prestack wave-equation depth migration (shot profile)** migrates each shot gather on its own:
- the source wavefield starts at the source position;
- the recorded wavefield starts at the receivers;
- both are continued down through the model, and their zero-lag correlation at each depth is the image of that shot.

The images of all shots are summed. Prestack migration handles events that a stack has already smeared (conflicting
dips, strong lateral velocity changes), at a much higher cost.

1. Use shot gathers with source and receiver x (any geometry step) and the recording time zero at the shot. Mute the
   direct wave and first arrivals. Do not apply NMO.
2. On land, apply elevation or refraction statics to a flat datum first: the recording surface is taken as flat, and
   depths are measured below it.
3. Run the step on the whole dataset; it migrates the shots one by one. For a first look, select every second or
   fourth shot (**Select traces**) and lower the highest frequency.

| Parameter | Meaning and choice |
|---|---|
| Extrapolator | *split-step* (fastest), *PSPI* or *FFD* (steepest dips in strong lateral velocity changes) |
| Depth step, maximum depth | As for the post-stack methods |
| Image trace spacing | 0 = the receiver spacing |
| Aperture beyond the spread | 500-2000 m: at least the lateral distance that dipping reflectors move |
| Lowest / highest frequency | 3-40 Hz. Run time grows with the number of frequencies, shots and depth steps |
| Source wavelet | *impulse*: the image keeps the wavelet of the data. *Ricker*: a smoother image |
| Divide by the source illumination | On (recommended): balances shallow and deep, and near and far, amplitudes |

![A stack migrated post-stack by split-step, and ten shots migrated prestack by the shot-profile method. Multiples
were not removed from the shots, so they are migrated too.](images/m43_wave_psdm.png)

Pitfalls:
- A velocity that is too slow under-migrates: diffractions are not fully collapsed and dipping events stay too
  shallow. Too fast over-migrates and draws smiles.
- Steep dips at high frequencies alias in space when the trace spacing exceeds v / (4 f sin(dip)). Lower the highest
  frequency, or interpolate traces first.
- About 20 traces of absorbing padding are added on each side, so energy leaving the section is not wrapped back.
  The outermost traces of a short line are therefore weaker.
- Run time grows with depth steps × frequencies × traces, and with the number of shots for prestack. Test with a
  coarser depth step and a lower highest frequency first.

Cross-reference to Yilmaz (rows for the table):

| Topic | SeisDesk |
|---|---|
| Finite-difference migration (15/45-degree equations, implicit schemes) | Finite-difference depth migration (45/65 degrees) |
| Depth migration by split-step, phase shift plus interpolation and Fourier finite difference | Split-step, PSPI and Fourier finite-difference depth migration |
| Prestack depth migration by wavefield extrapolation (shot profile) | Prestack wave-equation depth migration (shot profile) |
| Gaussian-beam migration | Gaussian-beam migration (post-stack) |

Limits of version 0.3 (lines for the chapter):
- The wave-equation migrations are 2D and isotropic, and they assume a flat recording surface. On land, apply statics
  or datuming to a flat datum first.
- The prestack wave-equation migration outputs a stacked image only, without offset or angle gathers. Check residual
  moveout with Kirchhoff PSDM image gathers.
- Gaussian-beam migration is post-stack only and simplified: one beam width for all frequencies, real traveltimes and
  no KMAH phase.
- The 45- and 65-degree finite-difference equations under-migrate steeper dips. All finite-difference schemes show
  some numerical dispersion at high frequencies on coarse trace spacing.
