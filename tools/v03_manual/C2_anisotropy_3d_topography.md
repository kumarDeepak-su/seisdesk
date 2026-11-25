## Anisotropy (VTI): eta in velocity analysis, NMO, DMO and migration

In shales and other finely layered rocks the velocity depends on the direction of travel. With a vertical symmetry
axis (VTI) the far offsets of a reflection arrive **earlier** than the NMO hyperbola predicts, and steep dips and
diffraction flanks migrate to the wrong place. Time processing describes such layers with two numbers per time:
the NMO velocity and the anellipticity **eta** (Alkhalifah & Tsvankin 1995). Eta = 0 is the isotropic case; values of
0.03-0.2 are common in shale sequences.

Eta can only be measured where the offsets are at least as large as the depth (offset / depth above about 1). At
shorter offsets it trades off with the NMO velocity, so pick the velocity first and eta second.

Step by step:
1. Pick velocities in the velocity analysis (Velocity > Velocity analysis, F6) as usual.
2. At a few CMPs, run the eta scan of the velocity analysis and pick eta where the far offsets bend upwards after
   hyperbolic NMO.
3. Check the result with **Residual moveout semblance scan (anisotropic)** in mode *eta (VTI)* on hyperbolically
   NMO-corrected gathers (below): the brightest eta at each time should match your picks.
4. NMO with **Use picked eta** ticked flattens the far offsets.
5. Instead of NMO + DMO, use **Anisotropic DMO (migration to zero offset, eta)** on gathers *without* NMO.
6. Migrate with **Phase-shift migration for VTI media (eta)** (post-stack) or **Kirchhoff time migration** with
   *Use picked eta* (pre- or post-stack).

**Phase-shift migration for VTI media (eta)** migrates a stacked section with Alkhalifah's acoustic VTI dispersion
relation. The RMS velocities at one CMP are Dix-converted to interval NMO velocity. The picked (effective) eta is
converted to interval eta with Alkhalifah's Dix-type formula. With eta = 0 the result is identical to the isotropic
phase-shift migration.

| Parameter | Meaning and choice |
|---|---|
| Velocity | Project picks (default), a constant or a function; RMS (stacking) velocities |
| CMP for the velocity function | -1 = the middle picked CMP; the migration uses one v(t) and one eta(t) for the whole line |
| Eta | *picked*: effective eta from the velocity picks; *constant*: the interval eta below |
| Constant eta (interval) | 0.05-0.15 for most shales; 0 = isotropic |
| Highest frequency | The top of the signal band (60-90 Hz) |

**Kirchhoff time migration** has two new parameters, **Use picked eta** and **Constant eta**. With eta, each leg of
the double-square-root traveltime uses the non-hyperbolic time of Alkhalifah & Tsvankin (1995). The velocity and eta
can vary along the line. With both parameters left off, the migration is unchanged.

![Phase-shift migration of diffractions in a VTI medium: the zero-offset section (left), migrated with eta = 0
(middle: the flanks are over-migrated into smiles) and with the right eta of 0.12 (right: focused
points).](images/m46_vti_phase_shift.png)

### Anisotropic DMO (migration to zero offset)

**Anisotropic DMO (migration to zero offset, eta)** does NMO and DMO in one step, also in VTI media:
- The input is prestack traces without NMO, with source and receiver x (any sort order).
- The traces are sorted into offset classes.
- Each class is Kirchhoff time-migrated with v_rms and eta, then demigrated (modelled back) to zero offset with the
  same v_rms and eta.
- The output is one corrected trace per CMP and offset class, ready for CMP stack (or the stack itself).
- Dipping events and diffractions then stack at their zero-offset times whatever their dip, and eta also flattens the
  far offsets of anisotropic layers.

| Parameter | Meaning and choice |
|---|---|
| Velocity | Project picks; the result is much less sensitive to velocity errors than a migration |
| Use picked eta / Constant eta | As in NMO; off and 0 = isotropic DMO |
| Offset class width | About the shot interval or 2-4 group intervals; each class must cover the line evenly |
| Aperture | At least the lateral move of the steepest dip: about depth x tan(dip), plus half the largest offset |
| Maximum angle | 60 degrees; lower it to suppress steep migration noise |
| Output | *gathers* (check flatness, then stack) or *stack* (average of the classes) |

Pitfalls:
- Do **not** apply NMO after this step: its output is already NMO- and DMO-corrected.
- The cost is one prestack time migration plus one zero-offset modelling per offset class. Take a few hundred CMPs
  at a time on a laptop.
- The amplitude of each class is matched to its input RMS, which keeps the classes comparable before stacking.
- A coarse CMP spacing aliases steep dips in both the migration and the demigration. Keep the CMP spacing at
  half the receiver interval.

### Residual moveout semblance scan

**Residual moveout semblance scan (anisotropic)** checks whether flattened gathers really are flat. Use it on
NMO-corrected CMP gathers, or on image gathers from Kirchhoff time or depth migration (output *image gathers*).
- At each selected CMP it computes the semblance across the offsets for a range of trial residual moveouts.
- The scan comes out as a dataset with one panel per CMP and one trace per trial value (header `rmo_r`); view it like
  any gather.
- A bright band at r = 0 means the gather is flat.
- A band at positive r means events still curve downwards at the far offsets (velocity too high, or multiples).
- The best value at each time is kept in the project results. The Residual moveout correction step then measures and
  applies a parabolic correction CMP by CMP.

| Parameter | Meaning and choice |
|---|---|
| CMPs | A comma-separated list; empty = three CMPs at a quarter, half and three quarters of the line |
| Residual term | *parabolic*: shift r (x / xmax)^2; *fourth order*: r (x / xmax)^4, for the far-offset hockey stick; *eta (VTI)*: the non-hyperbolic term left after hyperbolic NMO |
| Minimum r, Maximum r | Shift at the far offset in seconds (metres for depth image gathers), e.g. -0.05 to 0.05 s |
| Minimum eta, Maximum eta | Eta mode only, e.g. 0 to 0.3 |
| Number of trial values | 41-81 |
| NMO velocity | Eta mode only: the velocity the gathers were NMO-corrected with (project picks) |
| Semblance window | 20-40 ms, about one wavelet length |

![Residual moveout scan of an NMO-corrected marine CMP (left gather, right semblance panel): the primaries peak at
r = 0, the water-bottom multiples at positive r (undercorrected).](images/m47_rmo_scan.png)

## 3D migration of small volumes

Three steps migrate 3D data on the inline/crossline grid of **3D binning** (headers `iline`, `xline`, `cdpx`, `cdpy`):
- **3D phase-shift migration (post-stack, v(t))**: Gazdag's phase shift with one RMS velocity function, converted to
  interval velocity by Dix's formula. Accurate for steep dips in v(t).
- **3D Stolt migration (post-stack, constant v)**: fast f-k migration for checks.
- **3D Kirchhoff time migration (pre- or post-stack)**: straight-ray double-square-root traveltimes in v_rms(t)
  varying from bin to bin. Prestack input gives the migrated stack.

Step by step:
1. Geometry > 3D binning. Check the bin sizes in the log; they are stored with the data.
2. For the post-stack methods, NMO and CMP stack (the stack has one trace per bin).
3. Run the migration from the Flow editor or Processing > Apply one step.
4. View the result with gather key `iline` (one inline per page) or `xline`.

| Parameter | Meaning and choice |
|---|---|
| Velocity | Project picks, a constant or a function (Stolt: one constant) |
| CMP bin for the velocity function | Phase shift: -1 = the middle picked bin |
| Bin size along an inline / along a crossline | 0 = taken from the 3D binning; give them for data binned elsewhere |
| Aperture (radius) | Kirchhoff: the lateral reach of the steepest dip, e.g. 1000 m |
| Maximum angle | Kirchhoff: 60 degrees |

Pitfalls:
- **Small volumes only.**
  - The post-stack methods hold the cube and its spectrum in memory, padded to 1.5 times each side, as complex
    numbers.
  - For example, 60 inlines x 60 crosslines x 1000 samples needs roughly 1 GB.
  - The Kirchhoff migration visits every bin within the aperture for every input trace.
- Empty bins count as zero traces. A survey with big holes migrates them into smiles, so fill or crop it first.
- Dip and velocity errors in the crossline direction matter as much as inline: pick velocities on both.

![3D migration of a small cube: inline 12 through three diffractors and a dipping plane, unmigrated (left), after 3D
phase-shift migration (middle) and after 3D Kirchhoff time migration (right).](images/m48_3d_migration.png)

## Depth migration from topography

On land lines with rough topography, correcting the data to a flat datum by static shifts assumes vertical rays
through the near surface. Below hills and valleys this distorts the image. **Kirchhoff depth migration** with
**Recording surface = topography** migrates from the real source and receiver positions instead:
- Sources are placed at `selev - sdepth` and receivers at `gelev`.
- Depth is measured below the **highest ground elevation** of the line (reported in the log and stored with the
  section).
- The velocity model starts at that elevation. Cells above the ground take the velocity just below it, and nothing
  is imaged above the ground.

Step by step:
1. Load the geometry with elevations, for example Geometry > Land geometry from shot table and topography.
2. Do **not** apply elevation or refraction statics to a flat datum before this migration. Short-wavelength residual
   statics are fine.
3. Build a velocity model whose top is the highest elevation:
   - a constant or gradient model for a first look;
   - or a depth model built for the line (Velocity > Load depth model).
4. Run Kirchhoff depth migration with Recording surface = topography.
5. To read elevations off the section: elevation = reference elevation - depth.

| Parameter | Meaning and choice |
|---|---|
| Recording surface | *flat*: sources and receivers at z = 0 (data already at a flat datum); *topography*: at their elevations |
| Traveltime grid spacing | Fine enough to follow the topography: no more than the elevation change over two stations, e.g. 5-10 m |
| Image depth step, Maximum depth | Depth below the highest elevation |
| Aperture | As for flat-surface migration |

![A land line over hills: Kirchhoff depth migration assuming a flat surface (left: the flat reflectors copy the
topography) and from the topography (right: flat reflectors at the right depth below the reference elevation of 180
m).](images/m49_psdm_topography.png)

### Rows for the cross-reference to Yilmaz

| Topic | SeisDesk |
|---|---|
| Anisotropy: VTI moveout, eta, anisotropic DMO and migration | Eta scan in the velocity analysis, NMO with eta, residual moveout scan (eta mode), anisotropic DMO by migration to zero offset, VTI phase-shift migration, Kirchhoff time migration with eta |
| Residual moveout analysis of image gathers | Residual moveout semblance scan, residual moveout correction |
| 3D migration: one-pass post-stack, prestack time | 3D phase-shift, 3D Stolt and 3D Kirchhoff time migration (small volumes) |
| Migration from topography (no flat datum) | Kirchhoff depth migration with recording surface = topography |

### Lines for "Limits of version 0.3"

- **Anisotropy:**
  - VTI only (vertical symmetry axis); no tilted (TTI) or orthorhombic media.
  - The phase-shift migration uses the acoustic VTI approximation and one v(t), eta(t) for the whole line.
  - Kirchhoff time migration and the anisotropic DMO let velocity and eta vary along the line.
- **Anisotropic DMO** is a 2D Kirchhoff migration to zero offset: slow on long lines, and steep dips alias at coarse
  CMP spacing.
- **3D migration** is limited to small volumes held in memory. There is no 3D depth migration and no 3D DMO, and 3D
  phase shift uses one velocity function.
- **Migration from topography:**
  - It uses first-arrival (shortest-path) traveltimes, accurate to about 1 %, with one arrival per point (no
    multipathing).
  - The air above the ground is filled with the velocity just below it.
