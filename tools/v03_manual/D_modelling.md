## Modelling: synthetic data with the geometry of your line

The four **Modelling** steps replace the samples of a dataset with modelled data. They record at the same sources and
receivers (the `sx`, `gx`, elevation and source-depth headers) and keep every header, so the result can be shown, picked
and processed like field data. Typical uses:
- Test a processing flow on data whose answer is known.
- Check a velocity or depth model against the field records: refraction branches, reflection times and moveout.
- Plan a survey: offsets, record length, where reflections and ground roll interfere.
- See what each wave type looks like before you try to remove or enhance it.

| Step | What it models | Speed | Use it for |
|---|---|---|---|
| Acoustic finite-difference modelling (2D) | Every P wave: direct, head (refracted), reflected and diffracted waves, multiples; with the free surface also ghosts and surface multiples | Slow | Realistic shot records; tests of demultiple, refraction analysis and imaging |
| Elastic finite-difference modelling (2D) | P and S waves, converted waves, ground roll (with the free surface) | Slowest | Land noise studies, converted waves |
| Kirchhoff modelling from a depth model | Primary reflections with their diffraction tails | Fast | Checking a depth model, dipping and curved reflectors, migration tests |
| Ray-traced reflection synthetics | Primary reflection times and amplitudes, optionally the first arrivals | Fastest | Comparing a layered model with picks; velocity-analysis and NMO tests |

### Describing the model

Type one of three things in the **Model** field:

1. **Flat layers** as P velocities, a slash, and the depths of the interfaces below the first layer:
   `1500, 2000, 2600 / 300, 700` is 1500 m/s down to 300 m, 2000 m/s down to 700 m, and 2600 m/s below. S velocities
   and densities may follow after two more slashes: `1500, 2000, 2600 / 300, 700 / 0, 1150, 1500 / 1000, 2100, 2300`.
   - Without them, vs = vp / 1.73 and the density follows Gardner's rule (310 vp^0.25 kg/m3).
   - Write vs = 0 for water.
   - Give the density of water (1000) yourself: Gardner's rule is meant for rocks.
2. **A model file**: a text file in the project folder (type its name) or anywhere (type its full path). It has one
   line per layer from the top down: `vp vs rho : x1 z1, x2 z2, ...`. The polyline is the top of the layer; a single
   number after the colon is a flat top. Write `-` for vs or rho to use the defaults. Lines starting with `#` are
   comments. The first line's polyline is the surface, so topography goes there:

   ```
   # vp   vs   rho : top of the layer (x z, x z, ...)
   1800   -    -   : 0 10, 500 0, 1000 -10
   2400   1300 2200 : 0 300, 300 380, 600 330, 1000 250
   3000   -    -   : 0 520, 1000 420
   ```

3. **`project model`**: the project's depth model (from the Build depth model step, or loaded with Velocity > Load
   depth model). It gives the P velocity on a grid. S velocity is vp / 1.73 and density follows Gardner's rule.
   Ray-traced synthetics cannot use it; the other three steps can.

Depths are in metres, positive down from a reference level. A point belongs to the deepest layer whose top lies above
it, so an interface that runs below the next one pinches its layer out. Layers are homogeneous: describe a velocity
gradient with several thin layers, or use the project model.

**Sources, receivers and topography**
- The x positions come from `sx` and `gx`, so run a geometry step first.
- Sources sit `sdepth` (header) below the surface. Receivers sit at the **Receiver depth** below it, for example 8 m
  for a streamer.
- **Surface** = *model surface* uses the top of the first layer.
- **Surface** = *from elevation headers* builds the topography from `selev` and `gelev` (land lines). Depths then
  count from the highest station, including the interface depths of flat layers.

### Step by step

1. Load the line and run its geometry, so that `sx`, `gx` and, for land, `selev` and `gelev` are set.
2. Try one shot first: extract it with **Select traces (header range)** on `fldr`. Finite-difference runs take a
   while on a whole line.
3. Open the step: **Processing > Apply one step to the dataset...**, then a *Modelling* step. In the flow editor the
   steps are under **Modelling**.
4. Fill in the model and the parameters below, and run. The log shows the grid size and the number of grid points per
   shortest wavelength.
5. The output dataset (for example *line1 FD acoustic*) keeps all headers. Display it next to the field data. You can
   pick it, process it or export it as SEG-Y.

![Acoustic finite-difference modelling: the P-velocity model with the shot (red star) and the receivers (left), and
the modelled shot gather (right).](images/m50_fd_acoustic.png)

**Parameters of all four steps**

| Parameter | Meaning and choice |
|---|---|
| Model | Flat layers, a model file, or `project model` (see above) |
| Surface | *model surface*, or *from elevation headers* for a land line with topography |
| Receiver depth | Depth of the receivers below the surface (0 for land, the cable depth for a streamer) |
| Ricker peak frequency | Dominant frequency of the source wavelet. Match the field data (Spectral analysis); a higher frequency makes finite differences slower |
| Record length, sample interval | 0 keeps the input's record length and sample interval |
| Model depth | 0 = 20 % below the deepest interface plus 100 m. Give it for the project model, whose grid may be much deeper than you need |
| Grid spacing | 0 = automatic: six points (finite differences) or four points (Kirchhoff) per shortest wavelength, taken at 2.5 times the peak frequency; about 1/400 of the model size for ray tracing |

**Finite differences**

| Parameter | Meaning and choice |
|---|---|
| Free surface | Acoustic: off by default (the top absorbs). On adds the source and receiver ghosts and the surface multiples; sources and receivers are then moved two cells below the surface, because the pressure there is zero. Elastic: on by default, with vacuum above the surface; it is needed for ground roll |
| Absorbing borders (acoustic) | PML (default) or sponge |
| Absorbing border width | 30 cells (acoustic) or 40 cells (elastic); widen it if reflections from the model edges appear |
| Recorded component (elastic) | *vertical* (land geophones), *horizontal*, or *pressure* |
| 2D to 3D waveform correction | On (default): a half-derivative filter that turns the waveform of the 2D model (a line source) into the point-source Ricker wavelet, with events centred on their traveltimes. Turn it off only to compare with other 2D codes |

The time step is set automatically for stability, and the result is resampled to the output sample interval.
Runtime grows with the number of grid cells times the number of time steps: halving the grid spacing makes a run
about eight times longer. A warning in the log means fewer than five grid points per shortest wavelength. Expect
numerical dispersion then (high frequencies arrive late and ring): use a smaller grid spacing or a lower frequency.

![Elastic finite-difference modelling of the same shot, vertical component with a free surface: P arrivals and the
slow ground roll.](images/m51_fd_elastic.png)

**Kirchhoff modelling**
- Every reflector point of the model, on a depth grid, contributes its normal-incidence reflection coefficient at the
  traveltime source to point to receiver.
- The traveltimes are first arrivals through the model. The contributions are weighted for spreading and obliquity,
  then convolved with the Ricker wavelet.
- **Aperture** (half width around the midpoint, default 2000 m) limits which reflector points are summed; reduce it
  to save time on long lines.

**Ray-traced reflection synthetics**
- **Ray tracing** *auto* uses exact two-point ray tracing (Snell's law) when all layers and the surface are flat, and
  the *minimum time* method otherwise. The minimum-time method computes traveltimes through the layers above each
  interface and takes the minimum of source time plus receiver time along it.
- **Reflection coefficients**: *normal incidence*, or *angle (Aki-Richards)* for amplitude variation with offset.
- **First arrivals** adds the direct and head waves, with amplitude 1/distance.
- The event times are kept in the project results, so they can be compared with picks.

![Ray-traced synthetics with first arrivals (left) and acoustic finite differences (right) for the same flat-layer
model. The reflection times agree; the ray synthetics hold only the primary reflections and the first-arrival
branch.](images/m52_ray_vs_fd.png)

![Kirchhoff modelling from a model file with a curved and a dipping reflector: the model (left) and a modelled shot
gather (right).](images/m53_kirchhoff_modelling.png)

**Pitfalls**
- **Low near-surface velocities make finite differences slow.** A weathering layer of 300-800 m/s forces a small grid
  spacing. Use a lower peak frequency, accept a coarser grid (with dispersion), or use ray or Kirchhoff modelling.
- **The elastic grid follows the slowest S velocity.** The default vs = vp / 1.73 is often too high for loose
  near-surface material; give vs explicitly.
- **Edge reflections.** The model extends 20 cells beyond the outermost stations, plus the absorbing borders. Long
  records can show weak reflections from the borders; widen them.
- **Kirchhoff and ray modelling are primaries only.** They have no multiples and no converted waves, and their
  amplitudes are approximate. Kirchhoff uses first-arrival traveltimes, so reflector points beyond the critical
  distance can add weak artefacts.
- **Amplitudes are relative.** Every result is scaled to a maximum of one.

**Rows for the Yilmaz cross-reference**

| Topic | SeisDesk |
|---|---|
| Forward modelling by the wave equation (acoustic and elastic finite differences) | Acoustic finite-difference modelling (2D), elastic finite-difference modelling (2D) |
| Ray-trace and Kirchhoff modelling of reflections, model checking | Ray-traced reflection synthetics, Kirchhoff modelling from a depth model |

**Lines for "Limits of version 0.3"**
- Modelling is 2D. The waveform correction makes line sources look like point sources, but there is no 3D modelling.
- All four methods assume isotropic media without attenuation: no Q and no anisotropy.
- Finite differences use explosive point sources only. The elastic free surface uses the vacuum method, which needs a
  finer grid than the body waves for accurate ground roll. Runtime grows quickly with frequency and model size.
- Kirchhoff modelling and ray-traced synthetics give primary reflections only, with approximate amplitudes. Ray
  synthetics need a layered model. The minimum-time ray method follows the fastest paths, so near the critical
  distance its times can follow refracted rather than reflected paths.
- Models are homogeneous layers or the project depth model; a velocity gradient inside a layer needs several thin
  layers.
