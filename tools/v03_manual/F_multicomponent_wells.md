## Multicomponent data (3C and 4C)

Three-component (3C) geophones record the vertical motion and two horizontal motions. Four-component (4C) shear
surveys record two orthogonal shear sources on two orthogonal horizontal receivers. The Multicomponent steps (flow
editor, category **Multicomponent**, or Processing > Apply one step) rotate the horizontal components, measure the
polarisation of the motion and filter by it.

**How SeisDesk tells the components apart.**
- A header holds the component code. By default this is the SEG-Y trace identification code `trid`:

  | trid | Component |
  |---|---|
  | 12 | vertical (Z) |
  | 13 | crossline (horizontal, 90 degrees anticlockwise from inline) |
  | 14 | inline (horizontal, along the sensor azimuth) |
  | 15, 16, 17 | rotated vertical, transverse, radial (written by the rotation) |

  If your files carry the component in another header, give that header and its codes in the step.
- The traces of one station and one shot are matched by **match keys**, `fldr,tracf` by default. When the channels
  are numbered per component (channel 1 = Z, 2 = inline, 3 = crossline of station 1, and so on), use `fldr,gx`
  (or `fldr,gx,gy`) instead. Traces without all their components pass unchanged.
- For 4C shear data there is no SEG-Y code for the source orientation, so the default is a header `comp` with codes
  1, 2, 3, 4 for XX, XY, YX, YY: the first letter is the source, the second the receiver. Set it with
  Geometry > Header formula, or in the shot table.

Check the codes in the header table (Geometry > Header table) before you run a step: the most common problem is a
file where `trid` is 1 on every trace.

### Horizontal rotation to radial and transverse

**Horizontal rotation (3C)** turns the inline and crossline components into:
- **radial**: along the direction from the source to the receiver, positive away from the source;
- **transverse**: 90 degrees anticlockwise from radial, seen from above.

After rotation, P and SV energy (P-to-S converted waves, Rayleigh waves) sits on the radial component and SH energy
(Love waves, SH reflections) on the transverse. This is the first step for converted-wave (PS) processing. It is also
the test of sensor orientation: a correctly rotated line has little coherent energy on the transverse component.

| Parameter | Meaning and choice |
|---|---|
| Rotation | *source-receiver azimuth*: from sx, sy, gx, gy for each trace (needs the geometry). *fixed angle*: one angle for all traces. |
| Fixed angle | Angle from the inline axis to the new radial axis, anticlockwise (degrees) |
| Sensor azimuth | Direction of the inline sensor axis, degrees anticlockwise from the survey x axis (0 when the inline component points along +x, the usual case for a 2D line) |
| Component header, codes | `trid`, 14 inline, 13 crossline |
| Match keys | Headers equal for the components of one station (`fldr,tracf`, or `fldr,gx`) |

- The rotated traces get the codes 17 (radial) and 16 (transverse).
- The angle used is written to header `hrot_az`.
- A receiver at the source position has no azimuth. It is rotated by the sensor azimuth only, and the log says how
  many there were.

![3C rotation: a shot recorded with the source 150 m off the receiver line, inline and crossline components before
rotation and radial and transverse after. The P and converted events move to the radial component and only the SH
event stays on the transverse.](images/m58_3c_rotation.png)

### Alford rotation of 4C shear data

**Alford rotation (4C shear)** (Alford 1986) is for shear-wave splitting studies, for example fractured reservoirs or
stress-aligned cracks.
- In an azimuthally anisotropic layer, a shear wave splits into a fast and a slow wave polarised along the symmetry
  axes. The off-diagonal components XY and YX then carry energy.
- The step finds, in a time window, the angle that minimises the off-diagonal energy and rotates all four
  components to it.
- After rotation, XX and YY are the fast and slow shear waves. The delay between them measures the strength of the
  anisotropy; the angle gives the fast direction.

| Parameter | Meaning and choice |
|---|---|
| Component header, codes | `comp` with 1, 2, 3, 4 = XX, XY, YX, YY |
| Match keys | `sx,gx`: the two source orientations are often separate records, so match by position |
| Window start, end | The shear arrivals of the layer you analyse; 0 = whole trace |
| One angle per gather of | Empty: an angle for every station. A key (for example a receiver line) pools the stations of a gather for a steadier angle |
| Put the fast shear wave on XX | Compares the arrival times of the rotated XX and YY and adds 90 degrees if YY arrives first |

- The angle (degrees anticlockwise from the acquisition x axis) is written to header `alford_az`.
- The log gives the range of angles and the off-diagonal energy before and after. A good fit drops it to a few
  per cent.
- For layer stripping, rotate and correct the shallow layer first, then analyse the next window.

### Polarisation analysis and polarisation filter

**Polarisation analysis (3C)** computes the covariance matrix of the vertical, inline and crossline components in a
sliding window (Jurkevics 1988). From its eigenvalues l1 >= l2 >= l3 and main eigenvector it outputs one attribute
per run:

| Attribute | Meaning |
|---|---|
| rectilinearity | 1 - (l2 + l3) / (2 l1): about 1 for linear motion (P and S body waves), about 0.5 for circular motion (ground roll) |
| planarity | 1 - 2 l3 / (l1 + l2): about 1 when the motion lies in a plane |
| azimuth | Direction of the main motion, degrees from the inline axis towards the crossline axis |
| incidence | Angle of the main motion from the vertical (0 = vertical motion, near-vertical P reflections) |
| largest eigenvalue | Energy in the window |

The output has one attribute trace per station, with the headers of its vertical trace. Display it with a colour map
(Rainbow) and a fixed scale: rectilinearity runs from 0 to 1.

**Polarisation filter (3C)** multiplies all three components by a weight:
- The weight is the rectilinearity raised to a **power**: 2 is a moderate filter, 4 a strong one.
- Optionally the weight also keeps only **near-vertical** motion (P reflections), or only **near-horizontal**
  motion. This uses the incidence limit, with a 15-degree taper beyond it.
- Linear body waves pass. Elliptical ground roll and incoherent noise are attenuated.

| Parameter | Meaning and choice |
|---|---|
| Window | 1-2 periods of the waves to separate: 0.05 s at 20-40 Hz, longer for low-frequency ground roll |
| Rectilinearity power | 2 (moderate) to 4 (strong) |
| Direction | *any*; *near-vertical* to keep P reflections; *near-horizontal* for shear or surface waves |
| Incidence limit | 30-40 degrees for P reflections |

**Pitfalls.**
- Too short a window gives noisy attributes. Too long a window mixes neighbouring events.
- Interfering events of different polarisation look elliptical and are attenuated too.
- Use the filter on raw shots before stack, and compare with an f-k or tau-p filter.

![Polarisation: the vertical component of a 3C shot with P reflections and elliptical ground roll, its
rectilinearity (0 blue to 1 red), and the vertical component after the polarisation filter (power 2, near-vertical
incidence).](images/m59_polarisation.png)

## Well logs: synthetic seismogram, well tie, Backus and Gassmann

The well-log window (View > Well logs and synthetic seismogram) loads the logs of one well from a LAS file. It makes
a synthetic seismogram, ties it to a seismic trace at the well, and offers two rock-physics tools: Backus averaging
and Gassmann fluid substitution.

**Step by step.**
1. **Load LAS...** reads a LAS 2.0 (or 1.2) file:
   - Curves, units and NULL values are read; NULL values become gaps.
   - Depth may be in metres or feet.
   - Wrapped files (WRAP YES) are not supported: save the log unwrapped.
2. Check the curves SeisDesk chose. It recognises the usual mnemonics:

   | Curve | Mnemonics recognised |
   |---|---|
   | Sonic | DT, DTC, DTCO, AC, ...; or a velocity curve VP |
   | Density | RHOB, RHOZ, DEN, ... |
   | Shear | DTS, DTSM, ...; or VS |

   - The units decide the conversion: us/ft or us/m for slowness, m/s or ft/s for velocity, g/cc or kg/m3 for
     density.
   - Without a density log, choose *(Gardner from Vp)*.
   - Without a shear log, choose *(mudrock line from Vp)* (Castagna et al. 1985).
3. **Time-depth.** Two-way time comes from integrating the sonic from the top of the log.
   - The unlogged section above it is given by **velocity above the log**. Its top sits at
     2 z_top / velocity.
   - **Load checkshots...** corrects the sonic times: give a table of depth (m) and time, one-way or two-way, in ms
     or s. The drift between the checkshot times and the integrated sonic is interpolated in depth and added.
4. Choose the **seismic** dataset (a stacked or migrated time section) and the **x position of the well**. The trace
   nearest to that position is used.
5. **Compute synthetic and tie.** SeisDesk then:
   - computes the impedance (Vp x density) and the reflection coefficients at every log sample;
   - places them at their two-way times, at the seismic sample interval;
   - convolves them with a zero-phase Ricker wavelet of the chosen peak frequency (match the dominant frequency
     of the seismic, from the spectrum window).

   The readout gives the **best shift** (positive: the seismic events come later than the synthetic) and the
   **correlation** at that shift, found within the maximum shift. The synthetic track is drawn with the shift
   applied. If the reversed polarity correlates better, the readout says so: check the polarity convention of the
   data.
6. **Export** the synthetic (time, reflectivity, synthetic, tied time) or the time-depth table (depth, two-way time,
   one-way time in ms) as CSV.

![The well-log window: Vp, density and impedance against depth, and the reflectivity, the synthetic (shifted by the
tie) and the seismic trace at the well against two-way time. The readout gives the shift and the
correlation.](images/m60_well_log_window.png)

| Setting | Recommended value |
|---|---|
| Velocity above the log | The average velocity from the datum to the top of the log, or use checkshots |
| Ricker peak | The dominant frequency of the seismic at the target |
| Max shift | 50-150 ms; a larger best shift usually means a datum or time-depth problem |

**Pitfalls.**
- The depth of the log must refer to the seismic datum. Logs measured from the kelly bushing need the KB elevation
  above datum subtracted (or a checkshot table in the same reference).
- Washouts make bad density and sonic values, so edit or smooth them first.
- A correlation below about 0.5 means the tie is not reliable.

### Backus averaging and Thomsen parameters

The **Backus / Thomsen** tab replaces finely layered isotropic logs by the equivalent transversely isotropic (VTI)
medium (Backus 1962), over a moving depth window.
- It plots the Thomsen (1986) parameters epsilon (P anisotropy), delta (near-vertical P moveout) and gamma (SH
  anisotropy) against depth, with the vertical velocities Vp0 and Vs0 of the averaged medium.
- Use them to judge how much layer-induced anisotropy to expect in velocity analysis (eta, from epsilon and delta)
  and in depth migration.
- The **averaging length** should not exceed about a third of the shortest wavelength, which is the slowest shear
  velocity divided by the frequency (Liner and Fei 2006). The tab shows this suggested maximum for the current log
  and Ricker frequency.
- Identical layers give zero anisotropy. Alternating fast and slow layers give positive epsilon and gamma.

### Gassmann fluid substitution

The **Gassmann** tab predicts the logs, and the synthetic, when the pore fluid changes, for example brine to gas
(Gassmann 1951).
- The dry-rock bulk modulus comes from the measured logs. The shear modulus does not change. The density changes
  with the fluid density times the porosity.

| Input | Typical values |
|---|---|
| Porosity | A porosity curve (PHIE, PHIT, NPHI; percent is recognised) or a constant |
| Zone top and bottom | The reservoir interval; bottom 0 = the whole log |
| Mineral bulk modulus | Quartz 36.6 GPa, calcite 70.8 GPa, clay about 21 GPa |
| Fluid 1 (in the logs) | Brine 2.8 GPa, 1.09 g/cc |
| Fluid 2 (substituted) | Gas 0.04 GPa, 0.25 g/cc; oil about 1.0 GPa, 0.8 g/cc |

**Substitute** plots Vp, Vs and density before and after, and the synthetic before and after. A gas sand lowers Vp
and density and slightly raises Vs. Samples with porosity up to 1 % keep their values.

![Gassmann substitution of brine by gas in a sand: Vp, Vs and density before (grey) and after (red), and the two
synthetics.](images/m61_well_gassmann.png)

---

Rows for the "Cross-reference to Yilmaz" table:

| Topic | SeisDesk |
|---|---|
| 3C recording, horizontal component rotation | Horizontal rotation (3C) to radial and transverse |
| Shear-wave splitting, 4C rotation | Alford rotation (4C shear) |
| Polarisation analysis and filtering | Polarisation analysis (3C), polarisation filter (3C) |
| Synthetic seismograms, well ties | Well-log window: synthetic seismogram, checkshot correction, tie to a seismic trace |
| Layer-induced anisotropy | Backus average with Thomsen epsilon, delta and gamma |
| Rock physics, fluid substitution | Gassmann fluid substitution with synthetic before and after |

Lines for "Limits of version 0.3":

- The multicomponent steps rotate and filter but do not do converted-wave (PS) binning, PS NMO or PS stacking.
- The Alford rotation uses one angle per station (or gather) and window; no automatic layer stripping.
- The polarisation filter treats each station on its own; no multi-station polarisation (no 3C f-k filtering).
- LAS 2.0 and 1.2 files only, unwrapped (WRAP NO). No LAS 3.0, DLIS or LIS.
- The synthetic is a normal-incidence (zero-offset) convolution with a Ricker or a given wavelet. There is no
  wavelet extraction from the seismic, no stretch-and-squeeze, and no offset-dependent (AVO) synthetic.
- Gassmann assumes a homogeneous mineral, connected porosity and low frequency; no mixing of several minerals or
  fluids.
