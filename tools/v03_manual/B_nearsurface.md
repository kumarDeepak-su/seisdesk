## Reduced time and linear moveout

Refraction work is easier in **reduced time**, t' = t - |offset| / v: a refracted branch whose apparent velocity equals
the reduction velocity becomes horizontal, its intercept time can be read directly, faster branches dip upwards and
slower ones downwards. SeisDesk offers it in two ways.

**In the viewer (display only).** Set **Reduce** in the display bar to the reduction velocity (for example the
velocity of the first refractor, 2000 m/s); "off" (0) returns to true time.
- Only the display changes; the data and the picks keep their true times.
- The read-out under the display shows both the reduced time and the true time of the point under the cursor.
- Picking pauses while reduced time is shown: switch it off to pick.

**As a processing step: Linear moveout (reduced time)** (flow editor, category Moveout):
- *forward* applies t' = t - |offset| / v + shift; *inverse* undoes it exactly. Use the pair around a step that works
  better on flattened events (for example a median filter across the traces, or an f-k or tau-p filter on linear
  noise).
- *dynamic (dispersion curve)* moves every frequency by its own phase velocity, from a dispersion curve
  `f:v, f:v, ...` (Hz, m/s; for example the picks of the MASW window). A dispersed surface-wave train collapses to a
  short pulse at the shift time: it can then be muted or filtered and the inverse step restores the rest
  (Seismic Unix sudlmo).

| Parameter | Meaning and choice |
|---|---|
| Reduction velocity | The apparent velocity that becomes flat: the refractor velocity, or the velocity of the linear noise |
| Time shift added | 0 for refraction displays; 0.1-0.2 s in dynamic mode, so that the collapsed pulse is not cut at time zero |
| Direction | forward, or inverse to restore the original times |
| Mode | one velocity, or dynamic with a dispersion curve |
| Dispersion curve | f:v pairs in Hz and m/s, used in dynamic mode; constant beyond its first and last frequency |
| Offset header | offset (the absolute value is used) |

Data shifted before time zero or past the end of the record are lost; keep the record long enough, or add a shift.

![A land shot in true time (left) and in reduced time at 2000 m/s (right): the first refracted branch becomes flat at
its intercept time, the faster deeper refractor dips upwards.](images/m38_reduced_time.png)

## Refraction convolution section

The **refraction convolution section** (Palmer 2001) images a refractor directly from the traces, without picking
every first break.
- At each receiver the trace of a forward shot (on one side) and the trace of a reverse shot (on the other side) are
  convolved. Convolution adds travel times, so the head waves give t_forward + t_reverse.
- Subtracting the reciprocal time (the time from the forward shot to the reverse shot) leaves twice the time-depth
  under the receiver (the GRM time-depth, i.e. the two-way delay time).
- The product of the two amplitudes cancels most of the geometric spreading, so amplitude changes along the
  refractor reflect the refractor itself (head coefficients, wavespeed contrasts).

Depth to the refractor at a receiver: z = (time on the section / 2) x V1 V2 / sqrt(V2^2 - V1^2), with V1 the
weathering velocity and V2 the refractor velocity (from the refraction analysis).

Run **Refraction convolution section** (Processing menu, or the flow editor, category Statics) on the shot gathers
with geometry (fldr, sx, gx). Step by step:
1. Look at the shots in reduced time and note the crossover distance beyond which the first arrival is the refractor
   you want.
2. Choose *automatic* pairs with **Minimum offset** just beyond that crossover distance: every receiver is paired with
   the nearest shot on each side at least that far away. With only two shots at the ends of a spread, use *two
   shots* and give their FFIDs.
3. Choose where the reciprocal time comes from: *first breaks of the traces* (measured automatically on the trace of
   each shot recorded at the other shot's position), *picks* (the picked refracted phase, best when you have picked
   it), or *value* (one known time, two-shot mode).
4. Optionally set **Keep after the first break** (for example 0.05 s) to convolve only the first-arrival part of the
   traces, which gives a cleaner image.

| Parameter | Meaning and choice |
|---|---|
| Shot pairs | automatic (nearest shot on each side) or two shots (given FFIDs) |
| Minimum offset | Beyond the crossover distance of the target refractor, so that its head wave is the first arrival |
| Reciprocal time from | first breaks of the traces, picks of the refracted phase, or a value |
| First-break energy window | Window of the automatic first-break measurement, about one period of the wavelet (0.01-0.03 s) |
| Keep after the first break | 0 = whole traces; 0.03-0.1 s keeps only the first arrival and its coda |
| Output length | Length of the section (0 = as the input records) |

The output has one trace per receiver (gx, cdpx = receiver x) and headers rcs_fwd, rcs_rev (the two FFIDs) and rcs_trec
(the reciprocal time in ms); the log and the results list the reciprocal time used for every pair. Measure the
reciprocal time the same way as the arrivals: onsets (first breaks) for onset-type data. A wrong reciprocal time moves
the whole section of that pair up or down by the error.

![Forward and reverse shots over a weathering layer of varying thickness, and their refraction convolution section: the
refractor at twice its time-depth under every receiver.](images/m39_refraction_convolution.png)

## Wave-equation datuming

Elevation and refraction statics move every trace by a vertical time shift. That is exact only for vertical rays; with
rough topography, a thick or fast near surface, or long offsets the shifts distort the moveout and smear the
diffractions. **Wave-equation datuming** (Berryhill 1979, 1984) instead extrapolates the recorded wavefield from the
topography to a flat datum with a constant velocity, following the real ray directions.

**Wave-equation datuming (Kirchhoff, prestack)** (Processing menu, or the flow editor, category Datuming):
- Receivers are moved first, in every common-shot gather (from their elevations, gelev), then sources, in every
  common-receiver gather (from selev minus sdepth). *receivers only* or *sources only* run one half.
- A datum below the surface is a downward continuation: give the velocity of the layer between the surface and the
  datum (for example the weathering velocity from the refraction analysis), and the step removes that layer. A datum
  above the surface is an upward continuation through a replacement medium: give the replacement velocity.
- Afterwards gelev and selev equal the datum and sdepth is 0, so the following steps see a flat recording surface.

| Parameter | Meaning and choice |
|---|---|
| Datum elevation | The flat datum (m), the same one you would use for statics |
| Near-surface velocity | Velocity of the medium between the surface and the datum (weathering or replacement velocity) |
| Move | receivers and sources, receivers only, or sources only |
| Aperture | Half width of the Kirchhoff sum (m); at least the largest horizontal distance of the rays between surface and datum, typically 2-4 times the elevation difference; 200-500 m |
| Maximum angle | Steepest ray taken into the sum, 60-75 degrees |
| Highest frequency | The output is limited to this frequency (the operator costs more and aliases more at high frequencies) |

Sampling matters. The Kirchhoff sum needs the traces closer together than the width of the operator: about
sqrt(v |dz| / f) for an elevation difference dz.
- With receivers every 5-10 m and elevation differences of tens of metres, the receiver side is well sampled.
- Sparse shots (every 50-100 m) are too coarse for the source side. There, and at frequencies where the operator
  cannot be sampled, the step falls back to the vertical-ray shift |dz| / v, which is what elevation statics would
  have done.
- At every frequency the steep parts of the operator that would alias are left out automatically.

**Wave-equation datuming (zero offset, stacked)** does the same for a stacked section: from a floating datum, the
elevation in a header (gelev by default) at every CMP, to a flat datum, with half the velocity because stacked times
are two-way.

![A shot recorded on hilly topography (left), after elevation statics (middle) and after Kirchhoff wave-equation
datuming to the same datum (right). The statics leave the reflection jittery at the far offsets, where the rays are
far from vertical.](images/m40_datuming.png)

## MASW: surface-wave dispersion and an approximate Vs profile

Ground roll is noise for reflection processing, but its dispersion (phase velocity changing with frequency) carries
the shear-wave velocity of the near surface. The **MASW window** (View menu, MASW dispersion analysis) computes the
dispersion image of one shot by the phase-shift method (Park, Miller & Xia 1998) and lets you pick the
fundamental-mode curve.

Step by step:
1. Select a shot gather dataset with offsets (raw shots: no AGC, no f-k filter, no mute of the ground roll), then open
   the window.
2. Choose the **Shot (FFID)** and the side: positive or negative offsets (one side of the source at a time; the side
   with more traces is preselected) or both sides with |offset|.
3. Set the offset range: leave out the nearest traces (near-field effects, about the first 5-10 m) and the farthest
   ones where the surface waves fade into noise. Set the frequency and velocity ranges to cover the ground roll
   (typically 3-60 Hz and 50-1000 m/s), then press **Compute**.
4. The image shows, at every frequency, how strongly each phase velocity adds up (normalised to 1 at every frequency).
   The fundamental mode is the lowest-velocity strong ridge; higher modes, if present, lie above it.
5. Press **Automatic picks**. The ridge is followed from the frequency where it is sharpest, keeping the
   lowest-velocity peak and continuity between frequencies. Correct the picks by hand: a click adds a pick (replacing
   one at the same frequency); a right click or Shift + click removes the nearest pick.
6. **Export picks (CSV)** writes frequency, phase velocity, wavelength, approximate depth and approximate Vs.

The right-hand plot is an **approximate Vs profile** by the wavelength rule. A wavelength lambda = v/f senses the
ground to about lambda/3, where Vs is about 1.09 times the phase velocity. It is a quick first look, not an inversion:
for layer velocities and thicknesses, invert the exported curve with a dedicated program.

Velocities below frequency x trace spacing are blanked in the image: there it would only show spatial-aliasing copies
of the real ridge. Short spreads give broad ridges at low frequencies; the longest wavelength you can trust is about
the spread length, which limits the depth of investigation to about a third of it.

![The MASW window: dispersion image of a shot with the automatic fundamental-mode picks (left) and the approximate Vs
profile from them (right).](images/m41_masw.png)

### Rows for the cross-reference to Yilmaz

| Topic | SeisDesk |
|---|---|
| Refraction statics: reduced-time displays, intercept times | Viewer reduction velocity, Linear moveout (reduced time) |
| Refraction imaging: GRM time-depths, refraction convolution section | Refraction convolution section |
| Wave-equation datuming (before and after stack) | Wave-equation datuming (Kirchhoff, prestack; zero offset, stacked) |
| Linear moveout, ground-roll analysis, dispersion | Linear moveout (one velocity or dynamic), MASW window |

### Lines for the limits of version 0.3

- Wave-equation datuming uses one constant near-surface velocity, measures obliquity from the vertical (not from the
  local slope of the topography), and falls back to vertical-ray shifts where sources or receivers are too sparse for
  the Kirchhoff operator.
- The refraction convolution section pairs one forward and one reverse shot per receiver; it does not stack several
  pairs, and it takes a single reciprocal time per pair.
- The MASW window computes and picks dispersion curves and gives a wavelength-rule Vs estimate; it does not invert
  dispersion curves for a layered Vs model.
- The reduced-time display pauses picking; picks are made in true time.
