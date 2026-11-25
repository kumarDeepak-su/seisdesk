## Wavelet phase: rotation, zero phase and minimum phase

The phase of the wavelet decides where a reflection's energy sits in time. A **zero-phase** wavelet is a symmetric
peak centred on the reflection time, which is what an interpreter wants. A **minimum-phase** wavelet has its energy
at the front; dynamite, air guns and most recording filters give one, and predictive deconvolution assumes one. The
**Wavelet** category of the flow editor has three tools for this (also under Processing > Apply one step).

**Phase rotation (constant phase)** rotates every frequency by the same angle: out = in cos(a) - H(in) sin(a), with
H the Hilbert transform.
- 180 degrees reverses the polarity.
- 90 or -90 degrees turns a zero-phase wavelet into an antisymmetric one (a zero crossing at the reflection time) and
  back.
- Use it to match the phase of two vintages, or of the stack and a well synthetic: try angles in steps of 15-30
  degrees and keep the one whose wavelet best matches.

**Zero-phase conversion** and **Minimum-phase conversion** keep the amplitude spectrum and replace the phase. Both
have three modes:

| Mode | What it does | Use it for |
|---|---|---|
| data wavelet | Estimates the wavelet's amplitude spectrum from the average autocorrelation of the traces in the design window. Zero-phase conversion takes the data wavelet as minimum phase; minimum-phase conversion takes it as zero phase. The filter is all-pass: amplitudes do not change, only the phase. | Zero phase: after spiking or predictive deconvolution (minimum-phase output), before interpretation. Minimum phase: correlated vibroseis or zero-phase processed data, before predictive deconvolution. |
| wavelet file | Uses a known wavelet from a text file and converts it, whatever its phase, to its zero- or minimum-phase equivalent (stabilised by the water level). | A measured air-gun or vibrator signature, a wavelet extracted at a well. |
| per trace | Treats every trace as a wavelet: the zero-phase result is centred on the trace's envelope peak, the minimum-phase result starts where the envelope first reaches 10 % of its peak. | Wavelet or signature traces only. On reflection data it destroys the timing of the events. |

| Parameter | Meaning and choice |
|---|---|
| Design window start / end | A window with good reflections and no first breaks, e.g. 0.2-2.0 s (data wavelet mode) |
| Spectral smoothing | Smooths the estimated amplitude spectrum, 3-10 Hz; more smoothing for short windows or few traces |
| Wavelet file | Text file: one amplitude per line (first sample at time zero, sampled at the data interval or at the given file interval), or two columns time and amplitude (seconds, or milliseconds when the times exceed 5) |
| Stabilisation (water level) | Percent of the peak power added before dividing by a wavelet spectrum; 0.1 % for a clean measured wavelet, 1 % or more for a noisy one |

![A marine shot with its zero-phase wavelet (left), rotated by 90 degrees (middle) and converted to minimum phase
(right).](images/m34_phase_tools.png)

**Pitfalls**
- Zero-phase conversion in the data wavelet mode is only right if the data wavelet really is minimum phase; run it
  after deconvolution, not on raw vibroseis data (those are already zero phase).
- Check the result on a well tie or on a strong isolated reflector: a correct zero-phase wavelet is symmetric.

## Wiener shaping filter, signature deconvolution and homomorphic deconvolution

**Wiener shaping filter** designs the least-squares filter that turns an input wavelet into a desired output wavelet
and applies it to every trace.
1. Choose the **input wavelet**: *estimated from the data* (minimum phase, from the average autocorrelation in the
   design window) or a *wavelet file*.
2. Choose the **desired output**: a zero-phase Ricker (peak frequency), a zero-phase Ormsby band-pass wavelet (four
   corners) or a file.
3. Set the operator length and the output delay. The delay (default: half the operator) lets the filter reach before
   the onset of the input wavelet, which a zero-phase output needs; it is removed again afterwards, so a zero-phase
   output is centred on the onset time of the input wavelet.

The log reports the relative error of the shaped wavelet; above about 10 % make the operator longer, the desired
wavelet broader or its band narrower.

| Parameter | Meaning and choice |
|---|---|
| Operator length | 0.1-0.3 s; at least the length of the input wavelet |
| Pre-whitening | 0.1-1 %; raise it if the output rings |
| Ricker peak frequency / band-pass corners | Inside the band of the data (look at the amplitude spectrum, F8); shaping cannot create frequencies the data do not have |
| Output delay | -1 = half the operator length |

**Signature (deterministic) deconvolution** removes a *known* source wavelet: each trace is divided in the frequency
domain by the wavelet spectrum (stabilised by the water level) and shaped to a spike or a zero-phase band-pass
wavelet.
- **Wavelet file:** the time origin in the file is honoured, so a two-column file with negative times (a zero-phase
  wavelet) works as well as a causal one.
- **Trace in the data:** the wavelet is a trace selected by a header value, for example a near-field hydrophone with
  trid = 9 or an auxiliary channel; with a gather key (fldr) every shot uses its own signature trace. The signature
  must start at the shot time (time zero of the record).
- Use it before statistical deconvolution: what is left afterwards is the earth's response.

**Homomorphic (cepstral) deconvolution** estimates a mixed-phase wavelet without assuming minimum phase (Ulrych
1971). The complex cepstrum (log amplitude and unwrapped phase) of all traces is averaged; the reflectivity differs
from trace to trace and averages out, the common wavelet remains at low quefrencies. The cepstrum is cut at the
**cepstral cut-off** (about the wavelet length, 0.02-0.05 s), turned back into a wavelet, and every trace is
deconvolved with it. The estimated wavelet is kept in the results. The method needs many traces (50 or more) with good
signal-to-noise ratio; on noisy data prefer spiking deconvolution followed by zero-phase conversion.

![A land shot (left), after signature deconvolution with the known source wavelet to a 5-80 Hz zero-phase wavelet
(middle) and after a shaping filter to a 30 Hz Ricker wavelet (right).](images/m35_shaping_signature.png)

## Vibroseis data

A vibrator sends a sweep several seconds long, so every reflection in an **uncorrelated** record is a whole sweep.
Correlation with the sweep compresses each one into a short zero-phase **Klauder wavelet** (the autocorrelation of the
sweep). The **Vibroseis** category has three steps.

**Vibroseis correlation**
1. Choose where the sweep comes from:
   - **parametric:** start and end frequency, length, type (linear, or a boost in dB per octave, which spends longer
     at high frequencies to make up for their absorption) and the cosine tapers at both ends, as set on the vibrator
     controller;
   - **pilot trace:** a recorded sweep in the data, selected by a header value (SEG-Y trid 6 = sweep), one per
     gather of the gather key; the pilot traces are removed from the output unless you untick the option;
   - **sweep file:** a text file as for wavelets.
2. **Output length:** the listen time; 0 keeps the record length minus the sweep length.
3. Run it as the first step, before geometry and editing. The output is scaled by the sweep energy, so a reflection
   keeps its amplitude.

**Klauder wavelet to minimum phase** applies the all-pass filter W_min / K, which turns the zero-phase Klauder
wavelet K into its minimum-phase equivalent W_min (Gibson & Larner 1984). Run it after correlation, with the same
sweep, and before predictive deconvolution; then deconvolution sees the minimum-phase wavelet it assumes.

**Convolve with a sweep (uncorrelated records)** does the opposite of correlation: it turns reflectivity or correlated
data into uncorrelated records. Use it to test sweep parameters on synthetic data or for teaching.

| Parameter | Meaning and choice |
|---|---|
| Start / end frequency | As recorded, e.g. 8-80 Hz (an up-sweep); a down-sweep has a start above the end |
| Sweep length | As recorded, e.g. 8-16 s |
| Sweep type, boost | Linear, or dB per octave (3-6 dB per octave are common) |
| Start / end taper | As on the controller, 0.2-0.5 s |
| Pilot trace header and value | trid and 6 for a recorded sweep trace |

![An uncorrelated vibroseis record (left), after correlation with the sweep (middle, zero-phase Klauder wavelet)
and after conversion of the Klauder wavelet to minimum phase (right).](images/m36_vibroseis.png)

**Pitfalls**
- A wrong sweep (length, direction, taper) leaves smeared correlation noise; check the sweep parameters on one record
  and compare the correlated first breaks with those of a nearby dynamite record.
- Correlation must come before any filtering or gain that changes the uncorrelated record in time.

## Trace interpolation

**Trace interpolation (f-x, Spitz)** inserts a new trace midway between every pair of neighbouring traces of each
gather (factor 2; factor 4 repeats it). It predicts across the traces with f-x prediction filters: the filter of the
coarse traces at half the frequency is the filter of the refined gather at the full frequency (Spitz 1991). Steep,
spatially aliased events — ground roll, steep dips, refractions — therefore stay continuous, which simple averaging
of neighbours cannot do.
1. Pick the gather key (fldr for shot gathers, empty for a stacked section) and the order key (offset for shots,
   cdp or cdpx for a stack). The traces must be regularly spaced along the order key; the log warns if they are not.
2. Choose the time and trace windows: the events must be roughly straight inside a window (0.5 s and 24 traces suit
   most shot gathers).
3. New traces get interpolated positions (offset, source and receiver x, CMP x), header interp = 1 and renumbered
   tracf. Run the geometry step again afterwards, so the CMP numbers follow the finer spacing.

**Fill missing or dead traces (POCS)** puts each gather on a regular grid of the position key (spacing: the median
trace spacing, or the value you give), adds empty traces where positions are missing — for example skipped stations
or a gap at the near offsets — and, if asked, treats dead (all-zero) traces as missing. It rebuilds them by projection
onto convex sets: the f-k spectrum is thresholded with a threshold that falls from iteration to iteration, and the
recorded traces are put back after every iteration (Abma & Kabir 2006). Recorded traces do not change; filled traces
get interp = 1.

| Parameter | Meaning and choice |
|---|---|
| Gather key / order or position key | fldr and offset for shot gathers; empty and cdpx for a stack |
| Prediction filter length (Spitz) | 3-6 traces; at least the number of dips crossing a window |
| Damping | 1 %; more for noisy data |
| Lowest / highest frequency | The signal band; outside it the new traces are averages of their neighbours |
| Grid spacing (POCS) | 0 = median spacing of the gather |
| Iterations, final threshold (POCS) | 50-100 iterations; final threshold 0.1-1 % of the largest coefficient |

![A marine shot with every second trace (left, steep events aliased), after f-x interpolation back to the full trace
spacing (second), the shot with missing traces (third) and the gaps filled by POCS
(right).](images/m37_interpolation.png)

**Pitfalls**
- Interpolated traces contain no new information: they help migration, multiple attenuation and f-k filters that
  suffer from spatial aliasing, but they do not improve the signal-to-noise ratio.
- POCS fills scattered gaps of a few traces well; it cannot rebuild a gap much wider than the wavelength of the
  events across the traces.

---

Rows for the "Cross-reference to Yilmaz" table:

| Topic | SeisDesk |
|---|---|
| Wavelet processing: phase rotation, zero- and minimum-phase conversion | Phase rotation, zero-phase conversion, minimum-phase conversion |
| Wiener shaping filters, deterministic (signature) deconvolution | Wiener shaping filter, signature deconvolution |
| Homomorphic deconvolution, complex cepstrum | Homomorphic (cepstral) deconvolution |
| Vibroseis correlation, Klauder wavelet, minimum-phase conversion | Vibroseis correlation, Klauder wavelet to minimum phase, convolve with a sweep |
| Trace interpolation, spatial aliasing, data regularisation | Trace interpolation (f-x, Spitz), fill missing or dead traces (POCS) |

Lines for "Limits of version 0.3":

- Zero- and minimum-phase conversion of the data wavelet assume the current phase (minimum or zero); a mixed-phase
  data wavelet needs a known wavelet file or homomorphic deconvolution.
- Homomorphic deconvolution is sensitive to noise and phase unwrapping; it needs many traces and a short wavelet.
- Vibroseis: no vibrator ground-force or harmonic-distortion handling, no slip-sweep separation; the correlation uses
  one sweep (or pilot) per gather.
- f-x interpolation assumes regularly spaced traces and locally straight events; POCS works on one 2D gather at a time
  (no 3D or 5D regularisation) and fills only gaps a few traces wide reliably.
