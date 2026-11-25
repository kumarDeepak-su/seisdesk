<!-- Group E (version 0.3): attributes, time-frequency analysis, eigenimage and Harlan filters, phase-weighted
stack, deghosting. Sections to merge into USER_GUIDE.md; suggested places are given in the comments. -->

<!-- New section, after "Time-depth conversion" in the Processing steps chapter. -->
## Seismic attributes and time-frequency analysis

Attributes turn a section (or a gather) into a display of one property of the waveform: reflection strength,
continuity, frequency content. All attribute steps are in the flow editor under **Attributes** and in
Processing > Apply one step. They work trace by trace, so they can be run on stacks, migrated sections or gathers.
Run them on the final section, with a band-pass first if the data are noisy: attributes of noise are noise.

**Hilbert transform (quadrature trace)** shifts every frequency component by -90 degrees: a cosine becomes a sine.
The input trace and its quadrature trace form the analytic (complex) trace, from which the instantaneous attributes
are computed. *-H{x}* gives the opposite shift (+90 degrees).

**Instantaneous attributes** (one attribute per run; choose it in **Attribute**):

| Attribute | What it shows | Typical use |
|---|---|---|
| Envelope | Amplitude of the analytic trace, independent of phase (reflection strength) | Bright spots, strong contrasts, unconformities |
| Instantaneous phase | Phase in degrees (-180 to 180), independent of amplitude | Continuity of weak events, pinch-outs, onlap; display with a cyclic colour map |
| Cosine of phase | Cosine of the instantaneous phase (-1 to 1) | Like the phase, but displays well in grey |
| Instantaneous frequency | Rate of change of the phase (Hz), averaged over the **smoothing window** with the squared envelope as weight | Low-frequency shadows below gas, thin-bed tuning, attenuation |
| Instantaneous bandwidth | Rate of change of the envelope relative to the envelope (Hz) | Sharpness of reflections |
| Envelope derivative, second derivative | Rise and curvature of the envelope | Sharp onsets of reflections |
| Sweetness | Envelope divided by the square root of the instantaneous frequency | Bright, low-frequency sand bodies |
| Apparent polarity | Sign of the trace at each envelope peak times the peak, over the whole envelope lobe | Polarity of strong reflections (positive or negative contrast) |

| Parameter | Meaning and choice |
|---|---|
| Attribute | One of the attributes above |
| Smoothing window | 0.01-0.04 s; used by frequency, bandwidth and sweetness. Longer = smoother, less detail |

**Spectral decomposition (single frequency)** gives the amplitude (or the phase) of every trace at one frequency, as a
function of time. A unit sinusoid at that frequency has amplitude 1, so the outputs at different frequencies can be
compared. Run it at several frequencies (for example 15, 30 and 45 Hz) to see where the spectrum changes:
thin beds tune at a frequency set by their thickness, and gas sands often lose the high frequencies below them.

| Method | Time window | Use |
|---|---|---|
| S-transform | One period of the frequency (short at high frequencies) | The default: sharp in time at high frequencies, sharp in frequency at low ones |
| STFT (Gabor) | One fixed Gaussian window for all frequencies (**window** = ±2 standard deviations) | Equal resolution at all frequencies; 0.1-0.2 s |
| CWT (Morlet) | Like the S-transform; **w0** sets the trade-off | w0 = 6 is close to the S-transform; larger w0 = sharper in frequency, smoother in time |

| Parameter | Meaning and choice |
|---|---|
| Frequency | The frequency to extract (below the Nyquist frequency) |
| Method | S-transform, STFT (Gabor) or CWT (Morlet) |
| STFT window length | Gabor transform only: 0.1-0.2 s |
| Morlet w0 | CWT only: 5-8 |
| Output | *amplitude*, or *phase (degrees)* of that frequency component |

![Instantaneous attributes of the marine stack: envelope (left), instantaneous phase (middle) and instantaneous
frequency (right).](images/m54_instantaneous_attributes.png)

**Time-frequency analysis of one trace** (View > Time-frequency analysis of a trace):
1. Display the dataset in the viewer (a stack, a gather or a single shot).
2. Open the window. It shows the amplitude of the selected trace as an image, with time down and frequency across,
   and the trace itself on the left (same time axis).
3. Choose the **trace** (number within the displayed traces), the **method** (S-transform, STFT (Gabor) or
   CWT (Morlet)), the **STFT window** or **Morlet w0**, and the **highest frequency**. The image updates at once.
4. Read the band of each event: where the colours are strong at a given time is the frequency content there. The
   line above the image gives the time and frequency of the strongest amplitude.

Use it to see how the spectrum changes with time (absorption, the high-cut filter for a time-variant band-pass),
to find the tuning frequency of a thin bed, or to check a filter or a deconvolution trace by trace.

![The time-frequency window: S-transform of one trace of the stack, with the trace on the left.](images/m55_time_frequency.png)

<!-- Filtering section: after "Trace mixing" (or the tau-p filter), before the noise table. -->
**Eigenimage (SVD) filter** splits windows of traces into eigenimages by singular value decomposition. An event that
is flat across a window (the same waveform on every trace) goes into the first eigenimage; random noise and dipping
events spread over all of them.
- *pass* keeps the first **k** eigenimages: a strong coherence filter for flat events, e.g. NMO-corrected CMP
  gathers (k = 1-3) or flattened horizons.
- *reject* removes them: for example, on CMP gathers corrected with the multiple velocity the multiples become flat
  and *reject* removes them while keeping the (curved) primaries.
- Events that are not flat can be flattened first with a **linear moveout velocity** (|offset| / v, for linear noise
  or refractions in shot gathers) or a **dip** in ms per trace (for a dipping horizon in a section); the flattening is
  undone after filtering.
- On sections with dipping or curved events use small **trace windows** (7-15 traces) and **time windows**
  (0.2-0.5 s), so the events are close to flat inside each window.

| Parameter | Meaning and choice |
|---|---|
| Number of eigenimages (k) | 1-3; more keeps more detail and more noise |
| Mode | *pass* (keep flat events) or *reject* (remove them) |
| Gather key | Empty for a section, `cdp` for CMP gathers, `fldr` for shots |
| Trace window, time window | 0 = whole gather and whole trace; otherwise overlapping windows blended with tapers |
| Flatten by linear moveout, by a dip | 0 = off |

**Signal/noise separation (Harlan, local slant stacks)** separates laterally coherent signal from random noise
(after Harlan, Claerbout and Rocca, 1984):
- In overlapping windows of traces the data are slant-stacked (local tau-p, dips up to the **largest dip**). Linear
  events focus into a few strong values, random noise spreads evenly over the tau-p panel.
- The noise level is estimated from the median absolute value of the panel. Values below **threshold** × noise level
  are set to zero (*hard*) or all values are shrunk towards zero by that amount (*soft*), and the panel is
  transformed back.
- *signal* outputs the coherent part; *noise* outputs what was removed. Always look at the noise output: it should
  contain no reflections.

| Parameter | Meaning and choice |
|---|---|
| Output | *signal* (the filtered data) or *noise* (data minus signal, for QC) |
| Gather key | Empty for a section; a key to work gather by gather |
| Trace window, time window | 12-24 traces, 0.2-0.5 s: events should be nearly straight inside a window |
| Largest dip | ms per trace; steeper events are treated as noise |
| Threshold | 1.5-3 × noise level; higher removes more noise and weakens weak events |
| Shrinkage | *soft* (smooth result, slightly weaker amplitudes) or *hard* (keeps amplitudes, more residual noise) |
| Highest frequency | The signal is band-limited to this frequency |

**Deghosting (source/receiver ghost)** removes the sea-surface ghost of marine data. A source or receiver at depth z
records every wave twice: directly, and 2 z cos(θ) / v later reflected from the sea surface with opposite
polarity. The ghost makes notches in the spectrum at 0 Hz and at multiples of v / (2 z cos θ) (for example 94 Hz for
a streamer at 8 m) and weakens the low frequencies.
- **Ghost to remove:** *receiver*, *source* or *both*.
- **Depths** come from the headers: `sdepth` for the source and minus `gelev` for the receivers (negative elevation
  below the sea surface). Type them in when the headers are empty.
- *vertical incidence* treats every trace as if the waves arrived vertically (cos θ = 1): good for near offsets,
  stacks and deep targets.
- *angle-dependent (tau-p)* works on shot gathers in the linear tau-p domain, where every slowness p has its own ghost
  delay (cos θ = sqrt(1 - p² v²)); better for long offsets and shallow targets. Source and receiver are given the
  same slowness, which is exact for flat layers.
- The **water level** (1-5 %) limits the gain in the ghost notches; raise it if the output becomes noisy or rings.

| Parameter | Meaning and choice |
|---|---|
| Ghost to remove | *receiver*, *source* or *both* |
| Mode | *vertical incidence* or *angle-dependent (tau-p)* |
| Source depth, receiver depth | 0 = from the headers |
| Water velocity | 1480-1540 m/s |
| Sea-surface reflection coefficient | 0.9-1.0 (a rough sea reflects less) |
| Water level | 1-5 % |
| Shot gather key, number of p values, lowest and highest frequency | tau-p mode only; 0 = Nyquist for the highest frequency |

![Random noise on the marine stack (left), the signal from the Harlan separation (middle) and the eigenimage filter
(rank 2, 9-trace windows, right).](images/m56_harlan_eigenimage.png)

![A shot without ghosts (left), the same shot with a source ghost at 10 m and a receiver ghost at 15 m (middle),
and after deghosting at vertical incidence (right).](images/m57_deghosting.png)

<!-- Rows for the "Noise attenuation, which tool for which noise" table. -->
| Noise | First choice | Alternatives |
|---|---|---|
| Random noise on sections with gentle dips | f-x deconvolution | Harlan separation, eigenimage filter (small windows) |
| Random noise on NMO-corrected CMP gathers | Eigenimage filter, *pass*, k = 1-3 | f-x deconvolution |
| Sea-surface ghost (marine) | Deghosting | Spectral whitening (partial) |

<!-- Stacking section: add to the list of methods. -->
  - *phase-weighted* (Schimmel and Paulssen, 1997): the live stack multiplied by the coherence of the instantaneous
    phases of the traces, raised to the **phase-weighted stack power** (2 is usual; 0 gives the live stack). Where
    the traces agree the coherence is 1 and the stack is kept; where they do not (random noise) it is close to 0.
    Very strong noise suppression for low-fold or noisy data, at the price of non-linear amplitudes. The coherence
    can be smoothed over a short **window** (0.01-0.05 s).

<!-- Rows for the "Cross-reference to Yilmaz" table. -->
| Topic | SeisDesk |
|---|---|
| Complex trace attributes (Hilbert transform, envelope, instantaneous phase and frequency) | Hilbert transform, instantaneous attributes |
| Time-frequency analysis, spectral decomposition | Time-frequency analysis window; spectral decomposition (S-transform, Gabor, Morlet) |
| Eigenimage (Karhunen-Loève) filtering | Eigenimage (SVD) filter |
| Signal and noise separation by slant stacks | Harlan signal/noise separation |
| Marine ghosts | Deghosting (source and receiver, vertical incidence or tau-p) |
| Coherence-weighted stacking | Phase-weighted stack |

<!-- Lines for the "Limits of version 0.3" chapter. -->
- Deghosting assumes a flat sea surface and one source and one receiver depth per shot gather (the median of the
  gather) in the tau-p mode; the tau-p mode uses the offsets as the slant-stack axis and the same slowness for source
  and receiver.
- The Harlan separation shrinks the local tau-p panel with a soft or hard threshold; it does not estimate the full
  signal and noise distributions of the original method. Inside a window the traces are assumed to be evenly
  spaced, and the signal is limited to the highest frequency given.
- The eigenimage filter flattens with one linear moveout velocity or one dip per run.
- Spectral decomposition extracts one frequency per run; the time-frequency window shows one trace at a time. On
  depth sections the frequency attributes are in cycles per metre.
