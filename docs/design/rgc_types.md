# RGC types the lab names — what the literature says

Checked 2026-09-25. Each PMID below was looked up in PubMed (esummary):
authors, year, title, journal and DOI match. Claims come from the paper's
abstract **[abs]** or full text **[full]**. Species is on every number:
[cat] [rabbit] [rat] [mouse].

Used by: the type atlas (PLAN.md Q44), suggested classes (Q36), the mosaic
check (Q42). Name matching: `src/analysis/class_names.py`.

## Where the names come from

- "Brisk sustained / brisk transient" starts in cat (Cleland & Levick 1974
  [abs]) and rabbit (Caldwell & Daw 1978 [abs]).
- Ravi et al. 2018 [full] defined the six types the lab uses, in **rat**,
  on a 512-electrode array with checkerboard noise: ON / OFF brisk
  sustained (bs), brisk transient (bt), small transient (st).
- Scalabrino et al. 2022 [full] used "ON brisk-sustained" for **mouse**.
- No paper we found links these array-defined types to genetically or
  morphologically identified mouse types. Every mouse match below is an
  inference. Bohlen et al. 2026 [full]: how to match RGC types between
  mouse and rat is still unclear.

## The lab's types

Signatures are from Ravi 2018 [rat, full] unless marked.

| Lab type (cells in lab .params) | White-noise signature | Likely mouse type (strength) |
|---|---|---|
| ON brisk sustained (656) | ON. STA time course weakly biphasic (least of the three pairs). Highest firing rate. RF larger than OFF bs. Most linear contrast response. [mouse] Easy to tell by its ACG (Scalabrino 2022). | Sustained ON alpha (weak–moderate). Ravi guessed rat bs = δ cells, not α. |
| OFF brisk sustained (635) | OFF. Weakly biphasic, but more than ON bs. Longer integration than ON bs. RF smaller than ON bs. [mouse] Tonic ~20 Hz (van Wyk 2009). | Sustained OFF alpha (weak–moderate). |
| OFF brisk transient (517) | OFF. Large RF (OFF bt > ON bt). Brief integration. Rectified. [mouse] tOFFα: no tonic firing, ~400 Hz bursts (van Wyk 2009) — expect a bursty ACG, low baseline. | Transient OFF alpha (moderate: cat bt = α). The tOFF "mini alpha" may be mixed in (Baden 2016). |
| ON brisk transient (472) | ON. Large RF, smaller than OFF bt. Brief integration, longer than OFF bt. | Transient ON alpha (weak). |
| OFF transient (795) | Not one of Ravi's six. No verified signature. | Unknown. Several OFF-transient types were once lumped (Goetz 2022). Check its mosaic: coverage above ~2–3 suggests a mixture (Baden 2016). |
| ON transient (26) | No verified signature. | Unknown. |
| OFF small transient (13) | Small RF, long time-to-zero, biphasic, most bursty (earliest ISI peak), most rectified. | Unknown in mouse. |

Metrics as Ravi 2018 defines them [rat]:
- Biphasic index = 1 − |a + b| / (|a| + |b|), with a and b the positive
  and negative lobe areas of the STA time course.
- Time-to-zero: the zero crossing of the time course nearest the spike.
- RF size: diameter of the circle with the area of the 1-SD fit contour.
- Across types, a larger RF goes with briefer integration.
- A rank-1 (space × time) fit held > 90 % of STA variance for all six types.

## Mosaics as a check of a type

- A real type tiles the retina: DeVries & Baylor 1997 [abs, rabbit];
  Ravi 2018 [full, rat]; Krieger 2017 [full, mouse]; Goetz 2022 [full].
- Ravi's test: nearest-neighbour distance normalised by RF size,
  NNND = 2R / (S1 + S2) (≈ 2 when RFs touch at 1 SD), compared with
  random resampling.
- Coverage above 1 is normal in mouse (~2–3; Baden 2016, Bae 2018).
  Density changes across the retina (Bleckert 2014), so compare locally.
- Arrays record only part of the cells (Ravi 2018; Bohlen 2026), so a gap
  does not disprove a type. **Two same-type cells too close together** is
  the stronger warning sign (a split or a mixed type).
- ON / OFF partner mosaics are anti-aligned (Roy 2021, rat and primate).

## How many types, and what white noise can separate

- Mouse: about 30–47 types (Sanes & Masland 2015 [abs]; Baden 2016;
  Bae 2018; Tran 2019 [abs]; Goetz 2022).
- On arrays the large-RF, large-spike types are the ones recovered well
  (Ravi 2018; Bohlen 2026).
- White noise + ACG separates polarity and the large, space-time-separable
  sustained / transient types.
- Needs other stimuli: direction- and orientation-selective cells (moving
  bars, gratings); types that differ in surround or spot-size tuning
  (spots of several sizes); temporal-frequency and contrast tuning
  (chirp). Goetz 2022 [full]: many types respond poorly or not at all to
  full-field stimuli or white noise.

## References (PubMed-verified)

| PMID | Reference | DOI |
|---|---|---|
| 4421622 | Cleland BG, Levick WR 1974. Brisk and sluggish concentrically organized ganglion cells in the cat's retina. J Physiol. | 10.1113/jphysiol.1974.sp010617 |
| 650447 | Caldwell JH, Daw NW 1978. New properties of rabbit retinal ganglion cells. J Physiol. | 10.1113/jphysiol.1978.sp012232 |
| 9325372 | Devries SH, Baylor DA 1997. Mosaic arrangement of ganglion cell receptive fields in rabbit retina. J Neurophysiol. | 10.1152/jn.1997.78.4.2048 |
| 30249795 | Ravi S et al. 2018. Pathway-Specific Asymmetries between ON and OFF Visual Signals. J Neurosci. | 10.1523/JNEUROSCI.2008-18.2018 |
| 33692544 | Roy S et al. 2021. Inter-mosaic coordination of retinal receptive fields. Nature. | 10.1038/s41586-021-03317-5 |
| 36040015 | Scalabrino ML et al. 2022. Robust cone-mediated signaling persists late into rod photoreceptor degeneration. eLife. | 10.7554/eLife.80271 |
| 41791371 | Bohlen MO et al. 2026. Projection targeting with phototagging to study the structure and function of retinal ganglion cells. Cell Rep Methods. | 10.1016/j.crmeth.2026.101308 |
| 19602302 | van Wyk M, Wässle H, Taylor WR 2009. Receptive field properties of ON- and OFF-ganglion cells in the mouse retina. Vis Neurosci. | 10.1017/S0952523809990137 |
| 28753612 | Krieger B et al. 2017. Four alpha ganglion cell types in mouse retina: Function, structure, and molecular signatures. PLoS One. | 10.1371/journal.pone.0180091 |
| 24440397 | Bleckert A et al. 2014. Visual space is represented by nonmatching topographies of distinct mouse retinal ganglion cell types. Curr Biol. | 10.1016/j.cub.2013.12.020 |
| 26735013 | Baden T et al. 2016. The functional diversity of retinal ganglion cells in the mouse. Nature. | 10.1038/nature16468 |
| 29775596 | Bae JA et al. 2018. Digital Museum of Retinal Ganglion Cells with Dense Anatomy and Physiology. Cell. | 10.1016/j.cell.2018.04.040 |
| 31784286 | Tran NM et al. 2019. Single-Cell Profiles of Retinal Ganglion Cells Differing in Resilience to Injury Reveal Neuroprotective Genes. Neuron. | 10.1016/j.neuron.2019.11.006 |
| 35830791 | Goetz J et al. 2022. Unified classification of mouse retinal ganglion cells using function, morphology, and gene expression. Cell Rep. | 10.1016/j.celrep.2022.111040 |
| 25897874 | Sanes JR, Masland RH 2015. The types of retinal ganglion cells: current status and implications for neuronal classification. Annu Rev Neurosci. | 10.1146/annurev-neuro-071714-034120 |

Not verified (do not cite from here): Ravi's rat mappings bs → δ,
bt → α, st → B1 rest on papers we did not look up; per-type mouse STA
time-to-peak values were not found.
