# Case Study — MedicalTreatment-Analytics

**Repository:** [MedicalTreatment-Analytics](https://github.com/Freddricklogan/MedicalTreatment-Analytics) · **Live report:** [freddricklogan.github.io/MedicalTreatment-Analytics](https://freddricklogan.github.io/MedicalTreatment-Analytics/) · **Author:** Freddrick Logan

---

## 1. Who has this problem

Anyone who inherits an analysis with its conclusions already written: a research assistant asked to reproduce a figure, an instructor grading a statistics exercise, a reviewer deciding whether the author checks claims against data. The mouse study is a teaching dataset; the habit it tests — computing before asserting — is not.

## 2. The problem, as a scenario

A reviewer reads the README: Capomulin and Ramicane most effective, a strong 0.84 correlation between weight and tumour volume, no deaths on the two best regimens. She runs the script. The correlation comes out at −0.02. The "deaths" were never computed. The three t-tests in the committed CSV have no correction for multiple comparisons. The PNGs in the repository were made from some earlier version of the data; the CSV now committed is a variant with fewer rows. That was the earlier version of this repository: conclusions from the textbook, data from somewhere else, and code that connected neither.

## 3. What it costs to leave it alone

A number copied from a course answer key and presented as a finding is the most damaging kind of error in analytical work, because it is exactly right for a dataset the reader cannot see. Uncorrected pairwise tests inflate false positives; on this data they declared a difference between two regimens that a proper test does not support. Committed output images drift from the code silently. None of this matters for a grade; all of it matters for whether the author can be trusted with a real study.

## 4. The approach, and the alternative I rejected

I rejected fixing the README's numbers by hand. The point is that the numbers should not be typed at all. The script became a package. `data.py` merges the two files, validates them, removes mice with duplicated timepoints and returns their ids, and selects each mouse's last timepoint. `stats.py` computes the per-regimen summary, quartiles and 1.5 × IQR outliers, the weight–volume regression, one-way ANOVA with Tukey HSD across the four regimens of interest, and Kaplan–Meier time-in-study curves with a log-rank test against placebo, using scipy and statsmodels. `report.py` renders a page and JSON from those results, and CI runs it and publishes the output, so the page — and this document — report what the committed files contain.

## 5. What the code does today

`pymaceuticals report --out dist` writes a report with the Executive Shell. It states the data and the cleaning rule, with the dropped mouse named. It tabulates mean, median, variance, standard deviation and SEM of tumour volume per regimen. For Capomulin, Ramicane, Infubinol and Ceftamin it shows each mouse's final volume as boxes and whiskers with the IQR bounds printed and any outlier named with its value. It reports the ANOVA F and p, then the Tukey table with adjusted p-values and a yes/no at 5 %. It reports the weight–volume regression for Capomulin — r, r², p, slope — and says on the page that the earlier README's 0.84 was not computed from these files. It draws Kaplan–Meier curves per regimen with the count of mice leaving before day 45 and a log-rank p against placebo. Raw numbers go to `report.json`.

## 6. Evidence

Nine tests at 100 % statement coverage cover the merge and cleaning (249 mice, 1,709 observations, one row per mouse in the final table), three validation rejections, summary statistics against pandas with the SEM formula, quartiles against NumPy with the single Ceftamin outlier at 71.82, a regression that recovers a known line exactly and reports no relationship on these files, ANOVA structure with the expected Tukey verdicts, the Kaplan–Meier estimate for Capomulin against the exact hand value 20/24, the log-rank test, and the report writer. The run reports ANOVA F = 78.5 with p < 0.001; Capomulin versus Ramicane adjusted p = 0.312; Ceftamin versus Infubinol 0.059; regression r = −0.022 with p = 0.920 over 24 mice; events before day 45 of 4, 5, 16 and 16. The report rendered with zero console errors and no horizontal scroll at 1280 or 400 pixels. `AUDIT.md` records nine findings.

## 7. What it would take to run this in production

For a real preclinical study the same package would need the protocol's definitions — what counts as an event, how censoring is recorded, which comparisons were pre-specified — encoded as configuration rather than defaults, a mixed-effects model for repeated measures over time rather than final-volume ANOVA alone, and a data-provenance record tied to the laboratory system. The structure would hold; the statistics would grow.

## 8. Limits and next steps

The committed files are a variant of the exercise dataset, so results differ from the widely published ones and the report says so. Final-volume ANOVA ignores the time course; time in study is a proxy for survival. Next, in order: a longitudinal mixed model of volume by regimen and day, a per-mouse trajectory panel, and configuration for the event definition.

## 9. Who should look at this

**Hiring manager:** evidence that I check inherited claims against data, apply the right tests with corrections, and let the pipeline write the numbers.
**Consulting client:** a small, complete example of a reproducible statistical report with its cleaning rules stated.
**Engineer:** read `src/pymaceuticals/stats.py` with `tests/test_stats.py`, especially the survival and Tukey sections.
