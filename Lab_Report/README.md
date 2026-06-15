# AI & ML Lab — Comprehensive Report

This folder contains the final LaTeX report covering all six models from
Lab 2 through Lab 4.

## Files

```
Lab_Report/
├── Report.tex          # Main LaTeX report (Overleaf-ready)
├── figures/            # All 21 referenced images, safe filenames
│   ├── training_loss_curve_1.png
│   ├── ...
│   └── custom_cluster_distribution_1.png
├── README.md           # This file
└── report.md           # Original brief from the instructor
```

The `figures/` directory is local to `Report.tex` — every
`\includegraphics{figures/...}` path is relative to the `Report.tex`
location, so no `Lab_2/` / `Lab_3/` / `Lab_4/` folders need to be
uploaded alongside the `.tex` to Overleaf.

## How to Compile on Overleaf

1. **Create a new project** on [Overleaf](https://www.overleaf.com).
2. **Upload `Report.tex` and the `figures/` folder** as a single zip,
   then drag-and-drop the zip into the project.
3. The project root should look like:
   ```
   .
   ├── Report.tex
   └── figures/
       ├── *.png  (21 files)
   ```
4. Set the compiler to **pdfLaTeX** (the default).
5. Click **Recompile**. Every figure should render in place.

## Local Compilation (if you have TeX Live / Tectonic)

```bash
cd Lab_Report
pdflatex -interaction=nonstopmode Report.tex
pdflatex -interaction=nonstopmode Report.tex   # second pass for the TOC
```

## Notes

* The report embeds screenshots via `\includegraphics{figures/...}` —
  this directory layout means the `Report.tex` is fully self-contained.
* All 21 image filenames are lowercase with underscores (no spaces),
  which keeps the `\includegraphics` calls portable and shell-safe.
* No PDF file is committed — only the `.tex` source, per the lab
  instructions.
