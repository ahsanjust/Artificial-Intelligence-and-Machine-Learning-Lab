# AI & ML Lab — Comprehensive Report

This folder contains the final LaTeX report covering all six models from
Lab 2 through Lab 4.

## Files

| File          | Description                                              |
|---------------|----------------------------------------------------------|
| `Report.tex`  | The complete LaTeX report (Overleaf-ready).              |
| `report.md`   | The original brief from the instructor.                  |

## How to Compile on Overleaf

1. **Create a new project** on [Overleaf](https://www.overleaf.com).
2. **Upload the file `Report.tex`** to the project root.
3. **Upload the entire repository** as a zip, or upload the screenshots
   folders individually, so that the relative image paths resolve.

   The cleanest way is to keep the directory layout
   ```
   .
   ├── Report.tex
   ├── Lab_2/
   │   ├── Linear_Regression/screenshots/...
   │   └── Logistic_Regression/screenshots/...
   ├── Lab_3/
   │   ├── SVM/screenshots/...
   │   └── KNN/screenshots/...
   └── Lab_4/
       ├── DT/screenshots/...
       └── KMeans/screenshots/...
   ```
   and set Overleaf's **main document** to `Lab_Report/Report.tex`.
4. Set the compiler to **pdfLaTeX** (the default).
5. Click **Recompile**. The report should compile without errors.

## Local Compilation (if you have TeX Live)

```bash
cd Lab_Report
pdflatex -interaction=nonstopmode Report.tex
pdflatex -interaction=nonstopmode Report.tex   # second pass for the TOC
```

## Notes

* The report embeds screenshots via `\includegraphics` with relative
  paths starting `../Lab_*` — keep that structure when you upload to
  Overleaf.
* All equations, tables, figures, captions, the table of contents and
  the reference list are included.
* No PDF file is committed — only the `.tex` source, per the lab
  instructions.
