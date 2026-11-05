#!/bin/zsh
# Build the Word version of the BE report from the LaTeX source (docs/report/BE_Project_Report).
#   scripts/build_report_docx.sh            -> docs/report/RetinAI_BE_Project_Report.docx
# Needs pandoc and python-docx. Works on a simplified copy: tabularx -> tabular, PDF figures -> PNG,
# the template's layout macros reduced to plain headings, and every [FILL-nn] marker highlighted.
set -e
ROOT=${0:A:h:h}
SRC=$ROOT/docs/report/BE_Project_Report
S=$(mktemp -d)
cp -R $SRC/chapters $SRC/backmatter $SRC/frontmatter $SRC/tables $SRC/project_details.tex $SRC/fillmacros.tex $S/
mkdir -p $S/figures && cp $ROOT/docs/report/figures/*.png $S/figures/
cd $S
sed -i '' 's/\.pdf}/.png}/g' chapters/*.tex backmatter/*.tex
sed -i '' 's/\\thispagestyle{empty}//g; s/\\clearpage//g; s/\\ (/~(/g' chapters/*.tex frontmatter/*.tex backmatter/*.tex
sed -i '' 's/\\ifdedication\\else\\thispagestyle{empty}\\fi//; s/\\ifcoguide\\Blank\[55mm\]\\\\\\CoGuideName\\\\Co-supervisor or Co-guide\\fi//' frontmatter/certificate.tex
sed -i '' 's/\\texttt{\\string\\input\\{backmatter\/checklist\\}}/\\texttt{input backmatter\/checklist}/; s/\\texttt{\\string\\finalcopytrue}/\\texttt{finalcopytrue}/; s/\\texttt{\\string\\coguidefalse}/\\texttt{coguidefalse}/' backmatter/checklist.tex
# the P5 macros from fillmacros.tex, without its packages and \FILL definition
grep '^\\newcommand{\\PFIVE' fillmacros.tex > pfive.tex
$ROOT/.venv/bin/python - <<'EOF'
import re, glob
for f in glob.glob("chapters/*.tex") + glob.glob("backmatter/*.tex") + glob.glob("frontmatter/*.tex"):
    s = open(f).read()
    n = lambda m: "l" * len(re.findall(r"Y|p\{[^}]*\}|l|c|r", m.group(1)))
    s2 = re.sub(r"\\begin\{tabularx\}\{\\textwidth\}\{@\{\}(.*?)@\{\}\}", lambda m: "\\begin{tabular}{" + n(m) + "}", s)
    s2 = s2.replace("\\end{tabularx}", "\\end{tabular}")
    s2 = re.sub(r"\\begin\{longtable\}\{@\{\}(.*?)@\{\}\}", lambda m: "\\begin{longtable}{" + n(m) + "}", s2)
    s2 = s2.replace("\\multicolumn{5}{c|}", "\\multicolumn{5}{c}")
    open(f, "w").write(s2)
EOF
cat > main_docx.tex <<'EOF'
\documentclass{report}
\usepackage{graphicx}\graphicspath{{figures/}}
\newcommand{\FrontHeading}[1]{\chapter*{#1}}
\newcommand{\BackHeading}[1]{\chapter*{#1}}
\newcommand{\Prompt}[1]{}
\newcommand{\Blank}[1][35mm]{\textunderscore\textunderscore\textunderscore\textunderscore\textunderscore\textunderscore\textunderscore\textunderscore\textunderscore\textunderscore\textunderscore\textunderscore}
\newcommand{\InstituteLogo}{[Institute emblem, 50 mm]}
\newcommand{\TightTable}{}
\newcommand{\FILL}[2]{\textbf{[FILL-#1: #2]}}
\input{pfive}
\input{tables/deployed_macros}
\input{project_details}
\newcommand{\StudentTitleRows}{\StudentOne~(\RollOne); \StudentTwo~(\RollTwo); \StudentThree~(\RollThree); \StudentFour~(\RollFour)}
\newcommand{\StudentSignatures}{\StudentOne~(\RollOne) Signature: \Blank \par \StudentTwo~(\RollTwo) Signature: \Blank \par \StudentThree~(\RollThree) Signature: \Blank \par \StudentFour~(\RollFour) Signature: \Blank}
\begin{document}
\chapter*{\ProjectTitle}
A Project Report submitted in partial fulfillment of the requirements of the degree of \DegreeName\ in \ProgramName\ by \StudentTitleRows. Supervisor: \GuideName. \InstituteLogo. \DepartmentName, \InstituteName, \InstitutePlace, \UniversityName, \SubmissionYear.
\input{frontmatter/certificate}
\input{frontmatter/declaration}
\input{frontmatter/abstract}
\input{frontmatter/nomenclature}
\input{chapters/01_introduction}
\input{chapters/02_literature}
\input{chapters/03_requirements}
\input{chapters/04_design}
\input{chapters/05_implementation}
\input{chapters/06_testing}
\input{chapters/07_results}
\input{chapters/08_conclusions}
\appendix
\input{backmatter/appendices}
\input{backmatter/references}
\input{backmatter/publications}
\input{backmatter/acknowledgements}
\input{backmatter/checklist}
\end{document}
EOF
pandoc main_docx.tex -o report.docx --number-sections --toc --toc-depth=2 --resource-path=.:figures 2>&1 | grep -v "Could not convert TeX math" || true
$ROOT/.venv/bin/python - <<EOF
import docx
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Pt
d = docx.Document("report.docx")
def paras():
    for p in d.paragraphs: yield p
    for t in d.tables:
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs: yield p
n = 0
for p in paras():
    for r in p.runs:
        if "FILL-" in r.text: r.font.highlight_color = WD_COLOR_INDEX.YELLOW; n += 1
        r.font.name = "Times New Roman"
for s in d.styles:
    try:
        if s.font is not None: s.font.name = "Times New Roman"
        if s.name in ("Normal", "Body Text", "First Paragraph", "Compact"): s.font.size = Pt(12)
    except Exception: pass
names = [x.name for x in d.styles]
for t in d.tables:
    if "Table Grid" in names: t.style = d.styles["Table Grid"]
d.save("$ROOT/docs/report/RetinAI_BE_Project_Report.docx")
print("docx: highlighted", n, "FILL runs,", len(d.inline_shapes), "figures,", len(d.tables), "tables")
EOF
rm -rf $S
