# BE project report LaTeX template

Start by reading `Student_Instructions.pdf`. Upload this folder as a ZIP to
Overleaf, select **XeLaTeX**, and set **main.tex** as the main document.
Locally, run `latexmk -xelatex main.tex` or run `xelatex main.tex` twice.
Recompile again after major content or reference changes until page numbers settle.

## First edits
1. Edit project_details.tex: project title, exact approved degree/program,
   student names and roll numbers, guide, department, institute and submission year.
2. Insert the approved institute emblem as figures/institute_logo.png.
   The supplied PDF has another institute's example emblem: it is NOT reused here.
3. Replace instructional prompts in frontmatter, chapters and backmatter.
4. Keep finalcopyfalse before the viva; set finalcopytrue ONLY for the approved
   final post-viva copy. Fill approval details only after actual approval.
5. Optional switches control dedication, publications, co-guide and cover artwork.

## Required font
The guideline requires Times New Roman, not merely a Times-like font.
The template uses installed Times New Roman when available. Otherwise place your
legally obtained files `times.ttf`, `timesbd.ttf`, `timesi.ttf`, `timesbi.ttf`
in fonts/. If neither is present, it compiles with a clearly warned preview
substitute (Nimbus Roman or TeX Gyre Termes). The included preview uses Nimbus
Roman. The substitute is NOT claimed to meet the final font requirement.

## Source rules and implementation choices
- A4; body 12 pt; 1.5 line spacing; 160 x 245 mm writing area.
- Physical left/right margins 30/20 mm, no running header, top 30 and bottom 22 mm.
  No additional gutter is added. Confirm any mirrored duplex layout with guide.
- Chapter headings 18 pt, sections 16 pt, subsections 14 pt; centred chapter headings.
- The source says an ADDITIONAL 75 mm chapter top margin. This template applies
  75 mm above the numbered chapter heading in addition to the normal 30 mm top.
  Confirm interpretation with the guide before binding; the source is approximate.
- Source paragraph gap wording is approximate. The template interprets a 2.5-line
  paragraph baseline gap as ordinary baseline plus 1.5 baselines of paragraph skip.
- Front auxiliary pages use lower-case Roman numerals; first auxiliary page is
  counted but unprinted. Title/optional cover are outside that sequence.
- Introduction starts Arabic page 1, hidden only on that first page. Later chapter
  opening numbers are printed. Chapters start fresh pages without forced recto blanks.
- Appendix headings use I, II, III, IV and precede Literature Cited.
- Acknowledgements are at the end as the PDF prescribes, after literature cited
  and any optional publications.
- Source section 3.7 contains inconsistent words alongside Roman iii; the template
  follows a continuous Roman sequence and places declaration after approval when
  approval is included, otherwise after certificate.

## Customising the team
Four student placeholders are provided. For fewer/more students, edit the macros
StudentTitleRows and StudentSignatures in settings.tex AND the student wording in
certificate.tex, approval.tex and declaration.tex. Check every preliminary page.

## Suggested content versus mandated format
Eight main chapters and the appendix topics are RECOMMENDATIONS for a BE hardware,
software or robotics project. The source mandates Introduction first, Review of
Literature normally second, Results and Discussions penultimate, and Conclusions
last. Merge or rename intervening chapters with guide approval.
Requirements traceability, individual contributions, meeting logs, SDG evidence and
the requirement-versus-result table are departmental evidence recommendations;
they are not described as extra mandatory pages of the attached guideline.

## Technical notes
- Contents, figures and tables lists are automatic; do not type page numbers.
- Use chapter/section/subsection commands and label/ref for references.
- Tables need captions ABOVE; figures need captions BELOW; cite reproduced material.
- The six-column verification table is intended for short entries and evidence IDs.
  Use a longtable or a detailed appendix if actual entries are long.
- No real project results or literature citations have been invented. Replace the
  demonstration bibliography entry and every square-bracketed placeholder.
- No binding flyleaves are inserted in the PDF. Ask the binder for two white sheets
  at the beginning and end, in addition to normal binding sheets.
- The final hard cover should be black with gold lettering, including the spine.
- BE report printing is double-sided at actual A4 size. Check print duplex order
  and hidden-page numbering with the guide before binding.

## Files
main.tex / settings.tex / project_details.tex : entry point, format and metadata
frontmatter/ : cover, title, certificate, approval, declaration, abstract and symbols
chapters/ : recommended chapter structure and writing prompts
backmatter/ : appendices, bibliography, publications and acknowledgements
Student_Instructions.pdf : guide to be given to students
Student_Instructions.tex : editable LaTeX version of that guide
BE_Project_Report_Preview.pdf : compiled template with placeholders, not a final report
figures/ and fonts/ : student-provided assets (no external logo/font included)
