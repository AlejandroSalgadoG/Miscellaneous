Presentation.pdf:  Presentation.tex Presentation.bib Pdf.tex Cdf.tex MGenerating.tex Moments.tex Relations.tex Generation.tex Application.tex
	pdflatex Presentation.tex
	bibtex Presentation
	pdflatex Presentation.tex
	pdflatex Presentation.tex
	pdflatex Presentation.tex

clean:
	rm -f Presentation.pdf
	rm -f *.nav *.snm *.toc* *.out *.dvi *.blg *.bbl *.aux *.log

exe:
	evince Presentation.pdf &
