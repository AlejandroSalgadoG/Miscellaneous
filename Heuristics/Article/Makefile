Trabajo4.pdf: Trabajo4.tex Trabajo4.bib
	pdflatex Trabajo4.tex
	bibtex Trabajo4
	pdflatex Trabajo4.tex
	pdflatex Trabajo4.tex
	pdflatex Trabajo4.tex

clean:
	rm -f Trabajo4.pdf
	rm -f *.nav *.snm *.toc *.out *.dvi *.blg *.bbl *.aux *.log

exe:
	evince Trabajo4.pdf &
