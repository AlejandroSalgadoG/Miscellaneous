Proyecto.pdf: Proyecto.tex Proyecto.bib Title.tex
	pdflatex Proyecto.tex
	bibtex Proyecto
	pdflatex Proyecto.tex
	pdflatex Proyecto.tex
	pdflatex Proyecto.tex

clean:
	rm -f Proyecto.pdf
	rm -f *.nav *.snm *.toc* .out *.dvi *.blg *.bbl *.aux *.log

exe:
	evince Proyecto.pdf &
