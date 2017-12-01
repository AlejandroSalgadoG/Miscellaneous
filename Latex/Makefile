Anteproyecto.pdf: Anteproyecto.tex Title.tex Anteproyecto.bib
	pdflatex Anteproyecto.tex
	bibtex Anteproyecto
	pdflatex Anteproyecto.tex
	pdflatex Anteproyecto.tex
	pdflatex Anteproyecto.tex

clean:
	rm -f Anteproyecto.pdf
	rm -f *.nav *.snm *.toc* .out *.dvi *.blg *.bbl *.aux *.log

exe:
	mupdf Anteproyecto.pdf
