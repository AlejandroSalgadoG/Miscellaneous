Final.pdf:  Final.tex MpiGrammar.tex MpiDistGrammar.tex MpiFile.tex
	pdflatex Final.tex
	pdflatex Final.tex
	rm -f *.nav *.snm *.toc* .out *.dvi *.blg *.bbl *.aux *.log

clean:
	rm -f Final.pdf
	rm -f *.nav *.snm *.toc* .out *.dvi *.blg *.bbl *.aux *.log

exe:
	mupdf Final.pdf
