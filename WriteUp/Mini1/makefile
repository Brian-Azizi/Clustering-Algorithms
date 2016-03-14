X = fmm

mini.pdf : $(X).tex references.bib
	pdflatex $(X).tex
	bibtex $(X).aux
	pdflatex $(X).tex
	pdflatex $(X).tex

clean : 
	rm $(X).aux $(X).bbl $(X).blg $(X).log $(X).pdf $(X).spl