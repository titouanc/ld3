all: report.pdf

%.pdf: %.tex %.bib
	pdflatex $<
	bibtex $(subst .tex,,$<)
	pdflatex $<
	pdflatex $<
	texcount $<
