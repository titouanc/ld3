all: report.pdf

%.pdf: %.tex
	pdflatex $<
	pdflatex $<