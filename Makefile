ARCHIVE = TitouanChristophe_vub.tar.gz

all: ${ARCHIVE}

${ARCHIVE}: report.pdf ld3q1.py ld3q2.py simulate.py strategies.py plot.py local_config.py
	tar c $^ | gzip > $@

%.pdf: %.tex %.bib
	pdflatex $<
	bibtex $(subst .tex,,$<)
	pdflatex $<
	pdflatex $<
	texcount $<
