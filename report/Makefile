DOC=writeup
LATEX=/Library/TeX/texbin/latex
PDFLATEX=/Library/TeX/texbin/pdflatex
BIBTEX=/Library/TeX/texbin/bibtex


all: $(DOC).pdf 

ShumanNotes.sty: $(DOC).pdf

clean:
	rm $(DOC).log \
	$(DOC).pdf \
	$(DOC).toc \
	$(DOC).nav \
	$(DOC).snm \
	$(DOC).aux \
	$(DOC).out \
	$(DOC).toc

$(DOC).pdf: $(DOC).tex citations.bib
	$(PDFLATEX) -file-line-error -interaction=nonstopmode $(DOC)
	$(BIBTEX)  $(DOC)
	$(PDFLATEX) -file-line-error -interaction=nonstopmode $(DOC)
	$(PDFLATEX) -file-line-error -interaction=nonstopmode $(DOC)
