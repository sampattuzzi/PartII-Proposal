# This is the Makefile for the demonstration dissertation
# written by Martin Richards
#
# Note that continuation lines require '\'
# and that TAB is used after ':' and before unix commands.

DISS = diss.tex propbody.tex refs.bib diss.sum frontmatter.tex appendix.tex

pandoc = /home/sam/.cabal/bin/pandoc

pandoc_opts = --chapters --template=template.latex

PROP = proposal.tex propbody.tex refs.bib

help:
	@echo
	@echo "USAGE:"
	@echo
	@echo "make			display help information"
	@echo "make proposal.pdf     	make the proposal and view it using xdvi"
	@echo "make diss.pdf 		make a .pdf version of the dissertation"
	@echo "make diss		view dissertation"
	@echo "make clean    		remove all remakeable files"
	@echo

proposal.pdf:	$(PROP)
	pdflatex proposal
	bibtex proposal
	pdflatex proposal
	pdflatex proposal

diss.pdf:	$(DISS)
	pdflatex diss.tex
	biber diss
	pdflatex diss.tex

diss:		diss.pdf
	evince diss.pdf &

diss.sum: diss.tex
	texcount -1 -sum diss.tex > diss.sum

wc:	diss.sum
	cat diss.sum

#diss.latex:	$(DISS)
#	$(pandoc) diss.md -o diss.tex $(pandoc_opts)

clean:
	rm -f diss.ps *.dvi *.aux *.log *.err
	rm -f core *~ *.lof *.toc *.blg *.bbl
	rm -f makefile.txt
