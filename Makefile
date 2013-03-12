# This is the Makefile for the demonstration dissertation
# written by Martin Richards
#
# Note that continuation lines require '\'
# and that TAB is used after ':' and before unix commands.

DISS = diss.md Makefile

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
	pandoc diss.md -o diss.pdf -V geometry:"margin=1in"

diss:		diss.pdf
	evince diss.pdf &

makefile.txt:	Makefile
	expand Makefile >makefile.txt

clean:
	rm -f diss.ps *.dvi *.aux *.log *.err
	rm -f core *~ *.lof *.toc *.blg *.bbl
	rm -f makefile.txt
