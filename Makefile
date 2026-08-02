PAPER       = ResearchReport
STY         = nips15submit_e.sty
FIGURES     = plots/sweep_C/fig_latency_vs_C.pdf \
              plots/sweep_C/fig_speedup_vs_C_logx.pdf \
              plots/sweep_C/fig_break_even.pdf
ARXIV_ZIP   = arxiv_submission.zip
PDFLATEX    = pdflatex -interaction=nonstopmode

.PHONY: all pdf arxiv clean

all: pdf arxiv

pdf: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(STY) $(FIGURES)
	$(PDFLATEX) $(PAPER).tex | grep -E "^!|Warning|Error|Output written" || true
	$(PDFLATEX) $(PAPER).tex | grep -E "^!|Warning|Error|Output written" || true

arxiv: $(ARXIV_ZIP)

$(ARXIV_ZIP): $(PAPER).tex $(STY) $(FIGURES)
	zip -j $@ $(PAPER).tex $(STY)
	zip $@ $(FIGURES)

clean:
	rm -f $(PAPER).pdf $(PAPER).aux $(PAPER).log $(PAPER).out \
	      $(PAPER).toc $(ARXIV_ZIP)
