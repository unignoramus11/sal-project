.PHONY: proposal plot clean

proposal:
	cd proposal && latexmk -pdf -interaction=nonstopmode proposal.tex

plot:
	cd proposal && uv run --with matplotlib --with numpy python make_plot.py

clean:
	cd proposal && latexmk -c
