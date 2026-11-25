.PHONY: proposal plot report final clean

proposal:
	cd proposal && latexmk -pdf -interaction=nonstopmode proposal.tex

plot:
	cd proposal && uv run --with matplotlib --with numpy python make_plot.py

report:
	uv run python scripts/make_tables.py
	cd paper && latexmk -pdf -interaction=nonstopmode report.tex
	uv run python scripts/check_report_numbers.py

# the final report: regenerate every table and figure from results/, build, then check every prose number
final:
	uv run python scripts/make_tables.py
	uv run python scripts/make_figures.py
	cd paper && latexmk -pdf -interaction=nonstopmode final.tex
	uv run python scripts/check_report_numbers.py --tex paper/final.tex

clean:
	cd proposal && latexmk -c
