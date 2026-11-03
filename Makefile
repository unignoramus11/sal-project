.PHONY: proposal plot report clean

proposal:
	cd proposal && latexmk -pdf -interaction=nonstopmode proposal.tex

plot:
	cd proposal && uv run --with matplotlib --with numpy python make_plot.py

report:
	uv run python scripts/make_tables.py
	cd paper && latexmk -pdf -interaction=nonstopmode report.tex
	uv run python scripts/check_report_numbers.py

clean:
	cd proposal && latexmk -c
