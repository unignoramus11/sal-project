.PHONY: proposal plot report final analysis clean

proposal:
	cd proposal && latexmk -pdf -interaction=nonstopmode proposal.tex

plot:
	cd proposal && uv run --with matplotlib --with numpy python make_plot.py

report:
	uv run python scripts/make_tables.py
	cd paper && latexmk -pdf -interaction=nonstopmode report.tex
	uv run python scripts/check_report_numbers.py

# every analysis that reads run directories, in dependency order (run after pulling new runs:
# tools/pull_results.sh for the PC, git pull for Ada); then `make final`
analysis:
	uv run python scripts/summarise.py
	uv run python scripts/control_summary.py
	uv run python scripts/state_level.py
	uv run python scripts/official_eval.py
	uv run python scripts/rebuild_checks.py
	uv run python scripts/fuse.py tp_zipa_rebuild_phone_gru_speaker_split prosody_gru_speaker_split
	uv run python scripts/prosody_checks.py
	uv run python scripts/mdl.py --summary
	uv run python scripts/benchmark_eval.py

# the final report: regenerate every table and figure from results/, build, then check every prose number
final:
	uv run python scripts/make_tables.py
	uv run python scripts/make_figures.py
	cd paper && latexmk -pdf -interaction=nonstopmode final.tex
	uv run python scripts/check_report_numbers.py --tex paper/final.tex

clean:
	cd proposal && latexmk -c
