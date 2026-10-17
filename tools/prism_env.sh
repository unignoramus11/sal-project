# Source on the PC (or Ada) before running scripts that use PRiSM's model wrappers:
#   . tools/prism_env.sh
# Activates envs/prism (Python 3.10, torch 2.6 + cu126, k2, lhotse, ESPnet fork, icefall) and points
# phonemizer at the bundled eSpeak-NG library (needed only so the wav2vec2 phoneme tokenizer loads).
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$_root/envs/prism/bin/activate"
_esp="$(python -c 'import espeakng_loader as e, os; print(os.path.dirname(e.get_library_path()))')"
export PHONEMIZER_ESPEAK_LIBRARY="$_esp/libespeak-ng.so"
export ESPEAK_DATA_PATH="$_esp/espeak-ng-data"
export PATH="/usr/lib/wsl/lib:$PATH"
export PYTHONWARNINGS="ignore::FutureWarning"
unset _root _esp
