#!/usr/bin/env python3
"""Opt-in test-time order ablation for SAL representation probes.

Run from the root of sal-project-nov. Existing training jobs are unchanged
unless --order-ablation is explicitly passed to scripts/run_rp.py.
"""
from pathlib import Path
import shutil

root = Path.cwd()
train = root / 'src/sal/probes/train.py'
run = root / 'scripts/run_rp.py'
if not train.is_file() or not run.is_file():
    raise SystemExit('ERROR: Run this script from the sal-project-nov repository root')

old_train = train.read_text()
old_run = run.read_text()
if 'post_eval: Callable[[torch.nn.Module], dict]' in old_train and 'ap.add_argument("--order-ablation"' in old_run:
    print('Already patched; nothing changed.')
    raise SystemExit(0)
if 'post_eval: Callable[[torch.nn.Module], dict]' in old_train or 'ap.add_argument("--order-ablation"' in old_run:
    raise SystemExit('ERROR: Partial patch detected; inspect files before retrying')


def change(s, a, b, path):
    hits = s.count(a)
    if hits != 1:
        raise SystemExit(f'ERROR: Expected one matching anchor in {path}, found {hits}: {a[:80]!r}. No files changed.')
    return s.replace(a, b, 1)

new_train = change(
    old_train,
    '    log: Callable[[str], None] = print,\n) -> dict:',
    '    log: Callable[[str], None] = print,\n'
    '    post_eval: Callable[[torch.nn.Module], dict] | None = None,\n) -> dict:',
    train,
)
new_train = change(
    new_train,
    '        result["predictions"][name] = (idx, p)\n    return result',
    '        result["predictions"][name] = (idx, p)\n'
    '    if post_eval is not None:\n'
    '        result["order_ablation"] = post_eval(model)\n'
    '    return result',
    train,
)

new_run = change(old_run, 'import pandas as pd\n', 'import pandas as pd\nimport torch\n', run)
new_run = change(
    new_run,
    'from sal.probes.train import TrainConfig, save_run, train_probe',
    'from sal.probes.train import TrainConfig, evaluate, predict, save_run, train_probe',
    run,
)
new_run = change(
    new_run,
    'ap.add_argument("--tag", default="")',
    'ap.add_argument("--tag", default="")\n'
    'ap.add_argument("--order-ablation", action="store_true",\n'
    '                help="save best model and evaluate test-only chunk shuffles")',
    run,
)
new_run = change(
    new_run,
    'args = ap.parse_args()',
    'args = ap.parse_args()\n'
    'if args.order_ablation and not args.tag:\n'
    '    ap.error("--order-ablation requires --tag to avoid overwriting earlier runs")',
    run,
)
new_run = change(
    new_run,
    '    res = train_probe(make, batcher, targets, splits, cfg)',
    '''    def post_eval(model):
        # Called with the best-validation weights after normal evaluation.
        outdir = args.out_root / name
        outdir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                   outdir / "best_model.pt")
        scores = {}
        for ms in (None, 200, 100, 20):
            test_batcher = FrameBatcher(
                store, meta["metadata_idx"].to_numpy(), targets,
                chunk_shuffle_ms=ms, seed=seed,
            )
            pred = predict(model, test_batcher, splits["test"], cfg)
            condition = "original" if ms is None else f"shuffle_{ms}ms"
            scores[condition] = evaluate(pred, targets[splits["test"]])
            print("ABLATION", condition, json.dumps(scores[condition]), flush=True)
        return scores

    res = train_probe(make, batcher, targets, splits, cfg,
                      post_eval=post_eval if args.order_ablation else None)''',
    run,
)

# Compile both complete source files before writing either one.
compile(new_train, str(train), 'exec')
compile(new_run, str(run), 'exec')

backups = [train.with_name('train.py.before_order_ablation'),
           run.with_name('run_rp.py.before_order_ablation')]
if any(p.exists() for p in backups):
    raise SystemExit('ERROR: A backup already exists; nothing changed')
for source, backup in zip((train, run), backups):
    shutil.copy2(source, backup)
train.write_text(new_train)
run.write_text(new_run)
print('PATCH_INSTALLED: optional --order-ablation enabled')
print('Existing jobs are unaffected unless passed --order-ablation.')
print('Backups:', *(str(p.relative_to(root)) for p in backups), sep='\n  ')
