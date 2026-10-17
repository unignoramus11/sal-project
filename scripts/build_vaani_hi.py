"""Rebuild Vaani-Hi from the public Vaani corpus (PRiSM recipe). See src/sal/data/rebuild.py.

Run:  uv run python scripts/build_vaani_hi.py                # full build into data/vaani_hi_rebuild
      uv run python scripts/build_vaani_hi.py --plan-only    # print the district/shard plan
"""
import argparse
import json
from pathlib import Path

from sal.data import rebuild
from sal.paths import DATA

ap = argparse.ArgumentParser()
ap.add_argument("--revision", default=None, help="Vaani commit (default: current head, then pinned)")
ap.add_argument("--raw-dir", type=Path, default=DATA / "vaani_hi_rebuild" / "raw")
ap.add_argument("--out-dir", type=Path, default=DATA / "vaani_hi_rebuild")
ap.add_argument("--pincode-csv", type=Path, default=DATA / "external" / "pincode.csv")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--plan-only", action="store_true")
args = ap.parse_args()

plan_file = args.out_dir / "plan.json"
if plan_file.exists() and args.revision is None:
    plan = rebuild.Plan(**json.loads(plan_file.read_text()))
    print(f"using saved plan at revision {plan.revision}")
else:
    rev = rebuild.resolve_revision(args.revision)
    plan = rebuild.make_plan(rev)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_file.write_text(json.dumps(rebuild.asdict(plan), indent=2))
    print(f"new plan at revision {rev}")

for state, ds in plan.districts.items():
    print(f"  {state:16s} {ds}")
print(f"  {len(plan.files)} shards")
if args.plan_only:
    raise SystemExit

rebuild.download(plan, args.raw_dir, workers=args.workers)
m = rebuild.assemble(plan, args.raw_dir, args.pincode_csv, args.out_dir)
print(json.dumps(m["counts"], indent=2, ensure_ascii=False)[:3000])
print("sha256", m["output"]["sha256"])
