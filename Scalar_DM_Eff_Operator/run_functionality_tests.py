import argparse
import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Sequence


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SCRIPT = os.path.join(THIS_DIR, "Fermi-LAT_analysis_eff_coupling_scalar.py")


@dataclass
class CmdResult:
    name: str
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str


def run_cmd(name: str, cmd: Sequence[str]) -> CmdResult:
    proc = subprocess.run(
        list(cmd),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    return CmdResult(
        name=name,
        cmd=list(cmd),
        returncode=int(proc.returncode),
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def assert_glob(pattern: str, *, min_matches: int = 1) -> List[str]:
    matches = sorted(glob.glob(pattern))
    if len(matches) < int(min_matches):
        raise RuntimeError(f"Expected at least {min_matches} match(es) for pattern: {pattern}\nFound: {matches}")
    return matches


def print_result(res: CmdResult) -> None:
    print("=" * 80)
    print(f"[{res.name}] rc={res.returncode}")
    print(" ".join(res.cmd))
    if res.stdout.strip():
        print("--- stdout ---")
        print(res.stdout.rstrip())
    if res.stderr.strip():
        print("--- stderr ---")
        print(res.stderr.rstrip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run smoke tests for Scalar EFT Fermi-LAT script. Runs single-point + tau-grid (band default + legacy dip) "
            "and checks that expected PNGs exist."
        )
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(THIS_DIR, "_test_outputs"),
        help="Output directory (plots will be written under outdir).",
    )
    parser.add_argument(
        "--grid-n",
        type=int,
        default=25,
        help="Grid size for tau-grid (used for both Lambda and mchi).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="cosmic",
        choices=["gc", "cosmic"],
        help="Baseline choice for the smoke tests.",
    )
    parser.add_argument(
        "--dip-depth",
        type=float,
        default=0.01,
        help="Target attenuation depth used to define tau_needed (smoke-test value).",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    results: List[CmdResult] = []

    # Single-point plot
    results.append(
        run_cmd(
            "single_point",
            [
                sys.executable,
                SCRIPT,
                "--baseline",
                str(args.baseline),
                "--Lambda",
                "1e6",
                "--mchi",
                "1.0",
                "--c_phi",
                "4e-2",
                "--outdir",
                args.outdir,
            ],
        )
    )

    # Tau-grid band (default)
    results.append(
        run_cmd(
            "tau_grid.band",
            [
                sys.executable,
                SCRIPT,
                "--tau-grid",
                "--baseline",
                str(args.baseline),
                "--dip-depth",
                str(float(args.dip_depth)),
                "--tau-grid-n-lambda",
                str(int(args.grid_n)),
                "--tau-grid-n-mchi",
                str(int(args.grid_n)),
                "--outdir",
                args.outdir,
            ],
        )
    )

    # Tau-grid legacy dip mode
    results.append(
        run_cmd(
            "tau_grid.dip",
            [
                sys.executable,
                SCRIPT,
                "--tau-grid",
                "--tau-energy-mode",
                "dip",
                "--dip-energy",
                "175",
                "--baseline",
                str(args.baseline),
                "--dip-depth",
                str(float(args.dip_depth)),
                "--tau-grid-n-lambda",
                str(int(args.grid_n)),
                "--tau-grid-n-mchi",
                str(int(args.grid_n)),
                "--outdir",
                args.outdir,
            ],
        )
    )

    # Verify outputs exist
    assert_glob(os.path.join(args.outdir, "spectrum_with_attenuation_*.png"), min_matches=1)
    assert_glob(os.path.join(args.outdir, f"tau_vs_lambda_scalar_{args.baseline}.png"), min_matches=1)
    assert_glob(os.path.join(args.outdir, f"tau_grid_scalar_{args.baseline}.png"), min_matches=1)

    for r in results:
        print_result(r)

    failed = [r for r in results if int(r.returncode) != 0]
    if failed:
        raise RuntimeError(
            "One or more commands failed:\n" + "\n".join([f"- {f.name}: rc={f.returncode}" for f in failed])
        )

    print("=" * 80)
    print("All scalar functionality tests passed.")
    print(f"Outputs written under: {os.path.abspath(args.outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
