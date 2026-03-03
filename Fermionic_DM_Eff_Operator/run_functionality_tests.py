import argparse
import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Sequence


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SCRIPT = os.path.join(THIS_DIR, "Fermi-LAT_analysis_eff_coupling_fermionic.py")


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
            "Run smoke tests for Fermionic EFT Fermi-LAT script across multiple operators and fermion types. "
            "Runs single-point + tau-grid (band default + legacy dip) and checks that expected PNGs exist."
        )
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(THIS_DIR, "_test_outputs"),
        help="Output directory (plots will be written under outdir/<operator>/<fermion_type>/...).",
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

    operators = [
        "rayleigh_full",
        "dipole_magnetic",
        "dipole_electric",
        "charge_radius",
        "anapole",
    ]
    fermion_types = ["dirac", "majorana"]

    results: List[CmdResult] = []

    for op in operators:
        for ftype in fermion_types:
            # Majorana forbids dipole operators in this code.
            if ftype == "majorana" and op in ("dipole_magnetic", "dipole_electric"):
                continue

            case_dir = os.path.join(args.outdir, op, ftype)
            os.makedirs(case_dir, exist_ok=True)

            # Single-point plot
            results.append(
                run_cmd(
                    f"single_point.{op}.{ftype}",
                    [
                        sys.executable,
                        SCRIPT,
                        "--operator",
                        op,
                        "--fermion-type",
                        ftype,
                        "--baseline",
                        str(args.baseline),
                        "--Lambda",
                        "1e3",
                        "--mchi",
                        "1.0",
                        "--c_s",
                        "4e-2",
                        "--c_p",
                        "4e-2",
                        "--outdir",
                        case_dir,
                    ],
                )
            )

            # Tau-grid band (default)
            results.append(
                run_cmd(
                    f"tau_grid.band.{op}.{ftype}",
                    [
                        sys.executable,
                        SCRIPT,
                        "--tau-grid",
                        "--operator",
                        op,
                        "--fermion-type",
                        ftype,
                        "--baseline",
                        str(args.baseline),
                        "--dip-depth",
                        str(float(args.dip_depth)),
                        "--tau-grid-n-lambda",
                        str(int(args.grid_n)),
                        "--tau-grid-n-mchi",
                        str(int(args.grid_n)),
                        "--outdir",
                        case_dir,
                    ],
                )
            )

            # Tau-grid legacy dip mode
            results.append(
                run_cmd(
                    f"tau_grid.dip.{op}.{ftype}",
                    [
                        sys.executable,
                        SCRIPT,
                        "--tau-grid",
                        "--tau-energy-mode",
                        "dip",
                        "--dip-energy",
                        "175",
                        "--operator",
                        op,
                        "--fermion-type",
                        ftype,
                        "--baseline",
                        str(args.baseline),
                        "--dip-depth",
                        str(float(args.dip_depth)),
                        "--tau-grid-n-lambda",
                        str(int(args.grid_n)),
                        "--tau-grid-n-mchi",
                        str(int(args.grid_n)),
                        "--outdir",
                        case_dir,
                    ],
                )
            )

            # Verify outputs exist for this case
            assert_glob(os.path.join(case_dir, "spectrum_with_attenuation_*.png"), min_matches=1)
            assert_glob(os.path.join(case_dir, f"tau_vs_lambda_{op}_*.png"), min_matches=1)
            assert_glob(os.path.join(case_dir, f"tau_grid_{op}_*.png"), min_matches=1)

    for r in results:
        print_result(r)

    failed = [r for r in results if int(r.returncode) != 0]
    if failed:
        raise RuntimeError(
            "One or more commands failed:\n" + "\n".join([f"- {f.name}: rc={f.returncode}" for f in failed])
        )

    print("=" * 80)
    print("All fermionic functionality tests passed.")
    print(f"Outputs written under: {os.path.abspath(args.outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
