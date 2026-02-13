#!/usr/bin/env python3

import os

from check_component import run_component_check


def main():
    repo_dir = os.environ.get("REPO_PATH")
    if repo_dir is None:
        raise SystemExit("REPO_PATH not set")

    data_dir = os.path.join(repo_dir, "fermi_data", "totani")
    counts = os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits")
    template = os.path.join(data_dir, "processed", "templates", "mu_ics_counts.fits")
    plot_dir = os.path.join(os.path.dirname(__file__), "plots_check_ics")

    return run_component_check(
        label="ICS",
        template_path=template,
        counts_path=counts,
        plot_dir=plot_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
