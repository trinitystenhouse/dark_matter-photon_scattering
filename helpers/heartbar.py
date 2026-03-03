# helpers/heartbar.py
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


@dataclass(frozen=True)
class HeartBarStyle:
    """
    Heart-themed tqdm styling.

    Notes:
    - Prefer '♥♡' (single-width) over emoji hearts for stable alignment.
    - ascii="♡♥" means: empty="♡", filled="♥"
    """
    ascii: str = "♡♥"
    desc_prefix: str = "♥ "
    bar_format: Optional[str] = None  # leave None to use tqdm default


DEFAULT_HEART_STYLE = HeartBarStyle()


def heart_tqdm(
    it: Optional[Iterable[T]] = None,
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    style: HeartBarStyle = DEFAULT_HEART_STYLE,
    **kwargs,
) -> "tqdm[T]":
    """
    Create a tqdm progress bar filled with hearts.

    Usage:
        for x in heart_tqdm(range(100), desc="sampling"):
            ...

        pbar = heart_tqdm(total=1000, desc="emcee")
        ...
    """
    if desc is not None and style.desc_prefix:
        desc = f"{style.desc_prefix}{desc}"

    # tqdm's ascii expects a 2-char string for [empty, filled] in many cases.
    # "♡♥" works well on most terminals.
    kwargs.setdefault("ascii", style.ascii)

    if style.bar_format is not None:
        kwargs.setdefault("bar_format", style.bar_format)

    return tqdm(it, total=total, desc=desc, **kwargs)


@contextmanager
def heart_pbar(
    *,
    total: int,
    desc: str = "working",
    style: HeartBarStyle = DEFAULT_HEART_STYLE,
    **kwargs,
) -> Iterator["tqdm[None]"]:
    """
    Context manager for manual updates.

    Usage:
        with heart_pbar(total=nsteps, desc="emcee") as pbar:
            for ...:
                pbar.update(1)
    """
    pbar = heart_tqdm(total=total, desc=desc, style=style, **kwargs)
    try:
        yield pbar
    finally:
        pbar.close()


def emcee_heart_sample(
    sampler,
    p0,
    *,
    nsteps: int,
    desc: str = "emcee",
    style: HeartBarStyle = DEFAULT_HEART_STYLE,
    **kwargs,
):
    """
    Wrap emcee's sampling loop with a heart tqdm.

    Usage:
        for state in emcee_heart_sample(sampler, p0, nsteps=50000):
            ...
    """
    # Ensure emcee doesn't also print its own progress
    kwargs.setdefault("progress", False)

    with heart_pbar(total=nsteps, desc=desc, style=style) as pbar:
        for state in sampler.sample(p0, iterations=nsteps, **kwargs):
            pbar.update(1)
            yield state