# AGENTS.md

Guidance for AI coding agents in this repository. The full AI policy is in
`.github/CONTRIBUTING.rst`.

NeuroKit2 is a Python toolbox for physiological signal processing used in research: a change
to a numerical result changes people's findings.

## AI policy

- **A human stays in the loop.** Do not open PRs or issues, post comments, or push on your own.
  Prepare the change locally and hand it back. The human reviews every line and writes all PR,
  issue, and comment text in their own words.
- **Label fully automated submissions.** If you open a PR or issue that no human has reviewed
  line by line, end its title with `[beep boop]` (e.g.
  `[Fix] rsp_rate: handle empty troughs [beep boop]`). These are closed without review.
- **No AI co-author trailers** (`Co-Authored-By:` etc.) and no bot as commit author.

## Keep changes small

- **One change per PR**: one fix, one feature, or one improvement. Never bundle a fix with
  refactoring, vectorization, speedups, new options, style changes, or other fixes. Found
  several bugs? Tell the human; each gets its own branch, test, and NEWS entry.
- **Stop and ask the human** if the change exceeds ~75 lines of implementation (excluding
  tests/docs), touches more than one public function or more than one submodule, or needs a
  new file or dependency. Offer the smaller version or the split.
- **Sketch first.** For anything beyond a local fix, agree on the design with the human before
  writing code. New features need an issue a maintainer has agreed to.
- **YAGNI.** No new parameters, `method=` options, input-type branches, or checks for impossible
  inputs unless a test in the same PR needs them.
- **Reuse.** Check `neurokit2/signal/` and `neurokit2/misc/` for existing helpers. Fix shared
  behavior once in `signal_*`, not separately in `ecg_`/`ppg_`/`rsp_`.
- **Speedups are their own PR**, with an interleaved A/B benchmark against an unmodified `dev`
  and a check that outputs are unchanged.

## Scientific correctness

Any change to numerical output (formula, scaling, threshold, default, output name) must cite
the equation, section, or reference implementation it follows, and be listed under "Changes in
Results" in `NEWS.rst`. A method named `"author2020"` follows Author (2020), not another paper.

## Boundaries

Ask first before: adding or bumping dependencies; editing `.github/`, `.pre-commit-config.yaml`,
`pyproject.toml`, or `uv.lock`; changing public API (submodule `__all__`, signatures, defaults,
output column names such as `ECG_Rate`).

Never: skip, loosen, or delete tests; use `--no-verify` or silence lint; commit data files;
reformat unrelated code; target `master` (work branches off `dev`); adapt code whose license is
not MIT-compatible or is unknown (attribute adapted code in a comment with source and license).

## Commands

```bash
uv sync --all-groups --all-extras && uv run pre-commit install
uv run pre-commit run --files <changed files>
uv run pytest tests/tests_rsp.py::test_rsp_peaks
uv run sphinx-build -j auto -b html docs/ docs/_build/html
```

Run pre-commit and the tests for the changed module before handing back. Report failures.

## Code layout and style

- Public functions are `nk.<prefix>_<name>()`, usually one per file (`ecg/ecg_peaks.py`). New public
  functions go in the submodule's `__all__` and get an `.. autofunction::` entry in the right
  section of `docs/functions/<module>.rst` (not only the catch-all `automodule` at the bottom).
- Pipeline: `*_simulate` → `*_clean` → `*_peaks` → `*_rate` → `*_process` → `*_analyze` →
  `*_plot`. Output names are `<SIGNAL>_<Feature>`.
- Readable code over clever code. Follow the existing docstring style (bold summary line,
  `.. ipython:: python` examples, References as a bullet list). Errors start with
  `"NeuroKit error: <function>(): "`. Import optional dependencies inside the function.

## Tests

- Add to the existing `tests/tests_<module>.py`; never create a new test file. Prefer extending an
  existing test or `parametrize` over a new test function.
- One focused regression test per bug, using `nk.*_simulate()` with a fixed `random_state`
  rather than `nk.data()` (network).
- Confirm the test fails without the fix: make sure your local `dev` is up to date, restore the
  source files from it (`git diff dev -- neurokit2/ > fix.patch && git checkout dev -- neurokit2/`),
  run the test, then `git apply fix.patch` and rerun.

## Changelog

Add an entry to `NEWS.rst` under the top version in the matching subsection ("Compatibility",
"Changes in Results", "Breaking Changes", "New Features", "Fixes"). New contributors add themselves to
`AUTHORS.rst`.
