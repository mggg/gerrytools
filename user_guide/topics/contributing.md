# Contributing

Start with the repository's
[contribution guide](https://github.com/mggg/gerrytools/blob/2.0.0/CONTRIBUTING.md) for development
setup, testing expectations, and the pull-request process.

## Documentation checks

The documentation source lives under `user_guide/`. User-guide notebooks are committed without
outputs; the build executes them into an ignored MyST-NB cache.

```console
task docs
task docs-test
task docs-linkcheck
```

Use `task docs-serve` for a live local preview. To compare the development palette and code-theme
controls, set `DOCS_SWITCHER=1` before starting the server.

When a notebook has been edited interactively, the pre-commit hook removes output and normalizes
kernel metadata. You can run the same cleanup directly:

```console
uv run python user_guide/_clear_notebook_outputs.py user_guide/user/example.ipynb
```

Documentation examples should use public imports, deterministic data, and explicit skip reasons
when execution would require a network service, a credential, or Docker. The
{doc}`documentation style reference <../style_reference>` records the prose, color, and notebook
conventions these pages follow.
