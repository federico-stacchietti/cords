# Using `gradmatch_core`

`gradmatch_core` contains a minimal implementation of GradMatch that ships with this repository. To import it in your projects you must ensure Python can locate the package.

From the repository root run:

```bash
pip install -e .
```

This installs `cords` and `gradmatch_core` in editable mode so they are importable anywhere. Alternatively you may add the repository path to `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/cords:$PYTHONPATH
```

Replace `/path/to/cords` with the absolute path to this repository.

See `examples/gradmatch_core_custom_dataset.py` for a minimal usage example.
