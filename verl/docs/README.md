# verl documentations

## Build the docs

```bash
# If you want to view auto-generated API docstring, please make sure verl is available in python path. For instance, install verl via:
# pip install .. -e[test]

# Install dependencies needed for building docs.
pip install -r requirements-docs.txt

# Build the docs.
make clean
make html
```

## Open the docs with your browser

```bash
python -m http.server -d _build/html/
```
