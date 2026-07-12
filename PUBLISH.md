# Publishing himotoki-split

Repo: https://github.com/msr2903/himotoki-split

## Soft-launch checklist (Phase C)

1. **Tests**
   ```bash
   pip install -e ".[dev]"
   pytest tests/ -q
   ```
2. **Build**
   ```bash
   pip install build twine
   python -m build
   twine check dist/*
   ```
3. **Local install smoke**
   ```bash
   python -m venv /tmp/hs-smoke && /tmp/hs-smoke/bin/pip install dist/*.whl
   /tmp/hs-smoke/bin/python -c "from himotoki_split import split; print(split('猫が食べる', fallback=False).segments)"
   ```
4. **Optional ONNX**
   ```bash
   /tmp/hs-smoke/bin/pip install 'himotoki-split[onnx]'
   ```
5. **PyPI (TestPyPI first)**
   ```bash
   twine upload --repository testpypi dist/*
   # then:
   twine upload dist/*
   ```

Requires a PyPI API token (`TWINE_USERNAME=__token__`, `TWINE_PASSWORD=pypi-...`).

## GitHub Actions CI

Copy when the push token has `workflow` scope:

```bash
cp docs/ci.yml.example .github/workflows/ci.yml
git add .github/workflows/ci.yml && git commit -m "ci: enable GitHub Actions" && git push
```

## Creating the GitHub repository

Already created: `msr2903/himotoki-split`. Historical note for other forks:

```bash
gh repo create OWNER/himotoki-split --public --source=. --remote=origin --push
```
