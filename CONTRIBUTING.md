# Contributing to himotoki-split

Thank you for your interest in contributing to **himotoki-split**! We welcome contributions of all kinds, from bug fixes and feature implementations to documentation improvements.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- `uv` (recommended) or `pip`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/msr2903/himotoki-split.git
   cd himotoki-split
   ```

2. Install the package in editable mode with development dependencies:
   ```bash
   # Using uv (recommended)
   uv pip install -e ".[dev]"

   # Or using pip
   pip install -e ".[dev]"
   ```

## Development Workflow

### Coding Standards

- Follow PEP 8 style guidelines.
- Use `black` for formatting.
- Include docstrings for all public modules, classes, and functions.
- Keep tests updated and ensured they pass before contributing.

### Running Tests

We use `pytest` for testing. Before submitting a pull request, ensure all tests pass:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=himotoki_split
```

### Submitting Changes

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Commit your changes with clear, descriptive commit messages.
3. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request on GitHub. Provide a clear description of the changes and link to any relevant issues.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on the [GitHub Issues](https://github.com/msr2903/himotoki-split/issues) page.

Include the following information in bug reports:
- A clear, descriptive title.
- Steps to reproduce the issue.
- Expected behavior vs. actual behavior.
- Your environment (Python version, OS, etc.).

---

Thank you for making himotoki-split better!
