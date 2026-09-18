# EasyWAM website

This branch contains the static EasyWAM project site. It is organized like a small source project while keeping the branch-root `index.html` available for GitHub Pages.

## Edit the site

| Content | Location |
| --- | --- |
| Page shell and asset links | `src/pages/index.html` |
| Navigation, footer, and page sections | `src/components/` |
| Shared and section styles | `src/styles/` |
| Language, navigation, workflow, and copy behavior | `src/scripts/` |
| Images served by the site | `assets/` |

`index.html` is generated from the page shell and components. Edit the files under `src/`, then rebuild:

```bash
python scripts/build.py
python scripts/build.py --check
```

Open `index.html` directly, or serve the branch root with `python -m http.server 8000`. Commit the generated `index.html` with its source changes. The CSS and JavaScript are served directly from `src/`, so they need no bundler or dependency installation.

The website links to the main branch for project documentation and to Hugging Face for released checkpoints. Benchmark values and command examples should be checked against the current main-branch README and result summary when updated.
