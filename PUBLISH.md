# Creating the GitHub repository

This cloud agent cannot create repos under `msr2903` (integration token lacks `createRepository`).

From your machine (with push access):

```bash
gh repo create msr2903/himotoki-split --public --source=. --remote=origin --push
```

Or create an empty repo on GitHub, then:

```bash
cd /path/to/himotoki-split
git remote add origin https://github.com/msr2903/himotoki-split.git
git push -u origin main
```
