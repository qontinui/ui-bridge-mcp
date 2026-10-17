# Contributing

Thanks for your interest in contributing to qontinui. This document explains how to submit changes and what you're agreeing to when you do.

## License: MIT

This project is licensed under the **MIT License** (`MIT`). The full text is in [`LICENSE`](LICENSE).

## Developer Certificate of Origin (DCO)

Contributions are accepted under the **[Developer Certificate of Origin (DCO) 1.1](https://developercertificate.org)** — there is no Contributor License Agreement to sign. Certify that you wrote (or otherwise have the right to submit) your contribution by adding a `Signed-off-by` trailer to every commit:

```bash
git commit -s -m "your message"
```

`-s` appends `Signed-off-by: Your Name <your@email>` from your `git config user.name` / `user.email`. Your contributions are licensed under the same MIT License as the project (inbound = outbound), and you retain copyright in them.

## Code style

This repository follows the conventions documented in `CLAUDE.md` (project root). Highlights:

- Edit existing files in preference to creating new ones; keep changes scoped to the task.
- No comments unless the *why* is non-obvious. Don't narrate what well-named code already says.
- No backwards-compatibility shims unless explicitly requested.
- For Python: Black + ruff. For TypeScript/JavaScript: project's Prettier + ESLint config. For Rust: `cargo fmt` + `cargo clippy`.
- Run the test suite before opening a PR. CI will reject unformatted or lint-failing code.

## Submitting a change

1. Fork the repository and create a feature branch.
2. Make your change. Add tests where appropriate.
3. Run the local checks (lint, format, tests) — the per-repo `README` documents the exact commands.
4. Open a pull request against `main`. Describe the *why* in the PR body, not the *what* — the diff already shows the *what*.
5. Sign off every commit (`git commit -s`) — see the DCO section above.
6. A maintainer will review. Expect feedback; we keep the bar high because every change becomes part of the shipped product.

## Reporting bugs / requesting features

Open a GitHub issue. For security vulnerabilities, please email the maintainers directly rather than filing a public issue.

## Code of conduct

Be kind. Be specific. No harassment. Discussions stay on the technical merits.
