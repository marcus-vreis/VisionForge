# Security policy

VisionForge stores API tokens for dataset hubs and exposes a local HTTP API, so
security reports are taken seriously.

## Supported versions

Only the latest published release on PyPI (`visionforge-studio`) receives
security fixes. The `development` branch carries the next release.

## Reporting a vulnerability

**Do not open a public issue or discussion for a vulnerability.**

Use GitHub's private vulnerability reporting instead:

1. Open the [Security tab](https://github.com/marcus-vreis/VisionForge/security).
2. Click **Report a vulnerability** (direct link:
   <https://github.com/marcus-vreis/VisionForge/security/advisories/new>).
3. Describe the problem, the affected version and the steps to reproduce it.

Only the maintainer can read the report. Please do not include real tokens or
credentials; redact them.

## What to expect

- Acknowledgement of the report within 7 days.
- A first assessment (accepted, needs more information or declined) within 14
  days.
- A fix released as a patch version once confirmed, with credit in the advisory
  unless you ask to stay anonymous.

## Scope

In scope: the `visionforge` Python package, the local API served by
`visionforge gui`, credential storage and the published Docker image.

Out of scope: vulnerabilities in third-party dependencies that are not
reachable through VisionForge (report those upstream), and a GUI server that
you deliberately exposed to an untrusted network with `--host 0.0.0.0` (see
`docs/gui/tokens.md` for the exposure warning).
