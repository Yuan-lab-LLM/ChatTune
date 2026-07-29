# Security Policy

## Reporting a Vulnerability

Please report suspected security issues privately to the project maintainers instead of opening a public issue. Include:

- Affected component and version or commit.
- Reproduction steps or proof of concept.
- Impact and any known mitigations.
- Whether secrets, patient data, model artifacts, or infrastructure access may be exposed.

If this repository is hosted on GitHub, enable and use GitHub Security Advisories for private reporting. Until a public security contact is configured, contact the repository owner or organization maintainers through their private coordination channel.

## Supported Scope

Security fixes are prioritized for the current main branch and the latest public release. Runtime deployments should rotate any exposed tokens immediately and regenerate `runtime.env` with `bash runtime.sh init --force` when credentials may have leaked.

## Sensitive Data Rules

Do not publish:

- API keys, tokens, passwords, SSH keys, cookies, or private certificates.
- Real patient records, personal identifiers, or non-public clinical data.
- Internal IPs, private registry addresses, private model paths, or production database files.
- Runtime logs that may contain prompts, user content, paths, or credentials.
