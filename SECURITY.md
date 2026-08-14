# Security Policy

## Supported versions

The `main` branch is the only supported line.

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Email the maintainer via GitHub: [@zoryktom](https://github.com/zoryktom)

Include:

- a description of the issue
- steps to reproduce
- impact (RCE, model theft, prompt/data leak, etc.)

## Scope notes

- Do not use VisionGuard as the sole interlock on industrial equipment.
- Model weights downloaded from the internet should be treated as untrusted binaries.
- The HTTP API is intended for trusted local / VPC deployments unless you put it behind auth.
