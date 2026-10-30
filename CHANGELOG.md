# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-25

### Added

- `PlaceholderGuardMiddleware`: detects unfilled placeholders and forbidden terms
  in final text and tool arguments, with `retry`, `block` and `flag` strategies.
- `find_placeholders`, `Finding`, `PlaceholderLeakError`,
  `DEFAULT_PLACEHOLDER_PATTERNS`, `OPTIONAL_PATTERNS`.
