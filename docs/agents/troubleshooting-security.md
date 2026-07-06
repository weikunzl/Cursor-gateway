# Agent Reference: Troubleshooting, Security, and Performance

This file expands the operational constraints summarized in `AGENTS.md`.

## "Improperly formed request" Errors

Kiro API's `Improperly formed request` error is vague and may indicate many unrelated validation issues:

- invalid message role order.
- missing required fields.
- invalid tool schemas.
- tool-name length violations.
- malformed JSON.
- unsupported content types.
- authentication or permission issues.
- undocumented upstream constraints.

Debug these errors systematically. Do not guess from the error text alone.

Recommended approach:

1. Enable `DEBUG_MODE="errors"`.
2. Compare the failing request with the closest passing request.
3. Minimize the payload until the failure disappears.
4. Add a regression test for the identified validation quirk.
5. Fix only the API-level incompatibility, not user content.

## Tests Failing

```bash
pip install -r requirements.txt
pytest tests/unit/test_<module>.py::test_<name> -v -s
pytest -v
```

If a test attempts network access, fix the test by mocking the external service.

## Server Not Starting

Check common causes:

```bash
lsof -i :8000
python main.py --port 9000
```

Also verify required environment variables and credentials.

## Authentication Errors

Check the configured auth method:

```bash
ls -la ~/.aws/sso/cache/
echo "$KIRO_CREDS_FILE"
echo "$KIRO_CLI_DB_FILE"
```

Do not print or log token values. If direct refresh-token auth is used, verify presence without exposing the value.

## Security Rules

- Never log credentials, access tokens, refresh tokens, API keys, passwords, or raw authorization headers.
- Sanitize errors before returning them to clients.
- Validate request bodies with Pydantic models.
- Use HTTPS in production.
- Keep credential mounts read-only where possible.
- Treat debug logs as sensitive because they may include request metadata.

## Performance Rules

- Use connection pooling for non-streaming requests.
- Use per-request clients for streaming requests.
- Keep I/O async.
- Cache model metadata where appropriate.
- Stream large responses rather than buffering them.
- Avoid unnecessary request copying, JSON round-tripping, or deep cloning on hot paths.

## User-Facing Errors

Errors should tell users:

1. what failed.
2. whether retrying might help.
3. which configuration value or credential source to check.
4. whether debug logging can provide more context.

Avoid leaking implementation details, upstream raw errors, or stack traces.

## Common Investigation Checklist

1. Reproduce with the smallest failing request.
2. Check debug logs if enabled.
3. Inspect converter output before the upstream call.
4. Inspect upstream status and classified error.
5. Confirm retry behavior.
6. Add a focused regression test.
7. Run the relevant unit tests.
