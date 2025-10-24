# Security Policy

## Data Handling and Privacy

### Personal Information

This application processes text comments which may contain personally identifiable information (PII). Users are responsible for:

- **Data Minimization**: Only upload data necessary for classification
- **Consent**: Ensure proper consent before processing personal data
- **Anonymization**: Consider anonymizing data before upload when possible
- **Legal Compliance**: Comply with GDPR, CCPA, and other applicable data protection regulations

### Data Processing

**What we do:**

- Process text in-memory during classification
- Store temporary files only during active sessions
- Load models and tokenizers from local directories

**What we don't do:**

- Send data to external servers (all processing is local)
- Store uploaded files permanently
- Log the content of classified comments
- Share data with third parties

### Logging Policy

**What gets logged:**

- Application events (startup, shutdown, errors)
- File operations (upload, processing, download)
- Performance metrics (processing time, batch sizes)
- Error messages and stack traces

**What is NOT logged:**

- User text content or comments
- Predicted categories or labels
- File contents or data previews
- Usernames or personal identifiers

**Log Redaction**: While we avoid logging sensitive data, review log files (`app.log`, `logs/`) before sharing for troubleshooting.

### Model Security

**Model Integrity:**

- Models should be obtained from trusted sources (HuggingFace, official repositories)
- Verify model provenance before loading
- Consider scanning model files for malicious code if from untrusted sources

**Recommended practices:**

- Use official pre-trained models (e.g., `xlm-roberta-base`)
- Store models in secure directories with appropriate permissions
- Consider model file integrity checks (checksums) for production deployments

### Input Validation

The application implements multiple validation layers:

1. **File Upload Validation:**

   - File size limits (configurable in `config.yaml`)
   - Allowed file types: `.xlsx`, `.xls`, `.csv`
   - Maximum upload size: 100MB (default, adjustable)

2. **Data Validation:**

   - Column existence checks
   - Null value detection
   - Text length limits (max 512 tokens by default)

3. **Security Boundaries:**
   - No arbitrary code execution
   - No SQL or command injection vectors
   - Safe Excel/CSV parsing via pandas/openpyxl

### Dependency Security

**Current Status:**

- Dependencies defined in `pyproject.toml` (modern Python packaging standard)
- CI pipeline includes `pip-audit` for vulnerability scanning

**Recommendations:**

- Run `pip-audit` regularly: `pip install pip-audit && pip-audit`
- Keep dependencies updated: `pip install -U -e .`
- Review security advisories for PyTorch, Transformers, and Streamlit

### Deployment Security

**For Production Deployments:**

1. **Environment Variables:**

   - Do not hardcode secrets in `config.yaml`
   - Use environment variables for sensitive configuration
   - Example: `os.getenv('MODEL_PATH', default_path)`

2. **Access Control:**

   - Deploy behind authentication (e.g., Streamlit auth, reverse proxy)
   - Use HTTPS/TLS for web deployments
   - Implement rate limiting if internet-facing

3. **File System:**

   - Set appropriate file permissions (read-only models)
   - Use dedicated service accounts with minimal privileges
   - Isolate log directories and restrict access

4. **Monitoring:**
   - Monitor for unusual activity (large files, high request rates)
   - Set up alerting for errors and exceptions
   - Track resource usage (CPU, memory, disk)

### Incident Response

**If a security issue is discovered:**

1. **Do not disclose publicly** until a fix is available
2. Contact the maintainers privately via email
3. Provide details: vulnerability description, reproduction steps, impact assessment
4. Allow reasonable time for response and remediation

**Expected Response:**

- Acknowledgment within 48 hours
- Initial assessment within 1 week
- Fix and disclosure timeline based on severity

## Vulnerability Reporting

To report a security vulnerability, please email: **volmyrdunayev@gmail.com**

Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

## Security Best Practices for Users

1. **Data Protection:**

   - Review and sanitize data before upload
   - Use anonymized datasets for testing
   - Delete temporary files after classification

2. **Model Safety:**

   - Download models from official sources only
   - Verify model checksums when available
   - Scan custom models for malicious content

3. **Environment Security:**

   - Keep Python and dependencies updated
   - Use virtual environments (venv, conda)
   - Run with least-privilege user accounts

4. **Audit and Compliance:**
   - Review logs regularly for anomalies
   - Document data processing activities (GDPR Article 30)
   - Conduct periodic security assessments

## Updates and Changes

This security policy will be updated as the application evolves. Check the `CHANGELOG.md` for security-related changes in each release.

**Last Updated:** October 24, 2025  
**Version:** 2.1.0
