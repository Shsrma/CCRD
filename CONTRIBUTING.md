# Contributing to CCRD

Thank you for your interest in contributing to the Credit Card Fraud Detection System! This document provides guidelines for submitting issues, feature requests, and pull requests.

## Code of Conduct

- Be respectful and inclusive
- No harassment or discrimination
- Help others learn and grow

## Getting Started

### Fork & Clone

```bash
git clone https://github.com/your-username/CCRD.git
cd CCRD
```

### Setup Development Environment

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

## Making Changes

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Write Tests First (TDD)

```bash
# Create tests in backend/tests/
pytest tests/test_your_feature.py
```

### 3. Implement the Feature

- Follow PEP 8 style guide
- Use type hints
- Write docstrings
- Keep functions focused

### 4. Run Quality Checks

```bash
# Code formatting
black app tests
isort app tests

# Linting
flake8 app tests

# Type checking
mypy app

# Tests with coverage
pytest --cov=app
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add amazing feature"
# Use conventional commits: feat, fix, docs, test, refactor, style, chore
```

### 6. Push & Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a PR on GitHub with a clear description of your changes.

## Pull Request Guidelines

### PR Title Format

- ✅ `feat: add fraud alert history endpoint`
- ✅ `fix: correct JWT token validation bug`
- ✅ `docs: update API documentation`
- ❌ `Update code`
- ❌ `Fixed stuff`

### PR Description Template

```markdown
## Description
Briefly describe the changes.

## Motivation & Context
Why is this change needed? What problem does it solve?

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How to test the changes:
1. Step 1
2. Step 2

## Checklist
- [ ] Tests pass (`pytest`)
- [ ] Code is formatted (`black`, `isort`)
- [ ] No linting errors (`flake8`)
- [ ] Documentation updated
- [ ] Commit messages are clear
```

## Code Style Guidelines

### Python

- **Line length**: 100 characters
- **Formatter**: Black
- **Linter**: Flake8
- **Type hints**: Required for public APIs
- **Docstrings**: Google style

```python
def predict_fraud(
    features: list[float],
    threshold: float = 0.5
) -> tuple[int, float]:
    """
    Predict if a transaction is fraudulent.
    
    Args:
        features: Feature vector from ML model
        threshold: Fraud probability threshold
    
    Returns:
        Tuple of (prediction, probability)
    """
    # Implementation
    pass
```

### Commits

- Keep commits atomic (one feature/fix per commit)
- Use present tense: "add feature" not "added feature"
- Reference issues: "fix #123"

```bash
# Good
git commit -m "feat: add fraud alert API endpoint

- Implements GET /alerts endpoint
- Filters by status and date range
- Fixes #42"

# Bad
git commit -m "updated stuff"
```

## Testing Requirements

- **Minimum coverage**: 80%
- **Unit tests**: For all business logic
- **Integration tests**: For API endpoints
- **E2E tests**: For critical workflows

```python
# tests/test_auth.py
def test_login_success(client, test_user):
    """Test successful login returns JWT token."""
    response = client.post("/auth/login", json={...})
    assert response.status_code == 200
    assert "access_token" in response.json()
```

## Documentation

### README Updates

- Update if adding a new feature
- Keep examples current
- Explain complex features

### API Documentation

- Add/update docstrings to endpoints
- Include request/response examples
- Document error cases

### Commit Messages

- Reference issues
- Explain the "why"
- Keep it concise

## Review Process

1. **Automated Checks** (GitHub Actions)
   - Tests must pass
   - Code coverage must be 80%+
   - Linting must pass

2. **Code Review** (Maintainers)
   - Architecture review
   - Security review
   - Performance review

3. **Approval & Merge**
   - At least 2 approvals required
   - All feedback addressed
   - Branch must be up-to-date

## Reporting Issues

### Bug Report Template

```markdown
## Description
Clear description of the bug.

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- Python version: 3.10
- OS: Ubuntu 22.04
- FastAPI version: 0.104

## Screenshots
If applicable.
```

### Feature Request Template

```markdown
## Description
Clear description of the feature.

## Motivation
Why is this feature needed?

## Proposed Solution
How should it work?

## Alternatives Considered
Other approaches.
```

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy ORM](https://docs.sqlalchemy.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [pytest Documentation](https://docs.pytest.org/)
- [Python PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)

## Questions?

- Open a GitHub Discussion
- Email: [your.email@example.com](mailto:your.email@example.com)
- Read existing issues

---

Thank you for contributing! 🎉
