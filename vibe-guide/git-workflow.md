# Git Workflow & Commit Procedures

## Overview
This document outlines the Git workflow and commit procedures for the Javanese Hate Speech Detection project to ensure code quality, traceability, and team collaboration.

## Branch Strategy

### Main Branches
- **`main`**: Production-ready code
- **`develop`**: Integration branch for features
- **`feature/*`**: Feature development branches
- **`hotfix/*`**: Critical bug fixes
- **`release/*`**: Release preparation branches

### Branch Naming Convention
```
feature/api-endpoint-optimization
feature/model-performance-improvement
hotfix/critical-security-patch
release/v1.2.0
```

## Commit Message Format

### Structure
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks
- **perf**: Performance improvements

### Examples
```
feat(api): add batch prediction endpoint

Implement batch processing capability for multiple text inputs
to improve API efficiency for bulk operations.

- Add /batch-predict endpoint
- Implement input validation for batch requests
- Add comprehensive error handling
- Update API documentation

Closes #123
```

```
test(api): add unit tests for FastAPI endpoints

Add comprehensive unit test coverage for all API endpoints
including error handling and edge cases.

- Test all HTTP methods and status codes
- Mock external dependencies
- Validate response formats
- Test error scenarios

Reviewed-by: @backend-dev
```

## Pre-Commit Checklist

### Code Quality
- [ ] Code follows project style guidelines
- [ ] No debugging code or console.log statements
- [ ] No hardcoded secrets or API keys
- [ ] Proper error handling implemented
- [ ] Code is properly commented

### Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Test coverage maintained or improved
- [ ] Integration tests updated if needed

### Documentation
- [ ] README updated if needed
- [ ] API documentation updated
- [ ] Code comments added for complex logic
- [ ] CHANGELOG updated for significant changes

### Security
- [ ] No sensitive data in commit
- [ ] Dependencies are up to date
- [ ] Security best practices followed

## Workflow Steps

### 1. Feature Development
```bash
# Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat(scope): description"

# Push feature branch
git push origin feature/your-feature-name
```

### 2. Testing Before Merge
```bash
# Run all tests
python -m pytest src/tests/ -v

# Run API tests specifically
python -m pytest src/tests/test_api_unit.py -v

# Run integration tests
python -m pytest src/tests/integration/ -v

# Check code style (if linter is configured)
flake8 src/
```

### 3. Pull Request Process
1. Create pull request from feature branch to develop
2. Fill out PR template with:
   - Description of changes
   - Testing performed
   - Breaking changes (if any)
   - Related issues
3. Request review from team members
4. Address review feedback
5. Ensure all CI checks pass
6. Merge after approval

### 4. Release Process
```bash
# Create release branch
git checkout develop
git pull origin develop
git checkout -b release/v1.2.0

# Update version numbers and CHANGELOG
# Run final tests
python -m pytest

# Merge to main
git checkout main
git merge release/v1.2.0
git tag v1.2.0
git push origin main --tags

# Merge back to develop
git checkout develop
git merge release/v1.2.0
git push origin develop
```

## Code Review Guidelines

### Reviewer Checklist
- [ ] Code logic is correct and efficient
- [ ] Tests are comprehensive and meaningful
- [ ] Documentation is clear and complete
- [ ] Security considerations addressed
- [ ] Performance impact considered
- [ ] Code style is consistent

### Review Comments
- Be constructive and specific
- Suggest improvements, not just problems
- Ask questions for clarification
- Acknowledge good practices

## Emergency Procedures

### Hotfix Process
```bash
# Create hotfix from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-fix

# Make minimal fix
git commit -m "hotfix: critical security patch"

# Test thoroughly
python -m pytest

# Merge to main and develop
git checkout main
git merge hotfix/critical-fix
git tag v1.2.1
git push origin main --tags

git checkout develop
git merge hotfix/critical-fix
git push origin develop
```

### Rollback Procedure
```bash
# Revert to previous version
git checkout main
git revert <commit-hash>
git push origin main

# Or reset to previous tag
git reset --hard v1.1.0
git push origin main --force-with-lease
```

## Tools and Automation

### Recommended Git Hooks
- **pre-commit**: Run tests and linting
- **commit-msg**: Validate commit message format
- **pre-push**: Run full test suite

### CI/CD Integration
- Automated testing on pull requests
- Code quality checks
- Security scanning
- Automated deployment to staging

## Team Responsibilities

### All Developers
- Follow commit message conventions
- Write meaningful commit messages
- Test code before committing
- Keep commits atomic and focused

### Code Reviewers
- Review within 24 hours
- Provide constructive feedback
- Ensure quality standards
- Approve only when confident

### Project Maintainers
- Manage release cycles
- Maintain branch protection rules
- Monitor code quality metrics
- Handle emergency situations

## Troubleshooting

### Common Issues
1. **Merge Conflicts**: Use `git mergetool` or manual resolution
2. **Failed Tests**: Fix issues before merging
3. **Large Commits**: Break into smaller, logical commits
4. **Missing Documentation**: Update docs before merging

### Getting Help
- Check project documentation
- Ask team members
- Refer to Git documentation
- Use project communication channels

---

**Last Updated**: December 2024  
**Version**: 1.0  
**Maintainer**: Backend Development Team