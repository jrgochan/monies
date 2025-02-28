# Test Framework for Monies Application

This directory contains the test suite for the Monies crypto wallet and trend analysis application.

## Test Structure

- `unit/` - Unit tests for individual components and functions
  - Database models and operations
  - Authentication and security utilities
  - API configuration and testing
  - Data processing functions
  
- `integration/` - Tests for component interactions
  - Exchange API integration
  - AI analysis services
  - Data retrieval from external sources
  - Social media API integration
  
- `functional/` - Tests for complete features and pages
  - Dashboard functionality
  - Wallet management
  - Trading features
  - Settings page
  
- `fixtures/` - Shared test fixtures and data
  - Mock data providers
  - Test database setup
  - Authentication fixtures

## Test Coverage

The test suite aims for high coverage of critical components:
- Core business logic: 90%+
- UI components: 70%+
- API integrations: 80%+
- Error handling: 85%+

## Key Testing Principles

1. **Never use simulated data**: All tests verify that when APIs are unreachable, the application shows an error instead of using simulated data
2. **Proper mocking**: External APIs are properly mocked to avoid dependencies
3. **Error handling**: Tests cover both success and error paths
4. **Streamlit components**: Mock Streamlit components to test UI functionality

## Running Tests

Run all tests:
```bash
pytest
```

Run tests with coverage report:
```bash
pytest --cov=src
```

Generate HTML coverage report:
```bash
pytest --cov=src --cov-report=html
```

Run a specific test file:
```bash
pytest tests/unit/test_auth.py
```

Run a specific test:
```bash
pytest tests/unit/test_auth.py::test_password_hashing
```

## Test Markers

You can run tests by category using markers:

```bash
pytest -m "unit"  # Run all unit tests
pytest -m "integration"  # Run all integration tests
pytest -m "functional"  # Run all functional tests
pytest -m "api"  # Run all API endpoint tests
pytest -m "slow"  # Run tests that take longer to complete
pytest -m "not api_connection"  # Run tests that don't require internet
```

## Environment

Tests use an in-memory SQLite database and mock external services (Binance API, Yahoo Finance, OpenAI API, etc.) to avoid external dependencies.

## Testing Network Issues

The test suite includes special tests for API connection issues:
- Missing credentials
- Network connectivity problems
- Rate limiting
- Malformed responses

This ensures the application gracefully handles connectivity problems and displays appropriate error messages rather than using simulated data.

## Adding New Tests

1. Identify the appropriate test category (unit, integration, functional)
2. Create a new test file following the naming convention `test_*.py`
3. Add test functions using the naming convention `test_*`
4. Use fixtures from `conftest.py` as needed
5. Mock external dependencies
6. Add appropriate markers to categorize your tests
7. Run tests with coverage to verify