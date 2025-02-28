#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print with timestamp
log() {
  echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Print success message
success() {
  log "${GREEN}✅ $1${NC}"
}

# Print warning message
warning() {
  log "${YELLOW}⚠️ $1${NC}"
}

# Print error message
error() {
  log "${RED}❌ $1${NC}"
}

# Function to check if a command exists
command_exists() {
  command -v "$1" &> /dev/null
}

# Check Python version
check_python_version() {
  log "Checking Python version..."
  
  if ! command_exists python3; then
    error "Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
  fi
  
  python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
  log "Found Python version: $python_version"
  
  if command -v bc &> /dev/null; then  
    if [ "$(echo "$python_version < 3.9" | bc)" -eq 1 ]; then
      warning "Python version $python_version detected. Recommended version is 3.9 or higher."
      # Force continue for automation
      log "Automatically continuing with available Python version"
    else
      success "Python version $python_version is compatible"
    fi
  else
    # Fall back to basic version check if bc is not available
    major=$(echo "$python_version" | cut -d. -f1)
    minor=$(echo "$python_version" | cut -d. -f2)
    
    if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 9 ]); then
      warning "Python version $python_version detected. Recommended version is 3.9 or higher."
      # Force continue for automation
      log "Automatically continuing with available Python version"
    else
      success "Python version $python_version is compatible"
    fi
  fi
}

# Setup virtual environment
setup_venv() {
  log "Setting up virtual environment..."
  
  if [ -d ".venv" ]; then
    warning "Virtual environment already exists."
    # For automation, use existing venv
    log "Using existing virtual environment."
    return
  fi
  
  if ! command_exists python3 -m venv; then
    error "Python venv module is not available. Please install python3-venv package."
    exit 1
  fi
  
  python3 -m venv .venv
  success "Virtual environment created in .venv directory"
}

# Activate virtual environment
activate_venv() {
  log "Activating virtual environment..."
  
  if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    success "Virtual environment activated"
  else
    error "Virtual environment activation script not found. Setup may be incomplete."
    exit 1
  fi
  
  # Upgrade pip
  log "Upgrading pip..."
  pip install --upgrade pip
  success "Pip upgraded to latest version"
}

# Install dependencies
install_dependencies() {
  log "Installing dependencies..."
  
  if [ ! -f "requirements.txt" ]; then
    error "requirements.txt not found. Are you in the right directory?"
    exit 1
  fi
  
  pip install -r requirements.txt
  success "Dependencies installed successfully"
  
  if [[ "$1" == "dev" ]]; then
    log "Installing development dependencies..."
    # Install development dependencies if they exist
    if [ -f "requirements-dev.txt" ]; then
      pip install -r requirements-dev.txt
      success "Development dependencies installed successfully"
      
      # Install pre-commit hooks
      if command_exists pre-commit; then
        log "Installing pre-commit hooks..."
        pre-commit install
        success "Pre-commit hooks installed successfully"
      else
        warning "pre-commit command not found. Skipping hook installation."
      fi
    else
      log "No development dependencies file found, installing common development packages..."
      pip install pytest pytest-cov black flake8 mypy isort pre-commit
      success "Common development packages installed"
      
      # Install pre-commit hooks
      log "Installing pre-commit hooks..."
      pre-commit install
      success "Pre-commit hooks installed successfully"
    fi
  fi
}

# Setup environment
setup_env() {
  log "Setting up environment variables..."
  
  if [ ! -f ".env.example" ] && [ ! -f ".env" ]; then
    warning ".env.example and .env not found. Continuing without environment file."
    # Create a basic .env file for testing
    touch .env
    success "Created empty .env file for testing"
  elif [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    success "Created .env file from .env.example"
  else
    warning ".env file already exists. Using existing file."
  fi
}

# Initialize database
initialize_database() {
  log "Initializing database..."
  
  if [ ! -f "init_db.py" ]; then
    error "init_db.py not found. Are you in the right directory?"
    exit 1
  fi
  
  python init_db.py
  success "Database initialized successfully"
}

# Create convenience scripts
create_convenience_scripts() {
  log "Creating convenience scripts..."
  
  # Create run script
  cat > scripts/run.sh << 'EOL'
#!/bin/bash
set -e

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Virtual environment not found. Run setup.sh first."
  exit 1
fi

# Run the application
streamlit run app.py "$@"
EOL
  chmod +x scripts/run.sh
  
  # Create test script
  cat > scripts/test.sh << 'EOL'
#!/bin/bash
set -e

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Virtual environment not found. Run setup.sh first."
  exit 1
fi

# Run tests
pytest "$@"
EOL
  chmod +x scripts/test.sh
  
  # Create lint script
  cat > scripts/lint.sh << 'EOL'
#!/bin/bash
set -e

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Virtual environment not found. Run setup.sh first."
  exit 1
fi

# Run linters
echo "Running flake8..."
flake8 src app.py
echo "Running mypy..."
mypy src app.py
echo "Running black (check only)..."
black --check src app.py
EOL
  chmod +x scripts/lint.sh
  
  # Create format script
  cat > scripts/format.sh << 'EOL'
#!/bin/bash
set -e

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Virtual environment not found. Run setup.sh first."
  exit 1
fi

# Run black formatter
black src app.py
EOL
  chmod +x scripts/format.sh
  
  success "Convenience scripts created in scripts directory"
}

# Main setup function
main() {
  log "Starting setup for Crypto Wallet & Trend Analysis application..."
  
  # Check for development mode
  DEV_MODE=0
  if [[ "$1" == "dev" ]]; then
    DEV_MODE=1
    log "Setting up in development mode"
  else
    log "Setting up in standard mode (use './setup.sh dev' for development mode)"
  fi
  
  # Run setup steps
  check_python_version
  setup_venv
  activate_venv
  
  if [[ $DEV_MODE -eq 1 ]]; then
    install_dependencies "dev"
  else
    install_dependencies
  fi
  
  setup_env
  initialize_database
  create_convenience_scripts
  
  log ""
  success "Setup completed successfully!"
  log ""
  log "To activate the virtual environment in the future, run:"
  log "  source .venv/bin/activate"
  log ""
  log "To run the application:"
  log "  ./scripts/run.sh"
  log ""
  if [[ $DEV_MODE -eq 1 ]]; then
    log "Development tools:"
    log "  Run tests:        ./scripts/test.sh"
    log "  Lint code:        ./scripts/lint.sh"
    log "  Format code:      ./scripts/format.sh"
    log ""
  fi
  log "Remember to edit the .env file with your actual API keys and configuration."
}

# Run the script
main "$@"