#!/bin/bash
set -e

# Server setup script for Monies application
# Run this script on the server to configure Nginx and systemd

echo "Setting up Monies application on server..."

# Create directories
echo "Creating application directory..."
sudo mkdir -p /var/www/html/monies

# Install dependencies if not already installed
echo "Checking/installing system dependencies..."
if ! command -v nginx &> /dev/null; then
    echo "Installing Nginx..."
    sudo apt-get update
    sudo apt-get install -y nginx
fi

if ! command -v python3 &> /dev/null; then
    echo "Installing Python3..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip
fi

if ! command -v streamlit &> /dev/null; then
    echo "Installing Streamlit..."
    sudo pip3 install streamlit
fi

# Setup Nginx configuration
echo "Setting up Nginx configuration..."
NGINX_AVAILABLE="/etc/nginx/sites-available"
NGINX_ENABLED="/etc/nginx/sites-enabled"

# Check if we need to create a new site or update existing one
if [ -f "$NGINX_AVAILABLE/default" ]; then
    echo "Adding Monies configuration to the default site..."
    # Check if monies config is already included
    if ! grep -q "include /var/www/html/monies/nginx-monies.conf;" "$NGINX_AVAILABLE/default"; then
        # Add include directive before the closing bracket of the server block
        sudo sed -i '/server {/,/}/{s/}/    include \/var\/www\/html\/monies\/nginx-monies.conf;\n}/}' "$NGINX_AVAILABLE/default"
    fi
else
    echo "Creating new Nginx site configuration..."
    cat > /tmp/monies-site << 'EOF'
server {
    listen 80;
    server_name jrgochan.io;

    root /var/www/html;
    index index.html;

    include /var/www/html/monies/nginx-monies.conf;

    location / {
        try_files $uri $uri/ =404;
    }
}
EOF
    sudo mv /tmp/monies-site "$NGINX_AVAILABLE/monies"
    sudo ln -sf "$NGINX_AVAILABLE/monies" "$NGINX_ENABLED/monies"
fi

# Copy Nginx config file
echo "Copying Nginx configuration file..."
sudo cp ./nginx-monies.conf /var/www/html/monies/

# Setup systemd service
echo "Setting up systemd service..."
sudo cp ./monies.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable monies.service

echo "Restarting Nginx..."
sudo systemctl restart nginx

echo "Setup complete!"
echo "The application will be available at: http://jrgochan.io/monies"
echo "Start the service with: sudo systemctl start monies.service"
echo ""
echo "IMPORTANT: Make sure to copy your application files to /var/www/html/monies"
echo "and install the Python requirements before starting the service."
