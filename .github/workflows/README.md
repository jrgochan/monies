# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automating various tasks in the Monies project.

## Available Workflows

1. **Python Application CI** (`python-app.yml`): Builds and tests the application on multiple Python versions.
2. **Tests** (`test.yml`): Runs tests and uploads coverage reports.
3. **Deploy to jrgochan.io** (`deploy.yml`): Automates deployment to your server at jrgochan.io/monies.

## Setting Up Deployment

To enable the automatic deployment workflow to deploy to your server at jrgochan.io, follow these steps:

### 1. Create Required GitHub Secrets

In your GitHub repository, go to Settings > Secrets and variables > Actions, then add the following secrets:

- `SSH_PRIVATE_KEY`: Your private SSH key that has access to your server
- `SERVER_USER`: The username used to connect to your server

### 2. Initial Server Setup

Before the first deployment, you need to set up your server:

1. SSH into your server
2. Copy the server setup files from the repository:
   ```
   scp -r /path/to/local/monies/scripts/server_setup.sh \
           /path/to/local/monies/scripts/nginx-monies.conf \
           /path/to/local/monies/scripts/monies.service \
           user@jrgochan.io:/tmp/
   ```
3. SSH into your server and run the setup script:
   ```
   ssh user@jrgochan.io
   cd /tmp
   chmod +x server_setup.sh
   ./server_setup.sh
   ```

### 3. Deploy the Application

The workflow will automatically deploy when you push to the `main` branch. You can also manually trigger it from the "Actions" tab in your GitHub repository.

### Workflow Overview

The deployment workflow:

1. Checks out the code
2. Sets up Python and installs dependencies
3. Runs tests to ensure everything is working
4. Creates a deployment package with all required files
5. Securely connects to your server using SSH
6. Deploys the package to `/var/www/html/monies`
7. Restarts Nginx to apply changes

### Troubleshooting

If you encounter issues with the deployment:

1. Check the GitHub Actions logs for detailed error messages
2. Verify that your secrets are set correctly
3. Make sure your server is reachable and the SSH key has proper permissions
4. Check the server logs: `sudo journalctl -u monies -u nginx`
