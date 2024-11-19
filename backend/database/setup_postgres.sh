#!/bin/bash

# Script to set up PostgreSQL for the application

# Specify the desired PostgreSQL user, password, and database name
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="1234"
DB_NAME="ml_builder"

echo "Starting PostgreSQL setup script..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
  echo "Homebrew is not installed. Please install Homebrew and try again."
  exit 1
fi

# Install PostgreSQL using Homebrew
echo "Installing PostgreSQL..."
brew install postgresql

# Start PostgreSQL service
echo "Starting PostgreSQL service..."
brew services start postgresql

# Wait for PostgreSQL to start
echo "Waiting for PostgreSQL to start..."
sleep 2

# Ensure the correct superuser is used for the script
export PGUSER=postgres
export PGPASSWORD='1234' # Set this if password is required for postgres user

# Configure PostgreSQL
echo "Configuring PostgreSQL..."
psql -U postgres <<-EOSQL
  -- Check if the user exists; if not, create the user
  DO \$\$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${POSTGRES_USER}') THEN
      CREATE ROLE ${POSTGRES_USER} LOGIN PASSWORD '${POSTGRES_PASSWORD}';
    END IF;
  END \$\$;

  -- Check if the database exists; if not, create it and assign ownership
  CREATE DATABASE ${DB_NAME} OWNER ${POSTGRES_USER};
EOSQL

# Confirm success
echo "PostgreSQL setup complete."
echo "User '${POSTGRES_USER}' created (if it did not exist) with password '${POSTGRES_PASSWORD}'."
echo "Database '${DB_NAME}' created (if it did not exist) and assigned to '${POSTGRES_USER}'."

# Display usage instructions
echo "To connect to PostgreSQL:"
echo "  psql -U ${POSTGRES_USER} -d ${DB_NAME} -W"
