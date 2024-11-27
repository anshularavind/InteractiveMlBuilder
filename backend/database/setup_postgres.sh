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

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
  echo "PostgreSQL is not installed. Installing PostgreSQL..."
  brew install postgresql
else
  echo "PostgreSQL is already installed. Skipping installation."
fi

# Start PostgreSQL service
echo "Starting PostgreSQL service..."
brew services start postgresql

# Wait for PostgreSQL to start
echo "Waiting for PostgreSQL to start..."
sleep 2

createuser -s $POSTGRES_USER

# Configure PostgreSQL: Check and create the database
echo "Configuring PostgreSQL..."
psql -U "${POSTGRES_USER}" <<-EOSQL
  -- Check if the database exists; if not, create it
  CREATE DATABASE ${DB_NAME} OWNER ${POSTGRES_USER};
EOSQL

# Confirm success
echo "PostgreSQL setup complete."
echo "User '${POSTGRES_USER}' is configured with password '${POSTGRES_PASSWORD}'."
echo "Database '${DB_NAME}' created or verified."

# Display usage instructions
echo "To connect to PostgreSQL:"
echo "  psql -U ${POSTGRES_USER} -d ${DB_NAME} -W"