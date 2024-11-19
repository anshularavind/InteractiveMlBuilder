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
## Check if the PostgreSQL superuser exists
#echo "Checking if role '${POSTGRES_USER}' exists..."
#if ! psql -U "${POSTGRES_USER}" -c "\q" 2>/dev/null; then
#  echo "Role '${POSTGRES_USER}' does not exist. Attempting to create it using the current OS user..."
#
#  # Use the current OS user to create the superuser
#  CURRENT_USER=$(whoami)
#  if ! psql -U "${CURRENT_USER}" -c "\q" 2>/dev/null; then
#    echo "Could not connect as '${CURRENT_USER}'. Ensure a valid PostgreSQL role exists or check the configuration."
#    exit 1
#  fi
#
#  echo "Creating role '${POSTGRES_USER}'..."
#  psql -U "${CURRENT_USER}" <<-EOSQL
#    CREATE ROLE ${POSTGRES_USER} WITH SUPERUSER CREATEDB CREATEROLE LOGIN PASSWORD '${POSTGRES_PASSWORD}';
#EOSQL
#else
#  echo "Role '${POSTGRES_USER}' already exists."
#fi
#
## Export environment variables for PostgreSQL
#export PGUSER=${POSTGRES_USER}
#export PGPASSWORD=${POSTGRES_PASSWORD}

# Configure PostgreSQL: Check and create the database
echo "Configuring PostgreSQL..."
psql -U "${POSTGRES_USER}" <<-EOSQL
  -- Check if the database exists; if not, create it
  DO \$\$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DB_NAME}') THEN
      CREATE DATABASE ${DB_NAME} OWNER ${POSTGRES_USER};
    END IF;
  END \$\$;
EOSQL

# Stop PostgreSQL service
brew services stop postgresql

# Confirm success
echo "PostgreSQL setup complete."
echo "User '${POSTGRES_USER}' is configured with password '${POSTGRES_PASSWORD}'."
echo "Database '${DB_NAME}' created or verified."

# Display usage instructions
echo "To connect to PostgreSQL:"
echo "  psql -U ${POSTGRES_USER} -d ${DB_NAME} -W"