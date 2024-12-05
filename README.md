# Interactive Machine Learning Model Builder
[Fall 24 CS 35L Final Project] Interactive Machine Learning Model Builder | Anshul Aravind, Venkat Chitturi, Satyajit Kumar, Noah Cylich, Shreya Shirsathe, Riti Patel

## Installation Instructions
Make sure you have conda installed on your system. If not, you can install it from [here](https://docs.conda.io/projects/conda/en/latest/index.html). Make sure you also have brew as well.
```bash
bash backend/database/setup_postgres.sh  # Install PostgreSQL with the necessary user information
conda env create -f mlbuilder.yaml python=3.11.8  # Install Python dependencies
conda activate mlbuilder  # Activate the conda environment
cd frontend && npm install && cd ..  # Install Node.js dependencies
brew install redis  # Install Redis
```
## Running the Application
```bash
# Move to project dir, InteractiveMlBuilder
python backend/flask/server.py # Run the Flask server

# Open a new terminal in same, project dir, InteractiveMlBuilder
brew services start redis #start redis server
cd backend/flask 
celery -A server.celery worker --loglevel=info # Run the Celery worker

# Open 1 more terminal in same, project dir, InteractiveMlBuilder
cd frontend
npm start # Run the React server
```

## Stopping the Application
```bash
brew services stop redis # Stop the Redis server
brew services stop postgresql # Stop the PostgreSQL server
```