# Interactive Machine Learning Model Builder
[Fall 24 CS 35L Final Project] Interactive Machine Learning Model Builder | Anshul Aravind, Venkat Chitturi, Satyajit Kumar, Noah Cylich, Shreya Shirsathe, Riti Patel

## Installation Instructions
```bash
bash backend/database/setup_postgres.sh  # Install PostgreSQL with the necessary user information
pip install -r requirements.txt  # Install Python dependencies
cd frontend && npm install && cd ..  # Install Node.js dependencies
```
## Running the Application
```bash
# Move to project dir, InteractiveMlBuilder
python backend/flask/server.py # Run the Flask server

# Open a new terminal in same, project dir, InteractiveMlBuilder
brew services start redis #start redis server
cd flask 
celery -A server.celery worker --loglevel=info # Run the Celery worker

# Open 1 more terminal in same, project dir, InteractiveMlBuilder
cd frontend
npm start # Run the React server
```