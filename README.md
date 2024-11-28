# Interactive Machine Learning Model Builder
[Fall 24 CS 35L Final Project] Interactive Machine Learning Model Builder | Anshul Aravind, Venkat Chitturi, Satyajit Kumar, Noah Cylich, Shreya Shirsathe, Riti Patel

## Installation Instructions
```bash
bash backend/database/setup_postgres.sh  # Install PostgreSQL with the necessary user information
pip install -r requirements.txt  # Install Python dependencies
cd frontend && npm install && cd ..  # Install Node.js dependencies
```
## Running the Backend
```bash
cd backend
# Run in separate terminals
python ./flask/server.py # Run the Flask server
celery -A server.celery worker --loglevel=info # Run the Celery worker
```