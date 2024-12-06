# Interactive Machine Learning Model Builder
[Fall 24 CS 35L Final Project] Interactive Machine Learning Model Builder | Anshul Aravind, Venkat Chitturi, Satyajit Kumar, Noah Cylich, Shreya Shirsathe, Riti Patel

## Installation Instructions
Make sure you have conda installed on your system. If not, you can install it from [here](https://docs.conda.io/projects/conda/en/latest/index.html). Make sure you also have brew as well. Install node and npm from [here](https://nodejs.org/en/download/).

```bash
# Clone the repository into your local by running:
git clone https://github.com/anshularavind/InteractiveMlBuilder.git
cd InteractiveMlBuilder
```
In backend/flask/.env, replace Auth0 vars with your own setup's. If you're our TA just replace:
"<Auth0 Client Secret>" in "AUTH0_CLIENT_SECRET=<Auth0 Client Secret>" with the supplied Auth0 Client Secret. 
For everyone else, scroll to the bottom section, Custom Auth0.

```bash
bash backend/database/setup_postgres.sh  # Install PostgreSQL with the necessary user information
conda env create -f mlbuilder.yaml  # Install Python dependencies
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
And cancel the Celery worker, Flask server, and npm processes in their respective terminals.


## Custom Auth0
You'll need to replace all the Auth0 variables with your own setup's. Make sure your Auth0 setup has the following:
- Basic Auth0 Application with Credentials Application Authentication set to none.
- Allowed Callback URLs, Allowed Logout Urls, & Allowed Web Origins set to http://localhost:3000/callback
- Also make sure you have an Auth0 API setup, this is your Auth0_AUDIENCE variable.
- This API must have jsonwebtoken as a scope
- Go to triggers and modify post-login to include this intermediate code (with your own namespace):
```javascript
exports.onExecutePostLogin = async (event, api) => {
    const namespace = 'https://InteractiveMlApi/';
    api.accessToken.setCustomClaim(`${namespace}email`, event.user.email);
    api.accessToken.setCustomClaim(`${namespace}name`, event.user.name);
};
```