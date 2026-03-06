# End-to-End Book Recommender System

A collaborative filtering book recommendation system built with KNN on the Book-Crossing dataset. Features a FastAPI backend and a React (Vite) frontend.

## Project Structure

```
book-recommendation-system-new/
├── backend/
│   ├── api.py                  # FastAPI server
│   ├── books_recommender/      # ML pipeline package
│   │   ├── components/         # Pipeline stages
│   │   ├── config/
│   │   ├── pipeline/
│   │   └── ...
│   ├── artifacts/              # Generated model & data files
│   ├── config/config.yaml
│   └── logs/
├── frontend/                   # React (Vite) app
│   ├── src/
│   └── Dockerfile.frontend
├── main.py                     # Run training pipeline directly
├── requirements.txt
├── Dockerfile
└── docker-compose.yaml
```

## Pipeline Workflow

`config.yaml` → `entity` → `config/configuration.py` → `components` → `pipeline` → `backend/api.py` → `frontend/`

---

## How to Run (Local)

### Step 1 — Clone the repository

```bash
git clone https://github.com/VIVPM/book-recommendation-system-new.git
cd book-recommendation-system-new
```

### Step 2 — Create and activate a conda environment

```bash
conda create -n books python=3.10 -y
conda activate books
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the FastAPI backend

```bash
cd backend
uvicorn api:app --reload --port 8000
```

API available at: `http://localhost:8000`  
Swagger docs at: `http://localhost:8000/docs`

### Step 5 — Run the React frontend (new terminal)

```bash
cd frontend
npm install
npm run dev
```

Frontend available at: `http://localhost:5173`

### (Optional) Train the model via CLI

```bash
# From project root
python main.py
```

---

## How to Run (Docker)

### Step 1 — Install Docker and Docker Compose

```bash
sudo apt-get update -y && sudo apt-get upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

sudo usermod -aG docker $USER
newgrp docker
```

### Step 2 — Clone and build

```bash
git clone https://github.com/VIVPM/book-recommendation-system-new.git
cd book-recommendation-system-new

docker-compose up --build        # first time (builds images + starts)
# or
docker-compose up -d             # run in background (detached)
```

| Service  | URL |
|----------|-----|
| Backend  | http://localhost:8000 |
| Frontend | http://localhost:5173 |

### Run containers individually (without docker-compose)

**Backend:**
```bash
# Build
docker build -t VIVPM/bookapp-backend:latest ./backend

# Run
docker run -d -p 8000:8000 VIVPM/bookapp-backend:latest
```

**Frontend:**
```bash
# Build
docker build -t VIVPM/bookapp-frontend:latest ./frontend

# Run
docker run -d -p 5173:5173 VIVPM/bookapp-frontend:latest
```

### Useful Docker commands

```bash
# View running containers
docker ps

# Stop all services
docker-compose down

# Push backend image
docker login
docker build -t VIVPM/bookapp:latest .
docker push VIVPM/bookapp:latest

# Pull image
docker pull VIVPM/bookapp:latest
```

---

## AWS EC2 Deployment

> Port mapping required: **8000** (backend) and **5173** (frontend)

```bash
sudo apt-get update -y && sudo apt-get upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

git clone https://github.com/VIVPM/book-recommendation-system-new.git
cd book-recommendation-system-new

docker-compose up --build -d
```

---

## Model Evaluation

The collaborative filtering model uses **K-Nearest Neighbors (KNN)** with **Cosine Similarity** to overcome the extreme sparsity of the Book-Crossing dataset.

We test the model using **Leave-One-Out Offline Evaluation**. For a random sample of users, we take one book they highly rated (7/10 or higher), ask the model for 5 recommendations, and check if the model successfully recommends *other* books that the same user also highly rated.

### Evaluation Results (Top-5 Recommendations)
* **Hit Ratio @ 5:** `23.40%` — For over 23% of users, the model successfully guesses at least one of their other favorite books.
* **Precision @ 5:** `8.64%` — Out of the 5 books recommended, an average of ~9% are exact ground-truth favorites.
* **Recall @ 5:** `6.48%` — Our Top-5 list successfully catches over 6% of all the books a user has ever loved.

*(Note: In recommendation systems with sparse data, true Precision/Recall is often much higher than measured, as users haven't read/rated most of the good recommendations yet — known as the False Negative problem).*

Run the offline evaluation yourself:
```bash
python backend/evaluate.py
```