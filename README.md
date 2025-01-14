# AI Checker Backend

A Django REST API for processing PDF documents.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run database migrations:
```bash
python manage.py migrate
```

## Running the Project

1. Start the Django development server:
```bash
py# AI Checker Backend

A Django REST API for processing PDF documents.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run database migrations:
```bash
python manage.py migrate
```

## Running the Project

1. Start the Django development server:
```bash
python manage.py runserver
```

2. Access the API endpoints:
- Admin interface: http://localhost:8000/admin/
- API endpoints: http://localhost:8000/api/pdfs/

## API Endpoints

- `GET /api/pdfs/`: List all PDF documents
- `POST /api/pdfs/`: Upload a new PDF document
- `GET /api/pdfs/{id}/`: Retrieve a specific PDF document
