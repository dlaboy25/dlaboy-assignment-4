install:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

run:
	. venv/bin/activate && FLASK_APP=app.py flask run --host=127.0.0.1 --port=3000
