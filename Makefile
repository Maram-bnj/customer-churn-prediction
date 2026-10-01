install:
	pip install -r requirements.txt
	pip install -e .

train:
	python -m churn.train

evaluate:
	python -m churn.evaluate

app:
	streamlit run app/streamlit_app.py

test:
	pytest

.PHONY: install train evaluate app test
