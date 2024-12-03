.PHONY: jupyter
jupyter:
	uv run jupyter notebook --port 8080 --no-browser

.PHONY: cloud9_jupyter
cloud9_jupyter:
	uv run jupyter notebook --ip 0.0.0.0 --port 8080 --no-browser

.PHONY: streamlit
streamlit:
	uv run streamlit run app.py --server.port 8080

IPYNBS = $(shell ls notebooks/*.ipynb)

.PHONY: test
test:
	uv run jupyter nbconvert --inplace --execute $(IPYNBS)

.PHONY: clean
clean:
	uv run jupyter nbconvert --inplace --clear-output $(IPYNBS)
