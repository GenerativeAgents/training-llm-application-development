.PHONY: notebook
notebook:
	uv run jupyter notebook --port 8080 --no-browser

.PHONY: run
run:
	uv run streamlit run app.py --server.port 8080

IPYNBS = $(shell ls notebooks/*.ipynb)

.PHONY: test
test:
	uv run jupyter nbconvert --inplace --execute $(IPYNBS)

.PHONY: clean
clean:
	uv run jupyter nbconvert --inplace --clear-output $(IPYNBS)
