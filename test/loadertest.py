from ingestion.loader import DocumentLoader

loader = DocumentLoader("data/raw")

documents = loader.load()

for doc in documents:
    print(doc)
