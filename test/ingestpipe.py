from ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline("data/raw")

documents = pipeline.run()

for document in documents:
    print(document.metadata["filename"])
