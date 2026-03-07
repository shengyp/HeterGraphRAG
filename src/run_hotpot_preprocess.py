import json
from hotpot_utils import dataset_to_documents, dataset_to_chunks


def main():
    input_path = r"F:\资料\农业rag\KSEM\data\hotpotqa\hotpotqa.json"
    docs_output = r"F:\资料\农业rag\KSEM\data\hotpotqa\hotpot_docs.json"
    chunks_output = r"F:\资料\农业rag\KSEM\data\hotpotqa\hotpot_chunks.json"

    print("Loading dataset...")
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} samples")

    # 转 documents
    print("Converting to documents...")
    documents = dataset_to_documents(dataset)

    print(f"Generated {len(documents)} documents")

    with open(docs_output, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    # 转 chunks
    print("Converting to chunks...")
    chunks = dataset_to_chunks(dataset)

    print(f"Generated {len(chunks)} chunks")

    with open(chunks_output, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()