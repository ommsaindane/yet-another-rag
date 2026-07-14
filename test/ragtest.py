from agent.builder import build_rag

graph = build_rag()


def main():

    graph = build_rag()

    print()

    while True:
        query = input("Question (or 'exit'): ")

        if query.lower() == "exit":
            break

        result = graph.invoke(query)

        print()
        print("=" * 80)
        print(result["answer"])
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()
