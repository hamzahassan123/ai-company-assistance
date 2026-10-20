from temp_project.rag import ans


def main() -> None:
    """Run the interactive question-answering application."""
    print("Ask a question about the company knowledge base.")
    query = input("Question: ")
    if not query.strip():
        raise SystemExit("Question must not be empty.")

    answer = ans(query)
    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()