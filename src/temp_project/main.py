from temp_project.rag import ans

def main():
    print("Please enter your question")
    query = input("Question: ")
    answer = ans(query)
    print(f"Answer: \n{answer}")

if __name__ == "__main__":
    main()