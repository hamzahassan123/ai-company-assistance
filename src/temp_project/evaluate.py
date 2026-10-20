from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import evaluate


def main() -> None:
    from temp_project.config import require_gemini_api_key

    from temp_project.rag import ans

    judge = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=require_gemini_api_key(),
    )

    def target(inputs):
        return {"answer": ans(inputs["question"])}

    def correctness(run, example):
        question = example.inputs["question"]
        reference = example.outputs["answer"]
        prediction = run.outputs["answer"]

        prompt = f"""
You are evaluating a RAG system.

Determine whether the AI answer correctly answers the user's question
based on the reference answer.

Question:
{question}

Reference answer:
{reference}

AI answer:
{prediction}

Evaluation rules:
- Give 1 if the AI answer is substantively correct.
- Give 0 if it is incorrect, contradicts the reference, or fails to answer the question.
- Different wording is completely acceptable.
- Do not require the AI answer to match the reference word-for-word.

Return only:
1
or
0
"""

        response = judge.invoke(prompt)
        comment = str(response.content).strip()

        return {
            "key": "correctness",
            "score": comment == "1",
            "comment": comment,
        }

    evaluate(
        target,
        data="cogent-mind-evaluation",
        evaluators=[correctness],
        experiment_prefix="cogent-mind-v2",
        description="Semantic correctness evaluation of Cogent-Mind RAG",
    )


if __name__ == "__main__":
    main()