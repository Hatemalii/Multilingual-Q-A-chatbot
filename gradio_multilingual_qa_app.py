import os

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
import gradio as gr


# Set MODEL_PATH in your environment if your checkpoint is stored elsewhere.
MODEL_PATH = os.getenv("MODEL_PATH", "./checkpoint-7500")


device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device=device
)


def answer_question(context, question):
    if not context.strip():
        return "Please enter a context."
    if not question.strip():
        return "Please enter a question."

    result = qa_pipeline(question=question, context=context)
    answer = result["answer"]

    return answer


examples = [
    [
        "Mosasaurs were discovered in a limestone quarry at Maastricht on the Meuse. They lived during the late Cretaceous period.",
        "Where were the fossils discovered?"
    ],
    [
        "القاهرة هي عاصمة مصر. يبلغ عدد سكانها أكثر من 20 مليون نسمة.",
        "ما هي عاصمة مصر؟"
    ],
    [
        "Paris is the capital of France. القاهرة هي عاصمة مصر. Berlin is in Germany.",
        "What is the capital of France?"
    ],
    [
        "Paris is the capital of France. القاهرة هي عاصمة مصر. Berlin is in Germany.",
        "ما هي عاصمة مصر؟"
    ],
]


with gr.Blocks(title="Multilingual QA Demo") as demo:
    gr.Markdown(
        """
        # Multilingual Question Answering Demo
        Ask a question in one language and let the model extract the answer from the context.
        """
    )

    with gr.Row():
        with gr.Column():
            context = gr.Textbox(
                label="Context",
                lines=10,
                placeholder="Paste a paragraph here..."
            )
            question = gr.Textbox(
                label="Question",
                lines=2,
                placeholder="Ask your question in English, Arabic, or another language..."
            )
            submit_btn = gr.Button("Get Answer")

        with gr.Column():
            answer = gr.Textbox(label="Predicted Answer")

    submit_btn.click(
        fn=answer_question,
        inputs=[context, question],
        outputs=answer
    )

    gr.Examples(
        examples=examples,
        inputs=[context, question]
    )


if __name__ == "__main__":
    demo.launch()
