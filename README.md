# Multilingual Question Answering

A multilingual question answering project built by fine-tuning `xlm-roberta-base` on the TyDi QA dataset. The project includes the original training notebook, a lightweight evaluation notebook with `Exact Match (EM)` and `F1`, and a small Gradio demo for interactive testing.

## Project Goal

This project was built to strengthen hands-on understanding of:

- multilingual NLP
- extractive question answering
- transformer fine-tuning
- model evaluation with EM and F1
- simple model demo deployment with Gradio

The current goal is educational and portfolio-focused rather than production deployment.

## What The Model Does

Given:

- a context paragraph
- a question in the same or another supported language

the model extracts the answer span directly from the context.

## Project Files

- [Multilingual Q&A.ipynb](./Multilingual%20Q%26A.ipynb): original training and experimentation notebook
- [Multilingual_QA_Evaluation_Showcase.ipynb](./Multilingual_QA_Evaluation_Showcase.ipynb): cleaner evaluation notebook with EM/F1 and language-level analysis
- [gradio_multilingual_qa_app.py](./gradio_multilingual_qa_app.py): simple web interface for testing the trained checkpoint

## Dataset

This project uses the `google-research-datasets/tydiqa` dataset from Hugging Face.

TyDi QA is useful for multilingual extractive QA because it contains examples from multiple languages and helps evaluate cross-lingual behavior.

## Model

- Base model: `xlm-roberta-base`
- Task: Extractive Question Answering
- Framework: Hugging Face Transformers

## Evaluation

The evaluation notebook computes:

- `Exact Match (EM)`: checks whether the predicted answer exactly matches the ground-truth answer after normalization
- `F1 Score`: measures token overlap between prediction and ground truth

In my lightweight evaluation sample, the model reached approximately:

- `F1: 71`
- `Arabic EM: 51%`

These results show that the model has learned useful multilingual QA behavior, while still having room for improvement with additional training and better preprocessing.

## Demo

The project includes a Gradio interface where you can:

- paste a context paragraph
- ask a question
- see the extracted answer

### Run the demo

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python gradio_multilingual_qa_app.py
```

By default, the app looks for the checkpoint in:

```bash
./checkpoint-7500
```

If your model is stored elsewhere, set the `MODEL_PATH` environment variable before running:

```powershell
$env:MODEL_PATH="C:\path\to\checkpoint-7500"
python gradio_multilingual_qa_app.py
```

## Example Questions

### English

Context:

```text
Albert Einstein was born in Germany in 1879. He later developed the theory of relativity.
```

Question:

```text
Where was Albert Einstein born?
```

### Arabic

Context:

```text
القاهرة هي عاصمة مصر، وتُعد من أكبر المدن في العالم العربي.
```

Question:

```text
ما هي عاصمة مصر؟
```

### German

Context:

```text
Die Berliner Mauer fiel im Jahr 1989. Dieses Ereignis war ein wichtiger Moment in der deutschen Geschichte.
```

Question:

```text
Wann fiel die Berliner Mauer?
```

## Limitations

- the model was not trained to full completion
- performance varies across languages
- the demo is intended for showcasing and experimentation
- results depend on answer span quality and context clarity

## Future Improvements

- train for more steps or more complete epochs
- improve preprocessing and answer span alignment
- evaluate on a larger held-out split
- compare checkpoints and keep the best one automatically
- deploy the demo publicly

## GitHub Upload Tips

- do not upload large checkpoints directly to GitHub if they exceed GitHub file size limits
- keep the code, notebooks, screenshots, and README in the repo
- store large model files separately if needed
- add screenshots or a short demo GIF/video to make the repo stronger

## Author

Built by Hatem as part of a multilingual NLP learning and portfolio project.
