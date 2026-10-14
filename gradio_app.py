"""Hugging Face Gradio entry point for the existing CRAG graph."""
import os
from functools import lru_cache
import gradio as gr

@lru_cache(maxsize=1)
def get_graph():
    from agent.builder import build_rag
    return build_rag()

def answer(question):
    if not question or not question.strip():
        return 'Enter a question.', ''
    try:
        graph = get_graph()
        state = graph.invoke(question.strip())
        if state.get('context_is_relevant'):
            sources = sorted({f"{p.metadata.get('filename', p.document_id)} — {p.metadata.get('section_title', 'Document')}" for p in state.get('relevant_parents', [])})
            kind = 'Local corpus'
        else:
            sources = sorted({w.url for w in state.get('web_results', []) if w.url})
            kind = 'Web fallback'
        return state.get('answer', 'No answer generated.'), kind + '\n\n' + '\n'.join(f'- {s}' for s in sources)
    except Exception as exc:
        return f'Query failed: {type(exc).__name__}: {exc}', ''

with gr.Blocks(title='Yet Another RAG') as demo:
    gr.Markdown('# Yet Another RAG\nCorrective RAG with hybrid retrieval and source attribution')
    question = gr.Textbox(label='Question', lines=3)
    submit = gr.Button('Ask', variant='primary')
    result = gr.Markdown(label='Answer')
    sources = gr.Markdown(label='Sources')
    submit.click(answer, inputs=question, outputs=[result, sources])
    question.submit(answer, inputs=question, outputs=[result, sources])

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=1).launch()
