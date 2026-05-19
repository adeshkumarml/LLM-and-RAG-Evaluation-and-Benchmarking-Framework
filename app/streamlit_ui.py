import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from src.utils.utils import config_loader
import streamlit as st
import requests
import json

config = config_loader()

st.set_page_config(page_title = "RAG Evaluation Framework", page_icon = "🤖", layout = "wide")

st.title("RAG Evaluation and Benchmarking Framework")
st.markdown("""
            Evaluate Retrieval and Generation Quality Using: 
            - Exact Match (EM)
            - Semantic Similarity
            - Retrieval Metrics"
            - LLM-as-Judge Evaluation""")
st.markdown("How it works: Upload a corpus and evaluation dataset to benchmark baseline LLM responses and RAG-enhanced responses.")

st.subheader("Upload Evaluation Files")
col1, col2 = st.columns(2)
with col1:
    corpus_file = st.file_uploader("Upload Context Corpus (.txt)", type = ["txt"])
with col2:
    dataset_file = st.file_uploader("Upload Evaluation File (.json)", type = ["json"], help = "Evaluation file must contain question(s) and answer(s)")
    with st.expander("Expected dataset format"):
        st.code("""
                [
                    {
                        'question': 'What is the capital of India?',
                        'answer': ['New Delhi']
                    }
                ]""", language = "json")
    
run_eval = st.button("EVALUATE", use_container_width = True)
if run_eval:
    if corpus_file is None:
        st.error("Please upload corpus file!")
        st.stop
    if dataset_file is None:
        st.error("Please upload dataset file!")
        st.stop()
    st.success("File uploaded successfully!")

    with st.spinner("Please Wait... Running Evaluation..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/evaluate", 
                files = {
                    "corpus_file": corpus_file, 
                    "dataset_file": dataset_file,
                }, 
                timeout = 300)
            if response.status_code != 200:
                st.error(f"Evaluation failed: {response.text}")
                st.stop()
            results = response.json()
            st.success("Evaluation successful")
        except requests.exceptions.Timeout:
            st.error("Request timed-out!")    
        except requests.exceptions.ConnectionError:
            st.error("Couldn't connect to server!")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    aggregate_results = results["aggregate_results"]

    st.divider()

    st.subheader("Evaluation Summary")
    meta1, meta2, meta3, meta4, meta5 = st.columns(5)
    meta1.metric("Runtime (seconds)", f"{aggregate_results['runtime_secs']:.2f}")
    meta2.metric("Samples", f"{aggregate_results['num_samples']}")
    meta3.metric("Top-K", f"{aggregate_results['top_k']}")
    meta4.metric("Embedding Model", f"{aggregate_results['embedding_model']}")
    meta5.metric("LLM Model", f"{aggregate_results['llm_model']}")

    st.divider()

    st.subheader("Generation Quality Comparison")
    basecol, ragcol = st.columns(2)
    with basecol:
        st.markdown("### Baseline LLM")
        metcol1, metcol2, metcol3 = st.columns(3)
        st.metric("Semantic Similarity", f"{aggregate_results['baseline_semantic']:.2f}")
        metcol1.metric("Correctness", f"{aggregate_results['baseline_correctness']:.2f} ")
        metcol2.metric("Completeness", f"{aggregate_results['baseline_completeness']:.2f} ")
        metcol3.metric("Fluency", f"{aggregate_results['baseline_fluency']:.2f} ")

    with ragcol:
        st.markdown("### RAG System")
        metcol1, metcol2, metcol3 = st.columns(3)
        st.metric("Semantic Similarity", f"{aggregate_results['rag_semantic']:.2f}")
        metcol1.metric("Correctness", f"{aggregate_results['rag_correctness']:.2f} ")
        metcol2.metric("Completeness", f"{aggregate_results['rag_completeness']:.2f} ")
        metcol3.metric("Fluency", f"{aggregate_results['rag_fluency']:.2f} ")

    st.divider()

    st.subheader("Retrieval Performance")
    retcol1, retcol2, retcol3 = st.columns(3)
    retcol1.metric("Precison@K", f"{aggregate_results['precision_at_k']:.2f}")
    retcol2.metric("Recall@K", f"{aggregate_results['recall_at_k']:.2f}")
    retcol3.metric("Mean Reciprocal Rank", f"{aggregate_results['mrr']:.2f}")

    st.divider()

    st.subheader("Sample-level Evaluation Preview")
    st.caption(f"Expand to see sample preview. Download full JSON report below for complete results.")
    sample_results = results["evaluation_results"]
    if len(sample_results) > 3:
        n = 3 
    else:
        n = len(sample_results)

    for sample in sample_results[:n]:
        with st.expander(f"Sample: {sample['sample_id']}"):
            st.markdown("### Question")
            st.write(sample["question"])
            st.markdown("### Ground Truth")
            st.write(sample["ground_truths"])

            basecol, ragcol = st.columns(2)
            with basecol:
                st.markdown("### Baseline LLM Answer")
                st.write(sample["baseline_answer"]) 
                st.markdown("### Judge Scores")
                st.write(sample["baseline_judge_scores"])

            with ragcol:
                st.markdown("### RAG Answer")
                st.write(sample["rag_answer"]) 
                st.markdown("### Judge Scores")
                st.write(sample["rag_judge_scores"])
        
            st.divider()
            st.markdown("### Retrieved Chunks")
            for i, chunk in enumerate(sample["retrieved_chunks"], 1):
                st.write(f"Chunk {i}: {chunk}")
            
            st.divider()
            st.markdown("### Retrieval Metrics")
            retcol1, retcol2, retcol3 = st.columns(3)
            retcol1.metric("Precison@K", f"{sample['precision_at_k']:.2f}")
            retcol2.metric("Recall@K", f"{sample['recall_at_k']:.2f}")
            retcol3.metric("Mean Reciprocal Rank", f"{sample['mrr']}")

        st.divider()

        results_json = json.dumps(results, indent = 4, ensure_ascii = False)
        st.download_button(
            label = "Download Full Evaluation Report Here",
            data = results_json,
            file_name = "evaluation_report.json",
            mime = "application/json",
            use_container_width =True
        )

        