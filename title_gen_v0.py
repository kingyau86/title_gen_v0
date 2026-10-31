# -*- coding: utf-8 -*-
import re
import string
import time
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(
    page_title="Title Generator for Research Paper",
    layout="centered"
)

# --- Cached Model Loading ---
@st.cache_resource
def load_model():
    #MODEL_PATH = "./fine_tuned_t5_title_gen_v1"
    MODEL_PATH = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    return tokenizer, model

# --- UI Layout ---
st.title("🔬 Title Generation for Research Paper")
st.write("Provide your abstract from research paper below to generate a title using our custom model.")

# Data Entry Area
st.subheader("Data Input")
abstract_input = st.text_area(
    label="Paste your research abstract here:",
    placeholder="Type or paste text (English recommended) for our processing...",
    height=220
)

# Action Button
st.subheader("Model Controls")
submit_button = st.button(
    "Load model and generate title",
    type="primary",
    use_container_width=True
)

# Dedicated Space for the Output Result
st.subheader("Output Result")
output_container = st.container(border=True)


# --- Backend Processing & Logic Execution ---
if submit_button:
    if not abstract_input.strip():
        st.warning("Input required. Please input an abstract first.")
    else:
        with st.spinner("1. Loading model components... Please wait :)"):
            try:
                tokenizer, model = load_model()

                with st.spinner("2. Model is reading the abstract and generating title..."):
                    prompt = ("Summarize the main contribution of this abstract into a concise academic paper title. "
                        "Use a clear noun phrase or 'Topic: Subtopic' format. Do not ask questions or start with verbs.\n\n"
                        f"title:\n{abstract_input}")
                        #f"Write a short, professional paper title (noun phrase only, no full sentences) for this abstract: {abstract_input}"

                    model_input = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True
                    )

                    start_time = time.time()
                    model_output = model.generate(
                        input_ids=model_input["input_ids"],
                        attention_mask=model_input.get("attention_mask"),
                        do_sample=False,
                        num_beams=5,
                        length_penalty=0.8,
                        repetition_penalty=1.5,
                        max_length=25,
                        min_length=5,
                        no_repeat_ngram_size=3,
                        early_stopping=True
                    )
                    
                    elapsed_time = time.time() - start_time

                    # 1. Extract the title safely FIRST
                    generated_title = tokenizer.decode(model_output[0], skip_special_tokens=True)

                    # 2. Remove generic starting prefixes using regex
                    prefixes_to_remove = [
                        r"^(we|this paper|this study|this work)\s+(propose|presents|introduces|develops|investigates|examines|suggests|describes)\s+(a|an|the)?\s*",
                        r"^(is|are|can|does|how|why|what)\s+",
                        r"^a\s+(novel|study|research|analysis|paper)\s+(to|for|of|on|into)\s+"
                    ]

                    for pattern in prefixes_to_remove:
                        generated_title = re.sub(pattern, "", generated_title, flags=re.IGNORECASE)

                    # 3. Trim dangling words at the end of the title (prepositions, conjunctions, verbs)
                    dangling_endings = r"\s+(and|or|that|which|led|to|for|of|in|on|with|as|by|the|a|an)$"

                    while re.search(dangling_endings, generated_title, flags=re.IGNORECASE):
                        generated_title = re.sub(dangling_endings, "", generated_title, flags=re.IGNORECASE)

                    # 4. Strip lingering punctuation
                    generated_title = generated_title.translate(str.maketrans("", "", '!"#$%&\'()*+,-./;<=>?@[\\]^_`{|}~')).strip()

                # Render results into the container
                with output_container:
                    st.success(f"Generation completed in {elapsed_time:.2f} seconds!")
                    st.markdown("### Model Suggested Research Title:")
                    st.info(f"**{generated_title}**")

            except Exception as e:
                st.error(f"An error occurred during execution: {e}")
                st.info("Tips: Ensure 'transformers', 'torch', 'titlecase' and 'sentencepiece' are installed.")

else:
    # Initial placeholder state before clicking the button
    with output_container:
        st.info("Your title will appear here once processing completes.")
