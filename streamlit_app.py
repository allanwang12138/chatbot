if query and option:
    with st.spinner("Searching context..."):
        docs = db.similarity_search_with_score(query, k=5)
        filtered_docs = [doc for doc, score in docs if doc.page_content.strip()]
        context_text = "\n\n".join([doc.page_content for doc in filtered_docs])

    # Check if retrieved context is too empty
    if not context_text.strip() or len(context_text) < 100:
        response = "❗️This question appears to be outside the scope of the macroeconomics textbook."
    else:
        prompt_template = ChatPromptTemplate.from_template(
            PROMPT_DETAILED if option == "Detailed Answer" else PROMPT_CONCISE
        )
        prompt = prompt_template.format(context=context_text, question=query)
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

        with st.spinner("Generating answer..."):
            response = model.predict(prompt)

    if "Voice" in option and "❗️" not in response:
        voice_choice = st.session_state.get("voice", "alloy")
        with st.spinner(f"Generating voice with '{voice_choice}'..."):
            speech_response = openai.audio.speech.create(
                model="tts-1",
                voice=voice_choice,
                input=response
            )
            audio_path = "output.mp3"
            with open(audio_path, "wb") as f:
                f.write(speech_response.read())
            audio_file = open(audio_path, "rb")
            st.audio(audio_file.read(), format="audio/mp3")
    else:
        st.markdown("### Answer")
        st.write(response)

    with st.expander("📚 Relevant Context from Macroeconomics Textbook"):
        st.markdown(f"<div style='white-space: pre-wrap; font-size: 0.9em'>{context_text}</div>", unsafe_allow_html=True)
