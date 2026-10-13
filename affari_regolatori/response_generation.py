import shared_config

def generate_rag_response(bedrock_client, context_list, full_chat_history):
    formatted_context = ""
    for i, ctx in enumerate(context_list):
        formatted_context += f"<doc id='{i+1}'>\n{ctx}\n</doc>\n"
    
    base_system = (
        "You are a senior legal expert in energy regulation for Edison. "
        "Your task is to provide comprehensive operational opinions to support the business, relying strictly on the information provided in the context or on known current regulations.\n"
        "Use formal and discursive legal language, citing specific articles and regulatory references.\n"
        "Give the answer in Italian."
    )
    
    rag_system_prompt = (
        f"{base_system}\n\n"
        "RAG INSTRUCTIONS\n"
        "You have access to a set of retrieved documents enclosed in <context> tags.\n"
        "1. Any mention of specific regulations, articles, obligations, or deadlines must be derived EXCLUSIVELY from the <context>.\n"
        "2. Combine information from multiple chunks to answer fully.\n"
        "3. To fulfill the requirement of providing 'plausible practical cases', you are authorized to apply the rules found in the <context> to hypothetical scenarios based on your general knowledge of the energy sector.\n"
        "4. If the answer is not present in the <context>, you must reply exactly: [RISPOSTA_NON_TROVATA]\n"
        "5. CITE NATURALLY: Cite sources narratively within the text. It is STRICTLY FORBIDDEN to append reference tags, IDs, or brackets at the end of sentences.\n"
        "6. Omit any typographical or formatting instructions found in the original text and focus solely on the regulatory content.\n"
        f"<context>\n{formatted_context}\n</context>"
    )

    conversation = []
    for msg in full_chat_history:
        role = "assistant" if msg["role"] == "assistant" else "user"
        content = [{"text": msg["content"]}]
        conversation.append({"role": role, "content": content})
    
    try:
        response_stream = bedrock_client.converse_stream(
            modelId=shared_config.INFERENCE_PROFILE, 
            messages=conversation,
            system=[{"text": rag_system_prompt}],
            inferenceConfig={"maxTokens": 2048, "temperature": 0.5, "topP": 0.9},
        )
        for event in response_stream['stream']:
            if 'contentBlockDelta' in event:
                delta = event['contentBlockDelta'].get('delta', {})
                if 'text' in delta:
                    yield delta['text']
            elif 'reasoningUpdate' in event:
                delta = event['reasoningUpdate'].get('delta', {})
                if 'text' in delta:
                    yield delta['text']
                    
    except Exception as e:
        yield f"\nERROR during Bedrock call: {e}"

def generate_direct_response(bedrock_client, full_chat_history):
    conversation = []
    for msg in full_chat_history:
        role = "assistant" if msg["role"] == "assistant" else "user"
        content = [{"text": msg["content"]}]
        conversation.append({"role": role, "content": content})
    
    try:
        response_stream = bedrock_client.converse_stream(
            modelId=shared_config.INFERENCE_PROFILE, 
            messages=conversation,
            inferenceConfig={"maxTokens": 2048, "temperature": 0.5},
        )
        for event in response_stream['stream']:
            if 'contentBlockDelta' in event:
                delta = event['contentBlockDelta'].get('delta', {})
                if 'text' in delta:
                    yield delta['text']
            elif 'reasoningUpdate' in event:
                delta = event['reasoningUpdate'].get('delta', {})
                if 'text' in delta:
                    yield delta['text']
    except Exception as e:
        yield f"\nERROR during Bedrock call: {e}"