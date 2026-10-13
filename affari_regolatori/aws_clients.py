import streamlit as st
import boto3
from langchain_aws import BedrockEmbeddings
from sentence_transformers.cross_encoder import CrossEncoder

import shared_config


# --- AWS Clients and Embeddings ---
@st.cache_resource
def get_bedrock_client():
    print("Connecting to Bedrock...")
    client = boto3.client(service_name='bedrock-runtime', region_name=shared_config.REGION)
    return client
