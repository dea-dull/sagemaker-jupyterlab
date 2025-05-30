import boto3
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# S3 Configuration
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Pinecone Configuration
API_KEY = os.getenv("API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

# General Metadata
ORGANIZATION_NAME = os.getenv("ORGANIZATION_NAME")