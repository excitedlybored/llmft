from transformers import AutoModelForSeq2SeqLM
import logging

logging.basicConfig(level=logging.INFO)

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")