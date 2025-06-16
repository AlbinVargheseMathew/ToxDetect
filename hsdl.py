import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import gradio as gr

class HateSpeechDetector:
    def __init__(self):
        try:
            # Use a different model specifically trained for hate speech
            self.model_name = "unitary/toxic-bert"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline with the model
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1 if not torch.cuda.is_available() else 0,
                return_all_scores=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def predict(self, text):
        if not isinstance(text, str):
            return {"Error": "Input must be a string"}
        
        if not text.strip():
            return {"Error": "Input text cannot be empty"}

        try:
            # Get prediction scores
            results = self.classifier(text)[0]
            toxic_score = next((res['score'] for res in results if res['label'] == 'toxic'), 0)
            non_toxic_score = 1 - toxic_score
            
            return {
                "Hate Speech": round(float(toxic_score), 4),
                "Not Hate Speech": round(float(non_toxic_score), 4)
            }
        except Exception as e:
            return {"Error": f"Prediction failed: {str(e)}"}

def create_gradio_interface():
    try:
        # Initialize the detector
        detector = HateSpeechDetector()
        
        # Create Gradio interface
        iface = gr.Interface(
            fn=detector.predict,
            inputs=gr.Textbox(
                label="Enter text to analyze",
                placeholder="Type your text here..."
            ),
            outputs=gr.Label(label="Classification Results"),
            title="TOXDETECT: Hate Speech Detection",
            description="Enter text to analyze for potential hate speech content",
            examples=[
                ["This is a friendly and positive message!"],
                ["I hate all people from that country."],
                ["Everyone deserves to be treated with respect."]
            ],
            allow_flagging="never"
        )
        return iface
    except Exception as e:
        raise RuntimeError(f"Failed to create interface: {str(e)}")

if __name__ == "__main__":
    try:
        # Create and launch the interface
        iface = create_gradio_interface()
        iface.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True
        )
    except Exception as e:
        print(f"Application failed to start: {str(e)}")