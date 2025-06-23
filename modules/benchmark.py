from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
# import openai


class GeneratedHeadlinesBenchmark():
    def __init__(self):
        self.encode_model = SentenceTransformer("ai-forever/ru-en-RoSBERTa")
        # self.LLM_model = 'gpt-4-mini'
        self.tokenizer = AutoTokenizer.from_pretrained("ai-forever/ru-en-RoSBERTa")

    def RougeScore(self, reference_headline, generated_headline):
        results = {}
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], tokenizer=self.tokenizer)
        rouge_scores = scorer.score(reference_headline, generated_headline)
        results['ROUGE-1'] = rouge_scores['rouge1'].fmeasure
        results['ROUGE-2'] = rouge_scores['rouge2'].fmeasure
        results['ROUGE-L'] = rouge_scores['rougeL'].fmeasure

        return results

    def MeteorScore(self, reference_headline, generated_headline):
        return meteor_score([self.tokenizer.tokenize(generated_headline)], self.tokenizer.tokenize(reference_headline))

    def CiderScore(self, reference_headline, generated_headline):
        ref_words = set(reference_headline.split())
        gen_words = set(generated_headline.split())
        intersection = len(ref_words & gen_words)
        return intersection / (len(gen_words) + 1e-6)

    def CS_CR_Score(self, text, reference_headline, generated_headline):
        res = {}
        ref_embedding = self.encode_model.encode(reference_headline, convert_to_tensor=True)
        gen_embedding = self.encode_model.encode(generated_headline, convert_to_tensor=True)
        text_embedding = self.encode_model.encode(text, convert_to_tensor=True)

        generated_text_similarity = util.pytorch_cos_sim(text_embedding, gen_embedding).item()
        reference_text_similarity = util.pytorch_cos_sim(text_embedding, ref_embedding).item()
        max_sim = max(reference_text_similarity, generated_text_similarity) + 1e-6

        res['Cosine Similarity'] = util.pytorch_cos_sim(ref_embedding, gen_embedding).item()
        res['Conseptual Relevance'] = generated_text_similarity / max_sim
        return res

    # def LLM_Score(self, text, ref_headline, gen_headline):
    #     prompt = f"""
    #     You are an expert in evaluating headline quality.\n
    #     Given the following text:

    #     "{text}"

    #     Compare the two headlines:
    #     1. Reference: "{ref_headline}"
    #     2. Generated: "{gen_headline}"

    #     Score the generated headline on:
    #     - Informativeness (0-10)
    #     - Readability (0-10)
    #     - Creativity (0-10)
    #     - Overall Quality (0-10)

    #     Provide a JSON response with scores.
    #     """
    #     try:
    #         response = openai.ChatCompletion.create(
    #             model=self.LLM_model,
    #             messages=[{"role": "system", "content": "You are a helpful AI assistant."},
    #                       {"role": "user", "content": prompt}]
    #         )
    #         return response["choices"][0]["message"]["content"]
    #     except Exception as e:
    #         return {"error": str(e)}

    def calculate_metrics(self, text, reference_headline, generated_headline):
        return {
            'Rouge': self.RougeScore(reference_headline, generated_headline),
            'Meteor': self.MeteorScore(reference_headline, generated_headline),
            'Cider': self.CiderScore(reference_headline, generated_headline),
            'CS_CR': self.CS_CR_Score(text, reference_headline, generated_headline),
            # 'LLM_Score': self.LLM_Score(text, reference_headline, generated_headline)
        }
