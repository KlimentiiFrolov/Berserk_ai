from string import Template
from evaluate import load

prompt_template = Template("""Ты - эксперт‑фактчекер. Твоя задача: сравнить ответ модели с эталонным ответом и оценить фактическую точность.
Вопрос пользователя:
<question>
Эталонный ответ:
<reference>

Ответ модели:
<answer>

Критерии проверки:
- Все факты должны совпадать с эталоном.
- Не должно быть выдуманных дат, имён, чисел.
- Если в ответе модели есть дополнительная информация, она не должна противоречить эталону.

Шаги:
1. Выпиши все проверяемые факты из эталона (дата, имя, число, событие).
2. Проверь каждый факт в ответе модели.
3. Если есть расхождение, укажи его и тип ошибки (неверное название, вымышленное имя, карта и т.п.).
4. Дай итоговую оценку точности по шкале 0-100.
5. Кратко напиши, можно ли доверять ответу без ручной проверки.
""")
class LLM_as_a_judge:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, question, answer, reference):
        prompt = prompt_template.substitute(question=question, reference=reference, answer=answer)
        response = self.llm.generate(prompt)
        return response
    
    

class BertScore:
    def __init__(self, lang="en", model_type=None, device="cuda"):
        self.metric = load("bertscore")
        self.lang = lang
        self.model_type = model_type  # например, "microsoft/deberta-xlarge-mnli"
        self.device = device

    def evaluate(self, answer: list[str] | str, reference: list[str] | str):
        if isinstance(answer, str):
            answer = [answer]
        if isinstance(reference, str):
            reference = [reference]

        results = self.metric.compute(
            predictions=answer,
            references=reference,
            lang=self.lang,
            model_type=self.model_type,
            device=self.device,
            verbose=False,
            idf=False,
            rescale_with_baseline=False,
        )
        assert results is not None, "BERTScore evaluation failed."

        return {
            "precision": results["precision"].mean().item(),
            "recall": results["recall"].mean().item(),
            "f1": results["f1"].mean().item(),
            "per_example": [
                {
                    "precision": p,
                    "recall": r,
                    "f1": f,
                }
                for p, r, f in zip(results["precision"], results["recall"], results["f1"])
            ],
        }
