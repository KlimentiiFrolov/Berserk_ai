from typing import Literal, TypedDict
from string import Template
from evaluate import load

from rag.llm import LLMClient, LLMResponse, Message

JUDGE_SYSTEM_PROMPT = (
    "Ты - эксперт-фактчекер. Твоя задача: сравнить ответ модели "
    "с эталонным ответом и оценить фактическую точность."
)

JUDGE_USER_TEMPLATE = Template("""\
Вопрос пользователя:
<question>
${question}
</question>

Эталонный ответ:
<reference>
${reference}
</reference>

Ответ модели:
<answer>
${answer}
</answer>

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
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def evaluate(
        self,
        question: str,
        answer: str,
        reference: str,
    ) -> str:
        user_prompt = JUDGE_USER_TEMPLATE.substitute(
            question=question,
            reference=reference,
            answer=answer,
        )
        messages = [
            Message(role="system", content=JUDGE_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]

        response: LLMResponse = await self.llm.generate(messages)
        return response.text
    
    
class BertScoreExample(TypedDict):
    precision: float
    recall: float
    f1: float

class BertScoreResult(TypedDict):
    precision: float
    recall: float
    f1: float
    per_example: list[BertScoreExample]

class BertScore:
    def __init__(
        self,
        lang: Literal["ru", "en"] = "en",
        model_type: str = "microsoft/deberta-xlarge-mnli",
        device: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        self.metric = load("bertscore")
        self.lang = lang
        self.model_type = model_type
        self.device = device

    def evaluate(
        self,
        answer: list[str] | str,
        reference: list[str] | str,
    ) -> BertScoreResult:
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
                {"precision": p, "recall": r, "f1": f}
                for p, r, f in zip(
                    results["precision"],
                    results["recall"],
                    results["f1"],
                )
            ],
        }
