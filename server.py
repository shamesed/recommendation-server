from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
import os
import pathlib
import re
from google.genai import types
import json
import traceback

app = FastAPI()
# Инициализация клиента (ключ лучше хранить в переменной окружения!)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

#%%

SYSTEM_INSTRUCTION = """Ты работаешь на моё мобильное приложение Одежда по погоде и твоя задача - \n
        генерировать рекомендации по одежде и аксессуарам для пользователей. \n
        Рекомендация разделена на 18 пунктов, которые представлены в приложенной таблице \n 
        Тебе нужно её полностью изучить.\n
        Тебе выдаётся input строка вида: \n
        Температура: X°C, Пол: X, Возраст: X Группа:X \n Рекомендация: \n
        Твоя задача - сгенерировать 18 пунктов рекомендации, основываясь на этих параметрах. \n
        Отвечаешь строго в формате JSON со всеми ключами рекомендации, имя столбца соответствует названию ключа.\n
        Никакого текста вне JSON. \n
        Пример твоего ответа:\n
        {\n
          "first_layer": "Лёгкая рубашка с длинным рукавом",\n
          "first_layer_materials": "Хлопок, лён, вискоза или бамбук",\n
          "second_layer": null,\n
          "pants_first_layer": null,\n
          "pants_second_layer": "Лёгкие брюки",\n
          "brands_pants": null,\n
          "socks": "Хлопковые носки или отказаться от них, если позволяет обувь",\n
          "shoes": "Эспадрильи, лёгкие кроссовки, сандалии либо кеды",\n
          "brands_shoes": null,\n
          "outerwear": null,\n
          "brands_outerwear": null,\n
          "head": "Кепка или панама с большими полями",\n
          "hands": null,\ns
          "neck": null,\n
          "face": "Использовать солнцезащитный крем с SPF-50, надеть солнцезащитные очки",\n
          "extra": "Взять с собой бутылочку воды; избегать активных физических нагрузок",\n
          "extra_2": "Не находиться в местах, где прямые солнечные лучи попадают на кожу; по возможности не выходить на улицу",\n
          "jokes": "На улице такая жара, что мои планы на лето решили взять отпуск и отправиться на Северный полюс"\n
        }\n
        Для твоего обучения я приложил файл pdf, в котором представлены примеры рекомендаций для каждой температуры. \n
        Ты должен его тщательно изучить и обучиться на нём.\n
        Запомни, ты отвечаешь только в формате json."""
#%%

class WeatherData(BaseModel):
    temperature: int
    gender: str
    age: int
    group: str
#%%
# Retrieve and encode the PDF byte

filepath = pathlib.Path('ML.pdf')

#%%

@app.post("/recommend")
def recommend(data: WeatherData):
    try:
        prompt = f"""
        Температура: {data.temperature}°C,
        Пол: {data.gender}, 
        Возраст: {data.age}, 
        Группа: {data.group}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(   
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                system_instruction=SYSTEM_INSTRUCTION
            ),
            contents=[
                types.Part.from_bytes(
                    data=filepath.read_bytes(),
                    mime_type='application/pdf',
                )
            ]
        )

        # --- Обработка текста ответа ---
        raw_text = response.text
        cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.DOTALL)

        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = extract_json_substring(cleaned)  # твоя fallback-функция
            if not parsed:
                return {
                    "error": "Invalid JSON response",
                    "raw": raw_text
                }

        # --- Вот тут добавляем return ---
        return {"recommendation": parsed}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


