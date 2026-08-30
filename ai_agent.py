import json
import time
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from drive_manager import DriveManager

class GeminiDriveAgent:
    def __init__(self, api_key: str, drive_manager: DriveManager, model_name: str = "gemini-3.6-flash"):
        self.client = genai.Client(api_key=api_key)
        self.drive = drive_manager
        self.model_name = model_name
        self.photos_index: List[Dict[str, Any]] = []

    def set_photos_index(self, index: List[Dict[str, Any]]):
        self.photos_index = index

    def search_photos(self, query: str, folder_hint: Optional[str] = None) -> str:
        if not self.photos_index:
            return json.dumps({"error": "Индекс фотографий пуст."})

        query_tokens = query.lower().split()
        results = []

        for photo in self.photos_index:
            text_corpus = f"{photo['name']} {photo['path']} {photo['description']} {photo['created_at']}".lower()
            matches_query = any(token in text_corpus for token in query_tokens)
            
            matches_folder = True
            if folder_hint:
                matches_folder = folder_hint.lower() in photo['path'].lower()

            if matches_query and matches_folder:
                results.append({
                    "id": photo["id"],
                    "name": photo["name"],
                    "folder_path": photo["path"],
                    "description": photo["description"],
                    "date": photo["created_at"]
                })

        return json.dumps(results[:10], ensure_ascii=False)

    def _call_model_with_retry(self, contents, config, retries=3, delay=3):
        """Автоматический повтор запроса при перегрузке серверов."""
        for attempt in range(retries):
            try:
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
            except Exception as e:
                err_str = str(e)
                if ("503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str) and attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
                raise e

    def generate_response(self, conversation_history: List[Dict[str, Any]], user_message: str) -> tuple[str, List[Dict[str, Any]], List[bytes]]:
        system_instruction = """
Ты — персональный мультимодальный ИИ-напарник с доступом к Google Диску пользователя.
Твоя задача — помогать в рабочих и бытовых делах, опираясь на фотографии, документы и метаданные.

ПРАВИЛА РАБОТЫ:
1. Когда пользователь спрашивает о вещах, чеках, документах, схемах или фото — обязательно вызови функцию `search_photos`.
2. Изучи описания (description), которые пользователь вносил вручную.
3. Отвечай емко, точно и по существу на русском языке.
"""

        tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="search_photos",
                        description="Ищет фотографии на Google Диске по ключевым словам в описании, названии файла или папке.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "query": types.Schema(
                                    type=types.Type.STRING,
                                    description="Ключевые слова для поиска."
                                ),
                                "folder_hint": types.Schema(
                                    type=types.Type.STRING,
                                    description="Название папки (если указано)."
                                )
                            },
                            required=["query"]
                        )
                    )
                ]
            )
        ]

        contents = []
        for msg in conversation_history:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
        
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))
        loaded_images = []

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools,
            temperature=0.2
        )

        while True:
            response = self._call_model_with_retry(contents=contents, config=config)

            function_calls = response.function_calls
            if not function_calls:
                return response.text or "", contents, loaded_images

            for call in function_calls:
                name = call.name
                args = call.args

                if name == "search_photos":
                    search_res = self.search_photos(
                        query=args.get("query", ""),
                        folder_hint=args.get("folder_hint")
                    )
                    
                    # Фиксируем шаг модели
                    contents.append(response.candidates[0].content)
                    
                    # Передаем результат выполнения функции с ролью 'user'
                    contents.append(types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=name,
                                response={"result": search_res}
                            )
                        ]
                    ))

                    parsed_res = json.loads(search_res)
                    if parsed_res and isinstance(parsed_res, list) and len(parsed_res) > 0:
                        top_photo = parsed_res[0]
                        file_bytes = self.drive.download_image_bytes(top_photo["id"])
                        loaded_images.append(file_bytes)
                        
                        contents.append(types.Content(
                            role="user",
                            parts=[
                                types.Part.from_bytes(data=file_bytes, mime_type="image/jpeg"),
                                types.Part.from_text(text=f"Изображение '{top_photo['name']}' (описание: {top_photo['description']}). Изучи его для ответа.")
                            ]
                        ))
