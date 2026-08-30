import json
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from drive_manager import DriveManager

class GeminiDriveAgent:
    def __init__(self, api_key: str, drive_manager: DriveManager, model_name: str = "gemini-2.5-flash"):
        """
        Инициализация ИИ-агента Gemini.
        :param api_key: Ключ Google AI Studio.
        :param drive_manager: Экземпляр DriveManager для работы с файлами.
        :param model_name: Модель (gemini-2.5-flash или gemini-2.5-pro).
        """
        self.client = genai.Client(api_key=api_key)
        self.drive = drive_manager
        self.model_name = model_name
        self.photos_index: List[Dict[str, Any]] = []

    def set_photos_index(self, index: List[Dict[str, Any]]):
        """Обновляет локальный кэш метаданных фотографий."""
        self.photos_index = index

    def search_photos(self, query: str, folder_hint: Optional[str] = None) -> str:
        """
        Ищет фото по совпадению в описании, имени файла, пути папки или дате.
        """
        if not self.photos_index:
            return json.dumps({"error": "Индекс фотографий пуст или еще не загружен."})

        query_tokens = query.lower().split()
        results = []

        for photo in self.photos_index:
            text_corpus = f"{photo['name']} {photo['path']} {photo['description']} {photo['created_at']}".lower()
            
            # Проверка ключевых слов
            matches_query = any(token in text_corpus for token in query_tokens)
            
            # Проверка фильтра по папке, если указан
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

        # Возвращаем максимум 10 наиболее релевантных совпадений
        return json.dumps(results[:10], ensure_ascii=False)

    def generate_response(self, conversation_history: List[Dict[str, Any]], user_message: str) -> tuple[str, List[Dict[str, Any]], List[bytes]]:
        """
        Обрабатывает запрос пользователя, выполняет Function Calling и возвращает:
        (ответ_текстом, обновленная_история, список_просмотренных_изображений)
        """
        # Системный промпт с правилами работы
        system_instruction = """
Ты — персональный мультимодальный ИИ-напарник с доступом к Google Диску пользователя.
Твоя задача — помогать в рабочих и бытовых делах, опираясь на фотографии, документы и метаданные.

ПРАВИЛА РАБОТЫ С ДИСКОМ:
1. Когда пользователь спрашивает о вещах, чеках, документах, схемах или прошлых событиях — сначала вызови функцию `search_photos` для поиска файлов по ключевым словам, датам или названию папок.
2. Изучи описания (description), которые пользователь вносил вручную.
3. Если для точного ответа нужно увидеть само изображение (прочесть мелкий шрифт, оценить деталь, серийный номер, внешний вид), запроси загрузку фото.
4. Отвечай емко, точно и по существу на русском языке.
"""

        # Определение инструментов для Gemini
        tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="search_photos",
                        description="Ищет фотографии на Google Диске по ключевым словам в описании, имени файла, пути папки или дате создания.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "query": types.Schema(
                                    type=types.Type.STRING,
                                    description="Ключевые слова для поиска (например: 'счет за свет', 'гарантия холодильник', 'электрика кухня')."
                                ),
                                "folder_hint": types.Schema(
                                    type=types.Type.STRING,
                                    description="Необязательное уточнение названия папки (например: 'Ремонт', 'Документы')."
                                )
                            },
                            required=["query"]
                        )
                    )
                ]
            )
        ]

        # Преобразуем историю в формат Contents
        contents = []
        for msg in conversation_history:
            role = msg["role"]
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
        
        # Добавляем новое сообщение
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

        loaded_images = []

        # Цикл обработки Function Calling
        while True:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=tools,
                    temperature=0.2
                )
            )

            # Проверяем, есть ли вызов функции
            function_calls = response.function_calls
            if not function_calls:
                # Финальный текстовый ответ получен
                final_text = response.text or ""
                return final_text, contents, loaded_images

            # Обрабатываем вызовы функций
            for call in function_calls:
                name = call.name
                args = call.args

                if name == "search_photos":
                    search_res = self.search_photos(
                        query=args.get("query", ""),
                        folder_hint=args.get("folder_hint")
                    )
                    
                    # Запоминаем ответ функции
                    contents.append(response.candidates[0].content)
                    contents.append(types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=name,
                                response={"result": search_res}
                            )
                        ]
                    ))

                    # Если найдены кандидаты, сразу подгружаем первый релевантный оригинал в контекст для анализа
                    parsed_res = json.loads(search_res)
                    if parsed_res and isinstance(parsed_res, list):
                        top_photo = parsed_res[0]
                        file_bytes = self.drive.download_image_bytes(top_photo["id"])
                        loaded_images.append(file_bytes)
                        
                        # Передаем оригинальное изображение напрямую модели
                        contents.append(types.Content(
                            role="user",
                            parts=[
                                types.Part.from_bytes(
                                    data=file_bytes,
                                    mime_type="image/jpeg"
                                ),
                                types.Part.from_text(
                                    text=f"Вот оригинальное изображение файла '{top_photo['name']}' (путь: {top_photo['folder_path']}). Описание: '{top_photo['description']}'. Изучи его для ответа."
                                )
                            ]
                        ))