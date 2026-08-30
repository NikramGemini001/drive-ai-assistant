import time
from typing import List, Dict, Any
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

    def search_photos(self, query: str) -> List[Dict[str, Any]]:
        """Умный поиск фото по совпадениям в описании, названии, пути и дате."""
        if not self.photos_index:
            return []

        tokens = [t.strip().lower() for t in query.split() if len(t.strip()) > 2]
        if not tokens:
            tokens = [query.strip().lower()]

        scored_results = []

        for photo in self.photos_index:
            score = 0
            desc = photo.get("description", "").lower()
            name = photo.get("name", "").lower()
            path = photo.get("path", "").lower()
            created = photo.get("created_at", "").lower()

            for token in tokens:
                # Поиск по корню/подстроке (например, "распред" найдет "распредка", "распредкоробка")
                stem = token[:6] if len(token) > 6 else token
                
                if stem in desc:
                    score += 10
                if stem in path:
                    score += 6
                if stem in name:
                    score += 4
                if stem in created:
                    score += 3

            if score > 0:
                scored_results.append((score, photo))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_results[:3]]

    def _call_model_with_retry(self, contents, config, retries=3, delay=3):
        """Повтор запроса при временной перегрузке серверов."""
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
        matched_photos = self.search_photos(user_message)
        loaded_images = []
        parts = []

        system_instruction = """
Ты — персональный мультимодальный ИИ-напарник с прямым доступом к Google Диску пользователя.
Твоя задача — помогать в рабочих и бытовых делах, опираясь на фотографии, документы, чеки, схемы и метаданные.

ПРАВИЛА:
1. Если к запросу прикреплены найденные фото — детально изучи их оригиналы и описания, которые пользователь вносил вручную.
2. Отвечай точно, емко и по существу на русском языке. Опиши важные визуальные детали, если это требуется.
3. Если ничего подходящего не найдено, ответь по общему смыслу или подскажи, как переформулировать поиск.
"""

        # Если нашлись совпадения — скачиваем оригиналы и прикрепляем в контекст
        if matched_photos:
            meta_info = []
            for idx, photo in enumerate(matched_photos, start=1):
                meta_info.append(
                    f"Фото {idx}: '{photo['name']}'\n"
                    f"Папка: '{photo['path']}'\n"
                    f"Описание от пользователя: '{photo['description']}'\n"
                    f"Дата создания: {photo['created_at']}"
                )
                try:
                    file_bytes = self.drive.download_image_bytes(photo["id"])
                    loaded_images.append(file_bytes)
                    mime = photo.get("mime_type") or "image/jpeg"
                    parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime))
                except Exception:
                    pass

            context_text = "НАЙДЕННЫЕ НА ДИСКЕ ФОТОГРАФИИ:\n" + "\n---\n".join(meta_info) + "\n\n"
            parts.append(types.Part.from_text(text=context_text))

        parts.append(types.Part.from_text(text=f"Запрос пользователя: {user_message}"))

        contents = []
        for msg in conversation_history:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

        contents.append(types.Content(role="user", parts=parts))

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2
        )

        response = self._call_model_with_retry(contents=contents, config=config)
        return response.text or "", contents, loaded_images
