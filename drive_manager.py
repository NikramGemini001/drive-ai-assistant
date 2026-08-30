import io
from typing import List, Dict, Any, Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class DriveManager:
    def __init__(self, credentials_info: dict):
        """
        Инициализация клиента Google Drive.
        :param credentials_info: Словарь с данными сервисного аккаунта (из st.secrets или json).
        """
        self.creds = service_account.Credentials.from_service_account_info(
            credentials_info, scopes=SCOPES
        )
        self.service = build('drive', 'v3', credentials=self.creds)

    def _fetch_all_folders(self) -> Dict[str, Dict[str, Any]]:
        """Получает все доступные папки для построения дерева путей."""
        folders = {}
        page_token = None
        
        while True:
            response = self.service.files().list(
                q="mimeType = 'application/vnd.google-apps.folder' and trashed = false",
                spaces='drive',
                fields='nextPageToken, files(id, name, parents)',
                pageToken=page_token
            ).execute()
            
            for folder in response.get('files', []):
                folders[folder['id']] = {
                    'name': folder['name'],
                    'parents': folder.get('parents', [])
                }
                
            page_token = response.get('nextPageToken')
            if not page_token:
                break
                
        return folders

    def _build_folder_path(self, folder_id: str, folders_map: Dict[str, dict]) -> str:
        """Рекурсивно строит полный путь к папке."""
        path_segments = []
        curr_id = folder_id
        
        # Защита от зацикливания
        visited = set()
        while curr_id in folders_map and curr_id not in visited:
            visited.add(curr_id)
            folder = folders_map[curr_id]
            path_segments.insert(0, folder['name'])
            parents = folder.get('parents', [])
            curr_id = parents[0] if parents else None
            
        return "/" + "/".join(path_segments) if path_segments else "/"

    def get_all_photos_metadata(self) -> List[Dict[str, Any]]:
        """
        Сканирует все фотографии, вытягивает описания, даты и строит пути папок.
        """
        folders_map = self._fetch_all_folders()
        photos_index = []
        page_token = None
        
        # Ищем только изображения (jpg, png, heic, webp и т.д.)
        query = "mimeType contains 'image/' and trashed = false"
        
        while True:
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType, description, createdTime, modifiedTime, parents, size)',
                pageToken=page_token
            ).execute()
            
            for f in response.get('files', []):
                parent_id = f.get('parents', [None])[0]
                folder_path = self._build_folder_path(parent_id, folders_map) if parent_id else "/"
                
                photos_index.append({
                    "id": f["id"],
                    "name": f["name"],
                    "path": folder_path,
                    "description": f.get("description", "").strip(),
                    "created_at": f.get("createdTime", ""),
                    "modified_at": f.get("modifiedTime", ""),
                    "mime_type": f.get("mimeType", ""),
                    "size_bytes": f.get("size", 0)
                })
                
            page_token = response.get('nextPageToken')
            if not page_token:
                break
                
        return photos_index

    def download_image_bytes(self, file_id: str) -> bytes:
        """
        Скачивает файл в оригинальном разрешении для отправки в Gemini Vision.
        """
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            
        fh.seek(0)
        return fh.read()