from pathlib import Path
import pandas as pd
import PyPDF2
import datetime

class FileUploadCollector:
    def __init__(self):
        pass

    def collect(self, file_path: Path):
        """Detect file type and extract content along with metadata"""
        ext = file_path.suffix.lower()
        metadata = self._get_file_metadata(file_path)

        if ext == ".csv":
            content_data, extra_metadata = self.collect_csv(file_path)
        elif ext == ".pdf":
            content_data, extra_metadata = self.collect_pdf(file_path)
        elif ext in [".xlsx", ".xls"]:
            content_data, extra_metadata = self.collect_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        metadata.update(extra_metadata)

        return {
            "metadata": metadata,
            "content": content_data
        }

    def _get_file_metadata(self, file_path: Path):
        """Get basic file metadata"""
        stat = file_path.stat()
        return {
            "file_name": file_path.name,
            "file_size": stat.st_size,  # in bytes
            "last_modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    def collect_csv(self, file_path: Path):
        df = pd.read_csv(file_path)
        metadata = {
            "type": "csv",
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(5).to_dict(orient="records"),
        }
        content = df.to_dict(orient="records")  # full data for embedding
        return content, metadata

    def collect_pdf(self, file_path: Path):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join([page.extract_text() or "" for page in reader.pages])
        metadata = {
            "type": "pdf",
            "pages": len(reader.pages),
        }
        content = text.strip()  # full text for embedding
        return content, metadata

    def collect_excel(self, file_path: Path):
        excel = pd.ExcelFile(file_path)
        sheets_data = {}
        content_data = {}
        for sheet in excel.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)
            sheets_data[sheet] = {
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head(5).to_dict(orient="records"),
            }
            content_data[sheet] = df.to_dict(orient="records")  # full data for embedding
        metadata = {
            "type": "excel",
            "sheets": sheets_data,
        }
        return content_data, metadata
