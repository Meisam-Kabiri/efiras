import io
import time

import fitz
import torch
from PIL import Image  # Python Imaging Library (now called Pillow)
from transformers import DetrForObjectDetection, DetrImageProcessor


class FastTableDetector:
    def __init__(self):
        print("🔄 Loading DETR table detection model (166MB)...")
        start_time = time.time()

        self.processor = DetrImageProcessor.from_pretrained(
            "TahaDouaji/detr-doc-table-detection"
        )
        self.model = DetrForObjectDetection.from_pretrained(
            "TahaDouaji/detr-doc-table-detection"
        )

        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"🖥️ Using device: {self.device}")

    def has_tables_fast(self, pdf_path, page_num, confidence_threshold=0.98):
        """Fast table detection - optimized for speed"""
        start_time = time.time()

        # Convert PDF page to image (lower resolution for speed)
        doc = fitz.open(pdf_path)
        page = doc[page_num]

        # Use lower DPI for speed (1.5x instead of 2x), # 1.0x scale (72 DPI): too much blurry, 1.5x scale (108 DPI)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_data = pix.tobytes("png")
        # PIL expects file path, not raw bytes
        # Need to wrap bytes in BytesIO (fake file object): fake_file = io.BytesIO(img_data)
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        doc.close()

        # Detect tables
        inputs = self.processor(
            images=image, return_tensors="pt"
        )  # "pt" = PyTorch format

        # Move inputs to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():  # Speed optimization
            outputs = self.model(**inputs)

        # Why? Different libraries expect different order:
        # PIL: (width, height)
        # NumPy/PyTorch: (height, width)
        # OpenCV: (height, width)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )[0]

        # results is a LIST containing one dictionary per image:
        # results = [
        #     {  # Dictionary for first (and only) image
        #         'scores': tensor([0.9234, 0.8567, 0.7891]),     # Confidence scores
        #         'labels': tensor([0, 0, 0]),                    # Class labels (all tables = 0)
        #         'boxes': tensor([[100.2, 200.5, 400.8, 350.1], # Bounding boxes [x1, y1, x2, y2]
        #                         [50.1, 450.0, 300.7, 600.3],
        #                         [500.0, 100.0, 700.0, 250.0]])
        #     }
        # ]

        processing_time = time.time() - start_time
        has_tables = len(results["scores"]) > 0
        print(results["scores"].tolist())

        return {
            "has_tables": has_tables,
            "num_tables": len(results["scores"]),
            "processing_time": processing_time,
            "confidence_scores": [
                round(score.item(), 3) for score in results["scores"]
            ],
        }

    def batch_scan_pdf(self, pdf_path, max_pages=None):
        """Scan PDF with performance metrics"""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_scan = min(total_pages, max_pages) if max_pages else total_pages
        doc.close()

        print(f"🔍 Scanning {pages_to_scan} pages for tables...")

        pages_with_tables = []
        total_time = 0

        for page_num in range(pages_to_scan):
            result = self.has_tables_fast(pdf_path, page_num)
            total_time += result["processing_time"]

            if result["has_tables"]:
                pages_with_tables.append(
                    {
                        "page": page_num,
                        "num_tables": result["num_tables"],
                        "confidence_scores": result["confidence_scores"],
                        "processing_time": result["processing_time"],
                    }
                )

                print(
                    f"📊 Page {page_num + 1}: {result['num_tables']} tables "
                    f"(conf: {result['confidence_scores']}) - {result['processing_time']:.1f}s"
                )
            else:
                print(
                    f"📄 Page {page_num + 1}: No tables - {result['processing_time']:.1f}s"
                )

        avg_time = total_time / pages_to_scan
        print(f"\n📈 Performance Summary:")
        print(f"   • Total time: {total_time:.1f}s")
        print(f"   • Average per page: {avg_time:.1f}s")
        print(f"   • Pages with tables: {len(pages_with_tables)}")
        print(
            f"   • Cost savings: {((pages_to_scan - len(pages_with_tables)) / pages_to_scan * 100):.1f}%"
        )

        return pages_with_tables


# Usage examples
if __name__ == "__main__":
    # Initialize detector (downloads model first time)
    detector = FastTableDetector()

    # Quick test on single page
    pdf_file = "BaselFramework.pdf"
    result = detector.has_tables_fast(pdf_file, page_num=19)
    print(f"Page 1 result: {result}")

    # Batch scan entire PDF
    # table_pages = detector.batch_scan_pdf(pdf_file, max_pages=300)  # Limit for testing

    # # Now send only table pages to Azure
    # print(f"\n🎯 Send these {len(table_pages)} pages to Azure:")
    # for page_info in table_pages:
    #     print(f"   • Page {page_info['page'] + 1} ({page_info['num_tables']} tables)")
