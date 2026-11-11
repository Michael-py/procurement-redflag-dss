"""OCR a scanned PDF with the Windows built-in OCR engine (resumable)."""
import sys, os, json, time, asyncio
import pymupdf
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.globalization import Language
from winrt.windows.graphics.imaging import BitmapDecoder
from winrt.windows.storage.streams import InMemoryRandomAccessStream, DataWriter

async def ocr_png(engine, data: bytes):
    stream = InMemoryRandomAccessStream()
    writer = DataWriter(stream)
    writer.write_bytes(data)
    await writer.store_async()
    stream.seek(0)
    decoder = await BitmapDecoder.create_async(stream)
    bmp = await decoder.get_software_bitmap_async()
    res = await engine.recognize_async(bmp)
    out = []
    for line in res.lines:
        ws = list(line.words)
        out.append({"y": min(w.bounding_rect.y for w in ws), "x": min(w.bounding_rect.x for w in ws), "t": line.text})
    return out

async def main():
    pdf, outdir = sys.argv[1], sys.argv[2]
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    last = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
    os.makedirs(outdir, exist_ok=True)
    engine = OcrEngine.try_create_from_language(Language("en-US"))
    if engine is None: raise SystemExit("no OCR engine for en-US")
    doc = pymupdf.open(pdf); t0 = time.time()
    for i in range(first - 1, min(last, len(doc))):
        fn = os.path.join(outdir, f"p{i+1:04d}.json")
        if os.path.exists(fn): continue
        pix = doc[i].get_pixmap(dpi=170)
        lines = await ocr_png(engine, pix.tobytes("png"))
        json.dump(lines, open(fn, "w", encoding="utf-8"), ensure_ascii=False)
    print("done", len(doc), round(time.time() - t0, 1))
asyncio.run(main())
