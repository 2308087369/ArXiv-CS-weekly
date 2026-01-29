import fitz  # PyMuPDF
import os
import json
import re
import requests
import argparse
from typing import List, Dict, Optional, Union

class PaperFigureExtractor:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.metadata_path = os.path.join(output_dir, "figures_metadata.json")
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")
                return {}
        return {}

    def _save_metadata(self):
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def download_paper(self, arxiv_id: str, save_dir: str) -> Optional[str]:
        """Downloads a paper from arXiv if not already present."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Standardize arXiv ID (remove vX suffix for checking, but download latest or specific)
        clean_id = arxiv_id.split('v')[0]
        
        # Check if file exists (fuzzy match)
        for f in os.listdir(save_dir):
            if clean_id in f and f.endswith('.pdf'):
                print(f"Paper {arxiv_id} already exists at {os.path.join(save_dir, f)}")
                return os.path.join(save_dir, f)
                
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        save_path = os.path.join(save_dir, f"{arxiv_id}.pdf")
        
        print(f"Downloading {url} to {save_path}...")
        try:
            response = requests.get(url, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
            return save_path
        except Exception as e:
            print(f"Download failed: {e}")
            return None

    def extract_all_figures(self, pdf_path: str):
        """Extracts all figures from a PDF by finding 'Figure X' captions."""
        doc = fitz.open(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        print(f"Extracting all figures from {pdf_name}...")
        
        for i, page in enumerate(doc):
            # 1. Get all visual elements (images and drawings)
            visual_rects = []
            
            # Get images
            image_list = page.get_images()
            for img in image_list:
                xref = img[0]
                # get_image_rects returns a list of rects where this image appears
                rects = page.get_image_rects(xref)
                for r in rects:
                    visual_rects.append(r)
            
            # Get drawings (vector graphics)
            drawings = page.get_drawings()
            for d in drawings:
                r = d["rect"]
                # Filter out tiny specks or full page rectangles
                if r.width < 10 or r.height < 10 or (r.width > page.rect.width * 0.9 and r.height > page.rect.height * 0.9):
                    continue
                visual_rects.append(r)

            # 2. Search for "Figure" or "Fig." text blocks
            text_blocks = page.get_text("blocks")
            
            for block in text_blocks:
                text = block[4].strip()
                # Basic regex to identify figure captions
                if re.match(r'^(Figure|Fig\.)\s*\d+', text, re.IGNORECASE):
                    caption_rect = fitz.Rect(block[:4])
                    caption_text = text.replace('\n', ' ')
                    
                    # Extract the figure number for ID
                    match = re.search(r'((?:Figure|Fig\.)\s*\d+)', text, re.IGNORECASE)
                    fig_label = match.group(1).replace('.', '').replace(' ', '') if match else f"Fig_p{i+1}"
                    
                    # 3. Find visual elements associated with this caption
                    # Logic: Look for visuals directly above the caption
                    associated_rects = []
                    
                    # Define a search zone above the caption
                    # We look up until we hit text that is NOT part of the figure (heuristic)
                    # For now, let's just grab all visuals that are "close" above
                    
                    for r in visual_rects:
                        # Check if visual is above caption
                        if r.y1 <= caption_rect.y0 + 10: # Allow small overlap
                            # Check vertical proximity (e.g., within 500 units)
                            if caption_rect.y0 - r.y1 < 400:
                                # Check horizontal overlap
                                # Calculate intersection of X ranges
                                x_overlap = max(0, min(r.x1, caption_rect.x1) - max(r.x0, caption_rect.x0))
                                # If the visual overlaps horizontally with caption OR is centred similarly
                                # (Sometimes caption is narrow, figure is wide)
                                
                                # A loose check: is the visual roughly in the same column?
                                # Center of visual
                                vis_center = (r.x0 + r.x1) / 2
                                cap_center = (caption_rect.x0 + caption_rect.x1) / 2
                                
                                if abs(vis_center - cap_center) < page.rect.width / 3:
                                    associated_rects.append(r)
                    
                    final_clip_rect = None
                    
                    if associated_rects:
                        # Union of all associated visuals
                        x0 = min(r.x0 for r in associated_rects)
                        y0 = min(r.y0 for r in associated_rects)
                        x1 = max(r.x1 for r in associated_rects)
                        y1 = max(r.y1 for r in associated_rects)
                        
                        # Add some padding
                        final_clip_rect = fitz.Rect(x0 - 5, y0 - 5, x1 + 5, y1 + 5)
                        
                        # Ensure we don't go below the caption top
                        if final_clip_rect.y1 > caption_rect.y0:
                             final_clip_rect.y1 = caption_rect.y0
                    
                    else:
                        # Fallback to text-block based cropping if no visuals found (e.g. pure raster without metadata?)
                        # Or if our heuristic failed
                        print(f"  Warning: No visual objects found for {fig_label}, falling back to text heuristic.")
                        blocks_above = [b for b in text_blocks if b[3] < caption_rect.y0]
                        blocks_above.sort(key=lambda b: b[1])
                        
                        top_limit = 50
                        if blocks_above:
                            # Find the closest block above that overlaps horizontally
                            for b in reversed(blocks_above):
                                b_rect = fitz.Rect(b[:4])
                                # If block is significantly below the top of page
                                if caption_rect.y0 - b_rect.y1 > 10:
                                     # Check horizontal alignment
                                     if max(0, min(b_rect.x1, caption_rect.x1) - max(b_rect.x0, caption_rect.x0)) > 0:
                                         top_limit = b_rect.y1
                                         break
                        
                        # Determine width based on caption width (single column vs full width)
                        if caption_rect.width > page.rect.width * 0.6:
                            # Likely full width
                            x0, x1 = 0, page.rect.width
                        else:
                            # Likely column width, center around caption
                            # Estimate column width (e.g. half page)
                            col_width = page.rect.width / 2
                            if caption_rect.x0 < page.rect.width / 2:
                                x0, x1 = 0, col_width # Left column
                            else:
                                x0, x1 = col_width, page.rect.width # Right column
                                
                        final_clip_rect = fitz.Rect(x0, top_limit, x1, caption_rect.y0)

                    # Final Safety Clip
                    final_clip_rect = final_clip_rect & page.rect
                    
                    if final_clip_rect.height < 10 or final_clip_rect.width < 10:
                        print(f"  Skipping {fig_label}: Valid rect too small.")
                        continue

                    zoom = 2
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, clip=final_clip_rect)
                    
                    figure_id = f"{pdf_name}_{fig_label}"
                    output_path = os.path.join(self.output_dir, f"{figure_id}.png")
                    
                    pix.save(output_path)
                    print(f"  Saved {figure_id} to {output_path}")
                    
                    # Update metadata
                    self._update_metadata(pdf_name, figure_id, f"Figure {fig_label}", i+1, output_path, caption_text)

        doc.close()

    def extract_figure(self, pdf_path: str, search_term: str, figure_id: str = None):
        """Extracts a specific figure based on a search term."""
        doc = fitz.open(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        print(f"Searching for '{search_term}' in {pdf_name}...")
        
        found = False
        for i, page in enumerate(doc):
            text_instances = page.search_for(search_term)
            if not text_instances:
                continue
                
            caption_rect = text_instances[0]
            print(f"Found on page {i+1} at {caption_rect}")
            
            text_blocks = page.get_text("blocks")
            blocks_above = [b for b in text_blocks if b[3] < caption_rect.y0]
            blocks_above.sort(key=lambda b: b[1])
            
            top_limit = 50
            if blocks_above:
                last_block = blocks_above[-1]
                gap = caption_rect.y0 - last_block[3]
                if gap > 20:
                    top_limit = last_block[3] + 5
                else:
                    top_limit = last_block[3]
            
            zoom = 2
            mat = fitz.Matrix(zoom, zoom)
            
            if caption_rect.y0 - top_limit < 50:
                 top_limit = 0 # Fallback
            
            clip_rect = fitz.Rect(0, top_limit, page.rect.width, caption_rect.y0)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
            
            if not figure_id:
                sanitized_term = re.sub(r'[^\w\-]', '_', search_term)[:20]
                figure_id = f"{pdf_name}_{sanitized_term}"
            else:
                figure_id = f"{pdf_name}_{figure_id}"
                
            output_path = os.path.join(self.output_dir, f"{figure_id}.png")
            pix.save(output_path)
            print(f"Saved to {output_path}")
            
            # Find full caption
            full_caption = ""
            for b in text_blocks:
                if fitz.Rect(b[:4]).intersects(caption_rect):
                    full_caption = b[4].replace('\n', ' ').strip()
                    break
            
            self._update_metadata(pdf_name, figure_id, search_term, i+1, output_path, full_caption)
            found = True
            break 
            
        doc.close()
        if not found:
            print(f"Search term '{search_term}' not found in {pdf_name}")

    def _update_metadata(self, pdf_name, figure_id, search_term, page_num, output_path, caption):
        if pdf_name not in self.metadata:
            self.metadata[pdf_name] = {"pdf_path": "", "figures": []} # pdf_path updated later or assumed known
            
        existing_figs = self.metadata[pdf_name]["figures"]
        # Remove if exists to update
        self.metadata[pdf_name]["figures"] = [f for f in existing_figs if f['id'] != figure_id]
        
        self.metadata[pdf_name]["figures"].append({
            "id": figure_id,
            "search_term": search_term,
            "page": page_num,
            "output_path": output_path,
            "caption_snippet": caption[:500] + "..." if len(caption) > 500 else caption
        })
        self._save_metadata()

def main():
    parser = argparse.ArgumentParser(description="Extract figures from arXiv papers.")
    parser.add_argument("--pdf", nargs='*', help="Path(s) to local PDF")
    parser.add_argument("--arxiv", nargs='*', help="arXiv ID(s) to download")
    parser.add_argument("--term", help="Specific search term (e.g., 'Figure 2'). If omitted, extracts all figures.")
    parser.add_argument("--out", default="extracted_figures", help="Output directory")
    parser.add_argument("--id", help="Optional ID for the figure filename suffix (only used with --term)")
    
    args = parser.parse_args()
    
    # Initialize extractor with output directory
    extractor = PaperFigureExtractor(args.out)
    
    pdfs_to_process = []
    
    # Handle arXiv downloads
    if args.arxiv:
        base_dir = os.path.join(os.getcwd(), "Resource", "Downloads")
        for arxiv_id in args.arxiv:
            path = extractor.download_paper(arxiv_id, base_dir)
            if path:
                pdfs_to_process.append(path)
    
    # Handle local PDFs
    if args.pdf:
        for p in args.pdf:
            if os.path.exists(p):
                pdfs_to_process.append(p)
            else:
                print(f"File not found: {p}")
                
    if not pdfs_to_process:
        print("No valid PDFs to process. Please provide --pdf or --arxiv.")
        return

    for pdf_path in pdfs_to_process:
        if args.term:
            extractor.extract_figure(pdf_path, args.term, args.id)
        else:
            extractor.extract_all_figures(pdf_path)

if __name__ == "__main__":
    main()
