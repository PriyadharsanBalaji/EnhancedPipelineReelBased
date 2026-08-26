"""
run.py — CLI for the Manim Educational Video Pipeline.

Examples:
    # Full pipeline: PDF → 45 min video
    python run.py --pdf iemh101.pdf

    # Plan only (review storyboard before rendering)
    python run.py --pdf iemh101.pdf --plan-only

    # Resume interrupted pipeline
    python run.py --pdf iemh101.pdf --resume

    # Skip to rendering stage (stages 0-2 must be done)
    python run.py --pdf iemh101.pdf --stage 3

    # High quality render
    python run.py --pdf iemh101.pdf --quality h

    # Use specific models
    python run.py --pdf iemh101.pdf --storyboard-model gemma4:12b --manim-model maternion/manim-coder
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import run_pipeline


import json
import shutil
from stages.assembler import assemble_video

def chunk_sections(chapter_content: dict, max_chars_per_chunk: int = 25000) -> list:
    """Split sections into logical chunks based on character length."""
    sections = chapter_content.get("sections", [])
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sec in sections:
        sec_len = sec.get("content_length", len(sec.get("content", "")))
        if current_chunk and current_length + sec_len > max_chars_per_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0
        current_chunk.append(sec)
        current_length += sec_len
        
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def main():
    parser = argparse.ArgumentParser(
        description="Manim Educational Video Pipeline — PDF → 30-60 min animated video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --pdf iemh101.pdf                          Full pipeline
  python run.py --pdf iemh101.pdf --whole-book             Iteratively process whole book
  python run.py --pdf iemh101.pdf --whole-book --max-chunks 5  Limit to 5 chunks
        """,
    )

    parser.add_argument("--pdf", required=True, metavar="PATH", help="Path to educational PDF")
    parser.add_argument("--name", default=None, help="Output slug under outputs/")
    parser.add_argument("--minutes", type=int, default=45, help="Target video duration in minutes (per chunk)")
    parser.add_argument("--plan-only", action="store_true", help="Stop after storyboard generation")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing state")
    parser.add_argument("--no-resume", action="store_true", help="Ignore saved state, regenerate everything")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--quality", default="m", choices=["l", "m", "h", "k"])
    parser.add_argument("--storyboard-model", default=None)
    parser.add_argument("--manim-model", default=None)
    
    # V2 Flags
    parser.add_argument("--whole-book", action="store_true", help="Process the entire book by chunking it iteratively")
    parser.add_argument("--max-chunks", type=int, default=None, help="Maximum number of chunks to process (e.g. 5)")

    args = parser.parse_args()
    resume = not args.no_resume
    pdf_path = Path(args.pdf)
    
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)
        
    output_name = args.name if args.name else pdf_path.stem.lower().replace(" ", "_")
    base_dir = Path("outputs") / output_name

    print(f"\n{'='*60}")
    print(f"  Manim Educational Video Pipeline (V2)")
    print(f"{'='*60}")
    print(f"  PDF      : {pdf_path}")
    print(f"  Target   : {args.minutes} min/chunk")
    print(f"  Mode     : {'Whole-Book Chunking' if args.whole_book else 'Single Chapter'}")
    print(f"{'='*60}\n")

    if not args.whole_book:
        # Standard Single-Shot Pipeline
        result = run_pipeline(
            pdf_path=str(pdf_path),
            output_name=output_name,
            target_minutes=args.minutes,
            resume=resume,
            plan_only=args.plan_only,
            start_stage=args.stage,
            quality=args.quality,
            storyboard_model=args.storyboard_model,
            manim_model=args.manim_model,
        )
        print(f"\n🎬 Video ready: {result.get('output', 'unknown')}")
        return

    # V2 Iterative Whole-Book Processing
    base_dir.mkdir(parents=True, exist_ok=True)
    master_content_file = base_dir / "master_content.json"
    
    if resume and master_content_file.exists():
        print("[Chunker] Loading existing master PDF content...")
        with open(master_content_file, encoding="utf-8") as f:
            master_content = json.load(f)
    else:
        print("[Chunker] Extracting entire PDF...")
        from stages.pdf_extractor import extract_pdf
        master_content = extract_pdf(str(pdf_path), output_path=str(master_content_file))
        
    chunks = chunk_sections(master_content, max_chars_per_chunk=25000)
    print(f"\n[Chunker] Split book into {len(chunks)} chunks based on length.")
    
    if args.max_chunks and len(chunks) > args.max_chunks:
        print(f"[Chunker] Limiting to first {args.max_chunks} chunks as requested.")
        chunks = chunks[:args.max_chunks]

    chunk_videos = {}
    
    for i, chunk_sections_list in enumerate(chunks, 1):
        chunk_dir = base_dir / f"chunk_{i}"
        chunk_dir.mkdir(exist_ok=True)
        chunk_content_file = chunk_dir / "chapter_content.json"
        
        # Write isolated chapter_content.json for this chunk
        chunk_content = master_content.copy()
        chunk_content["sections"] = chunk_sections_list
        chunk_content["chapter_title"] = f"{master_content.get('chapter_title', 'Book')} - Part {i}"
        with open(chunk_content_file, "w", encoding="utf-8") as f:
            json.dump(chunk_content, f, indent=2)
            
        print(f"\n{'#'*60}")
        print(f"  Processing Chunk {i}/{len(chunks)}: {len(chunk_sections_list)} sections")
        print(f"{'#'*60}\n")
        
        result = run_pipeline(
            pdf_path=str(pdf_path), # Dummy, won't be used since chapter_content.json exists
            output_name=None,
            target_minutes=args.minutes,
            resume=resume,
            plan_only=args.plan_only,
            start_stage=1 if args.stage == 0 else args.stage, # Skip stage 0 extraction
            quality=args.quality,
            storyboard_model=args.storyboard_model,
            manim_model=args.manim_model,
            base_dir_override=str(chunk_dir),
        )
        
        if not args.plan_only and "output" in result:
            chunk_videos[i] = result["output"]
            
    if args.plan_only:
        print("\n📋 Review the storyboards in the chunk folders!")
        return
        
    if chunk_videos:
        print(f"\n{'='*60}")
        print(f"  SUPER ASSEMBLY: Stitching {len(chunk_videos)} chunks together...")
        print(f"{'='*60}")
        
        final_super_video = str(base_dir / f"THE_COMPLETE_{output_name.upper()}.mp4")
        
        # assemble_video takes a dict of scene_id -> path. We can trick it with chunk indices.
        # We also pass a dummy storyboard since add_section_cards is not used in V2 super assembly.
        assemble_video(
            rendered_files=chunk_videos,
            storyboard={}, 
            output_path=final_super_video,
        )
        
        print(f"\n🚀 SUCCESS! The complete book video is ready:")
        print(f"   {final_super_video}\n")

if __name__ == "__main__":
    main()
