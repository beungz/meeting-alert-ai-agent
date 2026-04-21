"""
transcript_processing.py — Utilities for converting and labeling transcript files.

Two-step pipeline:
  1. parse_aligned_nlp / process_aligned_files  — convert .aligned.nlp files to
     15-second chunked JSON (*_parsed.json)
  2. auto_label_json_files                       — scan each chunk for target
     finance keywords and write *_labeled.json

Designed to be run standalone (see __main__ block) or imported by the notebook.

Input transcript format
-----------------------
  The .aligned.nlp files are derived from the earnings22 dataset:
    https://github.com/revdotcom/speech-datasets
  (Subset: English language, North American dialect.)
  Full credit to Rev.com, Inc. Used here for non-commercial, educational research.
"""

import glob
import json
import os
import re

def parse_aligned_nlp(filepath, chunk_duration=15.0):
    """Parse a .aligned.nlp transcript file into fixed-duration text chunks.

    Each row in the file is a pipe-delimited token with a timestamp.  Tokens
    are accumulated until their timestamp crosses a chunk boundary, at which
    point the chunk is saved and a new one begins.

    Parameters
    ----------
    filepath       : path to the .aligned.nlp file
    chunk_duration : length of each output chunk in seconds (default 15.0)

    Returns
    -------
    list of dicts with keys: start_time, end_time, text, topic_present (always 0)
    """
    chunks = []
    current_text = []
    chunk_end = chunk_duration

    with open(filepath, "r", encoding="utf-8") as f:
        next(f)  # skip header line

        for line in f:
            parts = line.split("|")
            if len(parts) < 5:
                continue

            token  = parts[0].strip()
            ts_str = parts[2].strip()
            punct  = parts[4].strip()

            # Tokens without a timestamp are attached to the current chunk
            if not ts_str:
                current_text.append(token + punct)
                continue

            start_time = float(ts_str)

            # The while loop handles long silent stretches where a single token
            # may need to close multiple empty chunks before it belongs to a new one
            while start_time >= chunk_end:
                chunk_str = (
                    " ".join(current_text)
                    .replace(" .", ".")
                    .replace(" ,", ",")
                    .strip()
                )
                chunks.append({
                    "start_time":    chunk_end - chunk_duration,
                    "end_time":      chunk_end,
                    "text":          chunk_str,
                    "topic_present": 0,
                })
                current_text = []
                chunk_end += chunk_duration

            current_text.append(token + punct)

    # Flush any remaining tokens at the end of the file
    if current_text:
        chunk_str = (
            " ".join(current_text)
            .replace(" .", ".")
            .replace(" ,", ",")
            .strip()
        )
        chunks.append({
            "start_time":    chunk_end - chunk_duration,
            "end_time":      chunk_end,
            "text":          chunk_str,
            "topic_present": 0,
        })

    return chunks


def process_aligned_files(folder_path="."):
    """Convert all .aligned.nlp files in a folder to chunked JSON.

    For each *.aligned.nlp file found, calls parse_aligned_nlp and writes
    the result to <base>_parsed.json in the same folder.

    Parameters
    ----------
    folder_path : directory to search (default: current working directory)
    """
    nlp_files = glob.glob(os.path.join(folder_path, "*.aligned.nlp"))

    if not nlp_files:
        print(f"No .aligned.nlp files found in '{folder_path}'.")
        return

    for nlp_file in nlp_files:
        base_name     = os.path.basename(nlp_file).replace(".aligned.nlp", "")
        json_filepath = os.path.join(folder_path, f"{base_name}_parsed.json")

        chunked_data = parse_aligned_nlp(nlp_file)

        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(chunked_data, f, indent=4)

        print(f"Converted: {base_name} -> {json_filepath}")


def auto_label_json_files(folder_path, keywords):
    """Label each chunk in *_parsed.json files with topic_present=1 or 0.

    Scans for any of the given keywords (exact phrase, case-insensitive) in
    each chunk's text field, then writes *_labeled.json alongside the
    originals so the parsed files are preserved.

    Parameters
    ----------
    folder_path : directory containing *_parsed.json files
    keywords    : list of keyword/phrase strings to search for
    """
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(kw) for kw in keywords) + r")\b",
        re.IGNORECASE,
    )

    json_files = glob.glob(os.path.join(folder_path, "*_parsed.json"))

    if not json_files:
        print(f"No _parsed.json files found in '{folder_path}'.")
        return

    print(f"Found {len(json_files)} files in '{folder_path}'. Starting auto-labeling...\n")

    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        updated_chunks = 0
        for chunk in data:
            if pattern.search(chunk.get("text", "")):
                chunk["topic_present"] = 1
                updated_chunks += 1
            else:
                chunk["topic_present"] = 0

        # Save to a new file so the original _parsed.json is preserved
        output_file = file.replace("_parsed.json", "_labeled.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"Saved {os.path.basename(output_file)}")
        print(f"   -> Topics detected in {updated_chunks} out of {len(data)} chunks.")

    print("\nAll files labeled successfully.")


if __name__ == "__main__":
    _topics = [
        "dividend", "gross margin", "inventory", "capital allocation",
        "net income", "operating income", "pricing", "capex",
        "investment portfolio", "sales growth",
    ]
    process_aligned_files("testset_transcript")
    auto_label_json_files("testset_transcript", keywords=_topics)
