from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder
import pandas as pd
import io
import os
import re
import time
import random
from typing import Iterable, List, Union, Optional

EXPECTED_HEADER = "local_authority_district,news_url,brief_description,referenced_place,source_id"

# ---------------- CSV cleaning / parsing ----------------

def clean_csv_text(text: str) -> str:
    # keep only CSV between ```csv ... ``` or generic ``` ... ```
    t = text.strip()
    if "```csv" in t and "```" in t:
        start = t.find("```csv") + len("```csv")
        end = t.rfind("```")
        t = t[start:end].strip()
    elif t.startswith("```") and t.endswith("```"):
        # fenced but not csv-tagged
        t = t.strip("`").strip()
    return t

def parse_lenient_csv(csv_text: str) -> pd.DataFrame:
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("No content")

    header = lines[0].strip()
    if header.lower() != EXPECTED_HEADER.lower():
        raise ValueError(f"Header mismatch. Got: {header}")

    rows = []
    skipped = 0
    # For each data line:
    for i, ln in enumerate(lines[1:], start=2):
        try:
            # Strategy:
            #   [LAD],[URL],[DESCRIPTION],[PLACE],[SOURCE]
            # Split off the last two fields from the right (PLACE,SOURCE),
            # then split the remaining left part into first two fields (LAD,URL),
            # and everything else in the middle remains DESCRIPTION (commas allowed).
            left, referenced_place, source_id = ln.rsplit(",", 2)
            local_authority_district, news_url, brief_description = left.split(",", 2)
            rows.append({
                "local_authority_district": local_authority_district.strip(),
                "news_url": news_url.strip(),
                "brief_description": brief_description.strip(),
                "referenced_place": referenced_place.strip(),
                "source_id": source_id.strip(),
            })
        except Exception:
            skipped += 1
            continue

    if not rows:
        raise ValueError("All lines failed to parse")
    df = pd.DataFrame(rows, columns=[
        "local_authority_district","news_url","brief_description","referenced_place","source_id"
    ])
    if skipped:
        print(f"Parsed {len(df)} rows, skipped {skipped} malformed line(s).")
    return df

# ---------------- Rate limit helpers ----------------

RATE_LIMIT_REGEX = re.compile(r"Try again in (\d+)\s*seconds?", re.IGNORECASE)

def _sleep_from_error(err_msg: str, fallback_seconds: float = 10.0):
    """
    Parse 'Try again in N seconds' from the error message and sleep that long (+ jitter).
    """
    m = RATE_LIMIT_REGEX.search(err_msg or "")
    seconds = float(m.group(1)) if m else fallback_seconds
    # Add small jitter to avoid thundering herds
    jitter = random.uniform(0.3, 1.2)
    sleep_for = max(0.0, seconds) + jitter
    print(f"Rate limit: sleeping {sleep_for:.1f}s (hint: {seconds}s).")
    time.sleep(sleep_for)

def _should_retry_rate_limit(error_dict: Optional[dict]) -> bool:
    if not error_dict:
        return False
    code = error_dict.get("code") or error_dict.get("error", {}).get("code")
    return str(code).lower() == "rate_limit_exceeded"

def _get_error_message(error_dict: Optional[dict]) -> str:
    if not error_dict:
        return ""
    return (
        error_dict.get("message") or
        error_dict.get("error", {}).get("message") or
        str(error_dict)
    )

# ---------------- SDK wrappers with retries ----------------

def create_thread_with_retry(project: AIProjectClient, max_retries: int = 3) -> Optional[object]:
    attempt = 0
    while attempt <= max_retries:
        try:
            return project.agents.threads.create()
        except Exception as e:
            msg = str(e)
            # If SDK raises a http 429/limit here, we still parse the message
            if "rate limit" in msg.lower():
                _sleep_from_error(msg, fallback_seconds=8.0)
                attempt += 1
                continue
            # Other errors: bubble up
            raise
    return None

def run_with_retry(project: AIProjectClient, thread_id: str, agent_id: str,
                   wait_seconds: int = 300, max_retries: int = 3):
    """
    Start a run and wait for completion. Retry on rate limits. Returns the final run object.
    """
    attempt = 0
    while attempt <= max_retries:
        # Start
        run = project.agents.runs.create(thread_id=thread_id, agent_id=agent_id)

        # Poll with exponential backoff
        start = time.time()
        delay = 1.0
        while time.time() - start < wait_seconds:
            run = project.agents.runs.get(thread_id=thread_id, run_id=run.id)
            if run.status in ("completed", "failed", "incomplete"):
                break
            time.sleep(delay)
            delay = min(delay * 1.5, 6.0)

        if run.status == "completed":
            return run

        # If failed due to rate limit, sleep per hint & retry
        if run.status == "failed" and _should_retry_rate_limit(getattr(run, "last_error", None)):
            err_msg = _get_error_message(getattr(run, "last_error", None))
            print(f"Run failed due to rate limit: {err_msg}")
            _sleep_from_error(err_msg, fallback_seconds=10.0)
            attempt += 1
            continue

        # If incomplete, try a limited number of retries with a small rest
        if run.status == "incomplete" and attempt < max_retries:
            print("Run incomplete. Retrying after short delay...")
            time.sleep(3.0 + attempt)
            attempt += 1
            continue

        # Other failure or exhausted retries
        return run

    return run  # last seen run

def last_assistant_text(project: AIProjectClient, thread_id: str) -> Optional[str]:
    msgs = project.agents.messages.list(thread_id=thread_id, order=ListSortOrder.DESCENDING)
    for m in msgs:
        if m.role == "assistant" and getattr(m, "text_messages", None):
            return m.text_messages[-1].text.value
    return None

# ---------------- Main runner ----------------

def run_agent(lad_names: Union[str, Iterable[str]],
              inter_lad_sleep: float = 2.0,
              thread_max_retries: int = 3,
              run_max_retries: int = 3):
    # Accept a single LAD string or a list/iterable
    if isinstance(lad_names, str):
        lad_list: List[str] = [lad_names]
    else:
        lad_list = list(lad_names)

    project = AIProjectClient(
        credential=DefaultAzureCredential(),
        endpoint="https://shared-ai-hub-foundry.services.ai.azure.com/api/projects/Team-15-Communities"
    )

    agent = project.agents.get_agent("asst_MKXJWz6odu2T5o8yVuRPonxB")

    all_frames = []

    for lad_name in lad_list:
        # ---- Create thread (with retry on 429) ----
        thread = create_thread_with_retry(project, max_retries=thread_max_retries)
        if not thread:
            print(f"[{lad_name}] Failed to create thread after retries.")
            time.sleep(inter_lad_sleep)
            continue

        print(f"[{lad_name}] Created thread: {thread.id}")

        # ---- Your prompt (unchanged style) ----
        project.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content=(
                f"""Dont ask questions and only output the finished csv. Scan as many recent local news articles that are less than a week old for this local authority in the UK as you can: {lad_name} and ourput a csv according to this template:
                local_authority_district,news_url,brief_description,referenced_place,source_id
    Barnsley Borough Council,https://www.bury.gov.uk/my-neighbourhood/safety-in-the-community/hate-crime/greater-manchester-hate-crime-awareness-week-2023,Barnsley (via local partners) runs hate crime awareness and community safety work as part of Greater Manchester campaigns to encourage reporting and cohesion.,Barnsley,turn1search16
    Birmingham City Council,https://www.birmingham-community-safety-partnership.co.uk/get-help/hate-crime/,Birmingham's Community Safety Partnership describes hate-crime prevention work and resources to support vulnerable communities and improve cohesion.,Birmingham,turn0search13
    Bolton Borough Council,https://www.bolton.gov.uk/news/article/1847/bolton-shows-solidarity-against-hate-crime-in-annual-awareness-week,Bolton Council and partners run Greater Manchester Hate Crime Awareness Week activities and fund community groups to support victims and cohesion.,Bolton,turn0search6

                Include the source area of the article and also the area referenced in the article and be very exact, dont make up anything."""
            )
        )

        # ---- Run with retries / backoff on rate limit ----
        run = run_with_retry(
            project=project,
            thread_id=thread.id,
            agent_id=agent.id,
            wait_seconds=300,
            max_retries=run_max_retries
        )

        if run.status == "failed":
            err = getattr(run, "last_error", None)
            print(f"[{lad_name}] Run FAILED. Error: {err}")
        elif run.status == "incomplete":
            print(f"[{lad_name}] Run INCOMPLETE. Trying to salvage any output...")
        elif run.status == "completed":
            print(f"[{lad_name}] Run COMPLETED.")

        # ---- Try to salvage/parse output regardless (sometimes there is partial text) ----
        assistant_text = last_assistant_text(project, thread.id)
        if not assistant_text:
            print(f"[{lad_name}] No assistant text found.")
            time.sleep(inter_lad_sleep)
            continue

        assistant_text = clean_csv_text(assistant_text).strip()

        try:
            df = parse_lenient_csv(assistant_text)
            df.insert(0, "query_lad", lad_name)
            all_frames.append(df)
            print(f"[{lad_name}] Parsed {len(df)} rows.")
        except Exception as e:
            print(f"[{lad_name}] CSV parse failed: {e}")

        # ---- Be gentle between LADs ----
        time.sleep(inter_lad_sleep)

    if all_frames:
        result = pd.concat(all_frames, ignore_index=True)
        print(f"Total rows: {len(result)}")

        # ensure output folder
        os.makedirs("data", exist_ok=True)

        timedatestamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = f"data/lad_news_agentic_{timedatestamp}.csv"
        result.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Saved: {out_path}")
        return result
    else:
        print("No results produced.")
        return None


if __name__ == "__main__":
    # Example usage
    run_agent(["Barnsley", "Bolton", "Birmingham"], inter_lad_sleep=2.5)
