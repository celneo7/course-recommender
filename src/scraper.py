"""
NUSMods Course Review Scraper + Cross-Course Profiler
=====================================================
Two-phase scraper:
  Phase 1: Scrape reviews from NUSMods course pages (Selenium)
  Phase 2: For each unique author, fetch their other reviews via Disqus API

Prerequisites:
  pip install selenium
  Chrome + ChromeDriver installed

Setup:
  1. Register a Disqus API application at https://disqus.com/api/applications/
  2. Copy your Public Key
  3. Pass it via --api-key or set DISQUS_API_KEY environment variable

Usage:
  # Scrape a single course
  python scraper.py CS2040 --api-key YOUR_KEY

  # Scrape multiple courses
  python scraper.py CS2040 CS1101S CS2030S --api-key YOUR_KEY

  # Scrape without API (no cross-course profiling, Selenium only)
  python scraper.py CS2040 --no-api

Output:
  data/
    reviews/
      CS2040_reviews.json      — all reviews for CS2040
      CS1101S_reviews.json     — all reviews for CS1101S
      ...
    profiles/
      author_profiles.json     — cross-course review history per author
    combined/
      all_reviews.json         — all reviews across all courses, deduplicated
      author_course_matrix.json — which authors reviewed which courses
"""

import json
import time
import os
import re
import argparse
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Phase 1: Selenium Scraper (per course page)
# ---------------------------------------------------------------------------

def scrape_course_reviews(course_code, headless=True):
    """
    Scrape all reviews from a NUSMods course page via Selenium.
    Returns list of review dicts.
    """
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    url = f"https://nusmods.com/courses/{course_code.upper()}"
    print(f"\n[Phase 1] Scraping {course_code} from {url}")
    driver.get(url)

    wait = WebDriverWait(driver, 20)

    try:
        # Wait for Disqus iframe
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "iframe[id^='dsq-app']")))
        iframe = driver.find_element(By.CSS_SELECTOR, "iframe[id^='dsq-app']")
        driver.switch_to.frame(iframe)

        # Wait for comments inside iframe
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.post")))

        # Keep scrolling + clicking until post count stops growing
        prev_count = 0
        stale_rounds = 0
        while stale_rounds < 3:
            # Click all visible "Show more replies" buttons
            show_more_buttons = driver.find_elements(By.CSS_SELECTOR, "a[data-action='show-children']")
            for btn in show_more_buttons:
                try:
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(0.3)
                except:
                    pass

            # Click "Load more comments" at bottom if present
            try:
                load_more = driver.find_element(By.CSS_SELECTOR, "a[data-action='load-more']")
                driver.execute_script("arguments[0].click();", load_more)
                time.sleep(2)
            except:
                pass

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)

            current_count = len(driver.find_elements(By.CSS_SELECTOR, "li.post:not(.minimized)"))
            print(f"  Posts loaded: {current_count}")

            if current_count == prev_count:
                stale_rounds += 1
            else:
                stale_rounds = 0
            prev_count = current_count

        print(f"  Done loading. Total visible posts: {prev_count}")

        # --- Extract all posts ---
        posts = driver.find_elements(By.CSS_SELECTOR, "li.post")
        results = []

        for post in posts:
            post_id = post.get_attribute("id")  # e.g. "post-3071909979"

            # Skip deleted/minimized posts
            if "minimized" in (post.get_attribute("class") or ""):
                continue

            # Author name
            try:
                author = post.find_element(By.CSS_SELECTOR, "span.author a[data-username]").text.strip()
            except:
                author = "Unknown"

            # Author username (Disqus username — needed for API profile lookup)
            try:
                username = post.find_element(By.CSS_SELECTOR, "span.author a[data-username]").get_attribute("data-username")
            except:
                username = None

            # Author profile URL
            try:
                profile_url = post.find_element(By.CSS_SELECTOR, "span.author a[data-username]").get_attribute("href")
            except:
                profile_url = None

            # Message
            try:
                message = post.find_element(By.CSS_SELECTOR, "div.post-message").text.strip()
            except:
                message = ""

            # Date
            try:
                date = post.find_element(By.CSS_SELECTOR, "a.time-ago").get_attribute("title")
            except:
                date = ""

            # Likes / dislikes
            try:
                likes = post.find_element(By.CSS_SELECTOR, "span[data-role='likes']").text.strip()
            except:
                likes = "0"
            try:
                dislikes = post.find_element(By.CSS_SELECTOR, "span[data-role='dislikes']").text.strip()
            except:
                dislikes = "0"

            # Reply-to info
            try:
                post_body = post.find_element(By.CSS_SELECTOR, ":scope > div.post-content > div.post-body")
                parent_link = post_body.find_element(By.CSS_SELECTOR, "a[data-role='parent-link']")
                reply_to_author = parent_link.text.strip()
                reply_to_id = parent_link.get_attribute("href").split("#comment-")[-1]
            except:
                reply_to_author = None
                reply_to_id = None

            results.append({
                "course_code": course_code.upper(),
                "post_id": post_id.replace("post-", "") if post_id else "",
                "author": author,
                "author_username": username,
                "author_profile_url": profile_url,
                "message": message,
                "date": date,
                "likes": likes,
                "dislikes": dislikes,
                "reply_to_author": reply_to_author,
                "reply_to_post_id": reply_to_id,
            })

    except Exception as e:
        print(f"  ERROR scraping {course_code}: {e}")
        results = []
    finally:
        driver.quit()

    print(f"  Scraped {len(results)} posts for {course_code}")
    return results


# ---------------------------------------------------------------------------
# Phase 2: Disqus API — Fetch user's cross-course reviews
# ---------------------------------------------------------------------------

def fetch_user_posts_api(username, api_key, forum="nusmods", max_pages=10):
    """
    Fetch all posts by a user on the NUSMods Disqus forum using the API.
    Returns list of post dicts with thread (course) info.

    API endpoint: https://disqus.com/api/3.0/users/listPosts.json
    """
    import urllib.request
    import urllib.parse

    all_posts = []
    cursor = None

    for page in range(max_pages):
        
        params = {
            "api_secret": api_key,
            "limit": 100,
            "related": "thread",
            "include": "approved",
        }
        if cursor:
            params["cursor"] = cursor

        url = "https://disqus.com/api/3.0/users/listPosts.json?" + urllib.parse.urlencode(params) + "&user:username=" + urllib.parse.quote(username)

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"    API error for {username}: {e}")
            break

        if data.get("code") != 0:
            print(f"    API returned error code {data.get('code')} for {username}")
            break

        posts = data.get("response", [])
        for post in posts:
            # Extract course code from thread URL, fall back to thread title
            thread = post.get("thread", {})
            thread_link = thread.get("link", "") if isinstance(thread, dict) else ""
            thread_title = thread.get("title", "") if isinstance(thread, dict) else ""
            course_code = extract_course_from_url(thread_link)
            if course_code == "UNKNOWN" and thread_title:
                course_code = extract_course_from_title(thread_title) or "UNKNOWN"

            # Filter: only keep posts from nusmods (check thread link or forum field)
            post_forum = post.get("forum", "")
            # forum might be a string or a dict
            if isinstance(post_forum, dict):
                post_forum = post_forum.get("id", "") or post_forum.get("shortname", "")
            # Skip posts clearly from other forums
            if post_forum and post_forum != "nusmods" and "nusmods" not in thread_link:
                continue

            raw_msg = post.get("raw_message", post.get("message", ""))

            all_posts.append({
                "post_id": str(post.get("id", "")),
                "author": post.get("author", {}).get("name", "Unknown"),
                "author_username": post.get("author", {}).get("username", ""),
                "message": clean_message(raw_msg),
                "date": post.get("createdAt", ""),
                "likes": str(post.get("likes", 0)),
                "dislikes": str(post.get("dislikes", 0)),
                "course_code": course_code,
                "thread_title": thread_title,
                "thread_link": thread_link,
                "parent": post.get("parent"),
            })

        # Pagination
        cursor_info = data.get("cursor", {})
        if cursor_info.get("hasNext"):
            cursor = cursor_info.get("next")
            time.sleep(0.5)  # Rate limiting
        else:
            break

    return all_posts


def fetch_user_posts_selenium(username, driver=None):
    """
    Fallback: Scrape user's profile page via Selenium if no API key.
    Slower and less reliable than the API approach.
    """
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    own_driver = False
    if driver is None:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=options)
        own_driver = True

    profile_url = f"https://disqus.com/by/{username}/"
    print(f"    Scraping profile: {profile_url}")
    driver.get(profile_url)

    results = []
    try:
        wait = WebDriverWait(driver, 10)
        # Wait for comments to load (profile pages are JS-rendered)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.comment")))

        # Scroll to load more
        prev_count = 0
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            current = len(driver.find_elements(By.CSS_SELECTOR, "li.comment"))
            if current == prev_count:
                break
            prev_count = current

        comments = driver.find_elements(By.CSS_SELECTOR, "li.comment")
        for comment in comments:
            try:
                message = comment.find_element(By.CSS_SELECTOR, ".comment-body").text.strip()
            except:
                message = ""
            try:
                thread_link_el = comment.find_element(By.CSS_SELECTOR, "a.comment-thread-link")
                thread_link = thread_link_el.get_attribute("href")
                thread_title = thread_link_el.text.strip()
            except:
                thread_link = ""
                thread_title = ""
            try:
                date = comment.find_element(By.CSS_SELECTOR, "time").get_attribute("datetime")
            except:
                date = ""

            course_code = extract_course_from_url(thread_link)

            results.append({
                "author_username": username,
                "message": message,
                "date": date,
                "course_code": course_code,
                "thread_title": thread_title,
                "thread_link": thread_link,
            })
    except Exception as e:
        print(f"    Could not scrape profile for {username}: {e}")
    finally:
        if own_driver:
            driver.quit()

    return results


def extract_course_from_url(url):
    """Extract NUS course code from a NUSMods URL."""
    # Handles both:
    #   https://nusmods.com/courses/CS2040/...
    #   https://nusmods.com/modules/CFG1002/reviews
    match = re.search(r'nusmods\.com/(?:courses|modules)/([A-Z]{2,4}\d{4}[A-Z]?)', url, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "UNKNOWN"


def extract_course_from_title(title):
    """Extract NUS course code from a thread title string."""
    # e.g. "CFG1002 Career Catalyst" or "CS2040 Data Structures and Algorithms"
    match = re.search(r'\b([A-Z]{2,4}\d{4}[A-Z]?)\b', title)
    if match:
        return match.group(1).upper()
    return None


def clean_message(text):
    """Remove weird unicode line separators and normalize whitespace."""
    if not text:
        return text
    # Remove Unicode line/paragraph separators
    text = text.replace('\u2028', '\n').replace('\u2029', '\n')
    # Remove other common invisible/control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Normalize multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Phase 3: Build profiles and combined dataset
# ---------------------------------------------------------------------------

def build_author_profiles(all_reviews, cross_course_posts):
    """
    Combine scraped reviews with cross-course API data to build author profiles.

    Returns:
      author_profiles: dict keyed by username
      author_course_matrix: which authors reviewed which courses
    """
    profiles = defaultdict(lambda: {
        "username": "",
        "display_name": "",
        "courses_reviewed": [],
        "reviews": [],
        "total_reviews": 0,
    })

    # Add reviews from Phase 1 (course page scraping)
    for review in all_reviews:
        uname = review.get("author_username") or review.get("author", "Unknown")
        profiles[uname]["username"] = uname
        profiles[uname]["display_name"] = review.get("author", uname)
        course = review["course_code"]
        if course not in profiles[uname]["courses_reviewed"]:
            profiles[uname]["courses_reviewed"].append(course)
        profiles[uname]["reviews"].append({
            "course_code": course,
            "post_id": review["post_id"],
            "message": review["message"],
            "date": review["date"],
            "likes": review["likes"],
            "dislikes": review["dislikes"],
            "source": "course_page_scrape",
        })

    # Add cross-course posts from Phase 2 (API or profile scrape)
    for post in cross_course_posts:
        uname = post.get("author_username", "Unknown")
        profiles[uname]["username"] = uname
        if post.get("author"):
            profiles[uname]["display_name"] = post["author"]
        course = post.get("course_code", "UNKNOWN")
        if course not in profiles[uname]["courses_reviewed"]:
            profiles[uname]["courses_reviewed"].append(course)

        # Avoid duplicates (same post_id)
        existing_ids = {r["post_id"] for r in profiles[uname]["reviews"] if "post_id" in r}
        if post.get("post_id") and post["post_id"] not in existing_ids:
            profiles[uname]["reviews"].append({
                "course_code": course,
                "post_id": post.get("post_id", ""),
                "message": post["message"],
                "date": post["date"],
                "likes": post.get("likes", "0"),
                "dislikes": post.get("dislikes", "0"),
                "source": "api_profile_fetch",
            })

    # Finalise
    for uname in profiles:
        profiles[uname]["total_reviews"] = len(profiles[uname]["reviews"])

    # Build course matrix
    matrix = {}
    for uname, profile in profiles.items():
        matrix[uname] = {
            "display_name": profile["display_name"],
            "courses": profile["courses_reviewed"],
            "num_courses": len(profile["courses_reviewed"]),
            "num_reviews": profile["total_reviews"],
        }

    return dict(profiles), matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NUSMods Review Scraper + Cross-Course Profiler")
    parser.add_argument("courses", nargs="+", help="Course codes to scrape (e.g., CS2040 CS1101S)")
    parser.add_argument("--api-key", default=os.environ.get("DISQUS_API_KEY"), help="Disqus API public key")
    parser.add_argument("--no-api", action="store_true", help="Skip API-based profile fetching")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window")
    parser.add_argument("--forum", default="nusmods", help="Disqus forum shortname (default: nusmods)")
    parser.add_argument("--output-dir", default="data", help="Output directory (default: data)")
    args = parser.parse_args()

    # Create output directories
    os.makedirs(f"{args.output_dir}/reviews", exist_ok=True)
    os.makedirs(f"{args.output_dir}/profiles", exist_ok=True)
    os.makedirs(f"{args.output_dir}/combined", exist_ok=True)

    # ---- Phase 1: Scrape each course page ----
    print("=" * 60)
    print("PHASE 1: Scraping course review pages")
    print("=" * 60)

    all_reviews = []
    for course in args.courses:
        #reviews = scrape_course_reviews(course, headless=not args.no_headless)

        # Save per-course file
        out_path = f"{args.output_dir}/reviews/{course.upper()}_reviews.json"
        with open(out_path, "r", encoding="utf-8") as f:
            reviews = json.load(f)
        print(f"  Loaded {len(reviews)} reviews from {out_path}")

        # with open(out_path, "w", encoding="utf-8") as f:
        #     json.dump(reviews, f, indent=2, ensure_ascii=False)
        # print(f"  Saved {len(reviews)} reviews to {out_path}")

        all_reviews.extend(reviews)

    # Collect unique usernames (excluding Unknown/None)
    unique_authors = set()
    for r in all_reviews:
        uname = r.get("author_username")
        if uname and uname != "Unknown":
            unique_authors.add(uname)

    print(f"\nFound {len(unique_authors)} unique authors across {len(args.courses)} course(s)")

    # ---- Phase 2: Fetch cross-course profiles ----
    print("\n" + "=" * 60)
    print("PHASE 2: Fetching cross-course profiles")
    print("=" * 60)

    cross_course_posts = []

    if not args.no_api and args.api_key:
        print(f"Using Disqus API (forum: {args.forum})")
        for i, username in enumerate(sorted(unique_authors)):
            print(f"  [{i+1}/{len(unique_authors)}] Fetching posts for: {username}")
            posts = fetch_user_posts_api(username, args.api_key, forum=args.forum)
            cross_course_posts.extend(posts)
            print(f"    Found {len(posts)} posts across NUSMods")
            time.sleep(0.5)  # Rate limiting

    elif not args.no_api and not args.api_key:
        print("WARNING: No Disqus API key provided.")
        print("  To enable cross-course profiling, either:")
        print("  1. Pass --api-key YOUR_KEY")
        print("  2. Set DISQUS_API_KEY environment variable")
        print("  3. Register at https://disqus.com/api/applications/")
        print("\n  Falling back to Selenium profile scraping (slower)...")

        from selenium import webdriver
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=options)

        for i, username in enumerate(sorted(unique_authors)):
            print(f"  [{i+1}/{len(unique_authors)}] Scraping profile: {username}")
            posts = fetch_user_posts_selenium(username, driver)
            cross_course_posts.extend(posts)
            print(f"    Found {len(posts)} posts")
            time.sleep(1)

        driver.quit()
    else:
        print("Skipping cross-course profiling (--no-api flag)")

    # ---- Phase 3: Build profiles and combined dataset ----
    print("\n" + "=" * 60)
    print("PHASE 3: Building profiles and combined dataset")
    print("=" * 60)

    profiles, matrix = build_author_profiles(all_reviews, cross_course_posts)

    # Save profiles
    profiles_path = f"{args.output_dir}/profiles/author_profiles.json"
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(profiles)} author profiles to {profiles_path}")

    # Save course matrix
    matrix_path = f"{args.output_dir}/combined/author_course_matrix.json"
    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)
    print(f"  Saved author-course matrix to {matrix_path}")

    # Save combined reviews (all courses, deduplicated)
    seen_ids = set()
    combined = []
    for r in all_reviews:
        if r["post_id"] not in seen_ids:
            seen_ids.add(r["post_id"])
            combined.append(r)
    for p in cross_course_posts:
        pid = p.get("post_id", "")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            combined.append(p)

    combined_path = f"{args.output_dir}/combined/all_reviews.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(combined)} combined reviews to {combined_path}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Courses scraped: {', '.join(c.upper() for c in args.courses)}")
    print(f"  Total course page reviews: {len(all_reviews)}")
    print(f"  Unique authors: {len(unique_authors)}")
    print(f"  Cross-course posts fetched: {len(cross_course_posts)}")

    # Show top cross-course reviewers
    multi_course = {k: v for k, v in matrix.items() if v["num_courses"] > 1}
    if multi_course:
        print(f"\n  Authors who reviewed multiple courses ({len(multi_course)}):")
        for uname, info in sorted(multi_course.items(), key=lambda x: x[1]["num_courses"], reverse=True)[:10]:
            print(f"    {info['display_name']} ({uname}): {', '.join(info['courses'])} — {info['num_reviews']} total reviews")

    print(f"\n  Output directory: {args.output_dir}/")
    print("  Done!")


if __name__ == "__main__":
    main()