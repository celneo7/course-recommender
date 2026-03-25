from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json, time

options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

course_code = 'cs2040'

driver.get(f"https://nusmods.com/courses/{course_code}")

wait = WebDriverWait(driver, 20)

# Wait for Disqus iframe
wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "iframe[id^='dsq-app']")))
iframe = driver.find_element(By.CSS_SELECTOR, "iframe[id^='dsq-app']")
driver.switch_to.frame(iframe)

# Wait for comments inside iframe
wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.post")))

# Keep scrolling + clicking until post count stops growing
prev_count = 0

while True:  # stop after 3 rounds with no new posts
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
    print(f"Posts loaded: {current_count}")

    if current_count == prev_count:
        break # no more new posts

    prev_count = current_count

print("Done loading all comments.")

# --- Extract all posts (including deeply nested replies) ---
posts = driver.find_elements(By.CSS_SELECTOR, "li.post")
results = []

for post in posts:
    post_id = post.get_attribute("id")  # e.g. "post-3071909979"

    # Skip deleted posts
    if "minimized" in (post.get_attribute("class") or ""):
        continue

    try:
        author = post.find_element(By.CSS_SELECTOR, "span.author a[data-username]").text.strip()
    except:
        author = "Unknown"

    try:
        message = post.find_element(By.CSS_SELECTOR, "div.post-message").text.strip()
    except:
        message = ""

    try:
        date = post.find_element(By.CSS_SELECTOR, "a.time-ago").get_attribute("title")
    except:
        date = ""

    try:
        likes = post.find_element(By.CSS_SELECTOR, "span[data-role='likes']").text.strip()
    except:
        likes = "0"

    try:
        dislikes = post.find_element(By.CSS_SELECTOR, "span[data-role='dislikes']").text.strip()
    except:
        dislikes = "0"

    # Reply-to: get parent author name and parent comment ID from href
    try:
        # Only look inside THIS post's own post-body, not nested children
        post_body = post.find_element(By.CSS_SELECTOR, ":scope > div.post-content > div.post-body")
        parent_link = post_body.find_element(By.CSS_SELECTOR, "a[data-role='parent-link']")
        reply_to_author = parent_link.text.strip()
        reply_to_id = parent_link.get_attribute("href").split("#comment-")[-1]
    except:
        reply_to_author = None
        reply_to_id = None

    results.append({
        "post_id": post_id.replace("post-", "") if post_id else "",
        "author": author,
        "message": message,
        "date": date,
        "likes": likes,
        "dislikes": dislikes,
        "reply_to_author": reply_to_author,   # None if top-level
        "reply_to_post_id": reply_to_id,       # None if top-level
    })

driver.quit()

print(f"\nTotal scraped: {len(results)}")

with open(f"{course_code}_reviews.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved to .json")
