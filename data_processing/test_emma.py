from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
import time

def test():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        # Apply stealth correctly
        stealth = Stealth()
        stealth.apply_stealth_sync(page)
        
        print("Navigating to EMMA...")
        page.goto("https://emma.msrb.org/Search/AdvancedSearch.aspx", timeout=60000)
        page.wait_for_load_state("networkidle")
        print("Page title:", page.title())
        html = page.content()
        with open("emma.html", "w") as f:
            f.write(html)
        browser.close()

if __name__ == "__main__":
    test()
