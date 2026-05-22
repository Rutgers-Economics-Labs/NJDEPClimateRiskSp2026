# Tell Cursor to run: pip install playwright playwright-stealth pandas
# Then run: playwright install

import time
import pandas as pd
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync

def scrape_emma_bonds(start_date, end_date):
    results = []
    
    with sync_playwright() as p:
        # Launch browser (keep headless=False at first so you can see if it hits a CAPTCHA)
        browser = p.chromium.launch(headless=False) 
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        # Apply stealth to bypass basic bot detection
        stealth_sync(page)
        
        try:
            # 1. Go to EMMA Advanced Search
            print(f"Navigating to EMMA for {start_date} to {end_date}...")
            page.goto("https://emma.msrb.org/Search/AdvancedSearch.aspx", timeout=60000)
            
            # Wait for the page to actually load its JavaScript
            page.wait_for_load_state("networkidle")
            
            # --- CURSOR TODO: ADD PAGE INTERACTIONS HERE ---
            # 2. Select "State" -> "NJ"
            # 3. Enter Date Range (start_date to end_date)
            # 4. Click "Run Search"
            # 5. Wait for the results table to populate
            
            # 6. Automate clicking the "Export" button IF it exists, 
            # OR parse the HTML table directly if it doesn't.
            
            time.sleep(5) # Human delay
            
        except Exception as e:
            print(f"Failed on {start_date}: {e}")
            
        finally:
            browser.close()
            
    return results

# You will need to build a loop that calls this function month-by-month
# scrape_emma_bonds("01/01/2015", "01/31/2015")