import csv
import time
import random
from atproto import Client, models

# --- Configuration ---
# You MUST replace these placeholders with your actual Bluesky credentials
BLUESKY_HANDLE = 'redinjapanese.bsky.social'
BLUESKY_PASSWORD = 'ztfds9vu' 
OUTPUT_FILENAME = 'real_posts_for_labeling.csv'

# Define the categories and the search queries you will run.
# You will manually assign the final label (Positive, Hard Negative, Borderline) 
# in the CSV based on manual review.
SEARCH_QUERIES = [
    # Positive Misinformation (Target ~90 posts)
    ('Positive', 'cure cancer AND miracle'),
    ('Positive', 'reverse diabetes AND supplement'),
    ('Positive', '72 hour dry fast cures'),
    ('Positive', 'stop insulin AND cinnamon'),
    ('Positive', 'NMN miracle cure'),
    # Hard Negative (Target ~60 posts)
    ('Hard Negative', 'debunked OR misinformation AND weight loss'),
    ('Hard Negative', 'scam OR fake OR dangerous AND supplement'),
    ('Hard Negative', 'talk to your doctor OR consult physician'),
    # Borderline (Target ~30 posts)
    ('Borderline', 'thinking of trying fasting OR not sure if safe peptide'),
    ('Borderline', 'is this effective OR potential side effects OR maybe works'),
]

def authenticate_client():
    """Initializes and authenticates the Bluesky client."""
    client = Client()
    try:
        print(f"Attempting to log in as {BLUESKY_HANDLE}...")
        client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)
        print("Authentication successful.")
        return client
    except Exception as e:
        print(f"Authentication failed. Please check your handle and app password: {e}")
        return None

def collect_posts(client: Client):
    """
    Collects posts using defined search queries and organizes them into a list.
    Note: Bluesky search is limited by rate limits and result complexity. 
    You may need to run this script multiple times or refine queries.
    """
    all_posts = []
    
    # Define the maximum number of results to fetch per query attempt
    LIMIT = 100
    
    for category_label, query in SEARCH_QUERIES:
        print(f"\nSearching for '{query}' (Targeting category: {category_label})...")
        
        # Cursor for pagination
        cursor = None
        posts_found_for_query = 0
        
        # Attempt to get posts, max 2 pages per query to manage rate limits
        for i in range(2): 
            try:
                # --- FIX: Passing parameters inside the Params object ---
                response = client.app.bsky.feed.search_posts(
                    params=models.AppBskyFeedSearchPosts.Params(
                        q=query, 
                        limit=LIMIT, 
                        cursor=cursor
                    )
                )
                # --------------------------------------------------------
                
                if not response.posts:
                    print(f"  -> No more posts found for query '{query}' on page {i+1}.")
                    break

                for post_view in response.posts:
                    if posts_found_for_query >= 100: # Safety break
                        break

                    # Extract main post text
                    post_text = post_view.record.text if hasattr(post_view.record, 'text') else 'N/A'
                    
                    # Create a unique ID from the AT URI (the last segment)
                    uri_parts = post_view.uri.split('/')
                    unique_id = uri_parts[-1] if len(uri_parts) > 0 else f"id_{random.randint(10000, 99999)}"

                    # Data row structure for CSV
                    post_data = {
                        'id': unique_id,
                        'post_text': post_text.replace('\n', ' ').strip(), # Clean up text
                        'author_handle': post_view.author.handle,
                        'post_uri': post_view.uri,
                        # TEMPORARY_CLASSIFICATION is based on the query, but needs manual check
                        'TEMPORARY_CLASSIFICATION': category_label, 
                        # This column will be filled manually or in the next script step
                        'final_label': '', 
                        # Placeholder for the synthetic status
                        'is_synthetic': 'FALSE' 
                    }
                    all_posts.append(post_data)
                    posts_found_for_query += 1

                print(f"  -> Found {len(response.posts)} posts on page {i+1}. Total found so far: {posts_found_for_query}")

                if response.cursor:
                    cursor = response.cursor
                    # Wait for a few seconds to respect API rate limits
                    time.sleep(random.uniform(5, 10)) 
                else:
                    break # End of results for this query

            except Exception as e:
                print(f"  -> An error occurred while fetching posts for '{query}': {e}")
                time.sleep(15) # Longer wait on error
                break
                
    return all_posts

def save_to_csv(data: list, filename: str):
    """Saves the collected post data to a CSV file."""
    if not data:
        print("No data collected to save.")
        return

    fieldnames = ['id', 'post_text', 'author_handle', 'post_uri', 'TEMPORARY_CLASSIFICATION', 'final_label', 'is_synthetic']
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"\nSuccessfully saved {len(data)} posts to {filename}")
        print("NOTE: Please manually review the 'TEMPORARY_CLASSIFICATION' and fill in the 'final_label' column based on the 90/60/30 split before sending the file back.")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == '__main__':
    client = authenticate_client()
    
    if client:
        collected_data = collect_posts(client)
        save_to_csv(collected_data, OUTPUT_FILENAME)