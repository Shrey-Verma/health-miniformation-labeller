import csv
import time
import random
from atproto import Client, models

bluesky_handle = '.bsky.social'
bluesky_password = '' 
output_filename = 'live.csv'

# search queries and their associated categories
search_queries = [
    ('positive', 'cure cancer and miracle'),
    ('positive', 'reverse diabetes and supplement'),
    ('positive', '72 hour dry fast cures'),
    ('positive', 'stop insulin and cinnamon'),
    ('positive', 'nmn miracle cure'),
    ('hard negative', 'debunked or misinformation and weight loss'),
    ('hard negative', 'scam or fake or dangerous and supplement'),
    ('hard negative', 'talk to your doctor or consult physician'),
    ('borderline', 'thinking of trying fasting or not sure if safe peptide'),
    ('borderline', 'is this effective or potential side effects or maybe works'),
]

# authenticate and initialize bluesky client
def authenticate_client():
    # initializes and authenticates the bluesky client
    client = Client()
    try:
        print(f"attempting to log in as {bluesky_handle}...")
        client.login(bluesky_handle, bluesky_password)
        print("authentication successful")
        return client
    except Exception as e:
        print(f"authentication failed please check your handle and app password {e}")
        return None

# collect posts from bluesky using defined queries
def collect_posts(client: Client):
    all_posts = []
    limit = 100
    
    for category_label, query in search_queries:
        print(f"\nsearching for '{query}' targeting category {category_label}...")
        
        cursor = None
        posts_found_for_query = 0
        
        for i in range(2): 
            try:
                response = client.app.bsky.feed.search_posts(
                    params=models.AppBskyFeedSearchPosts.Params(
                        q=query, 
                        limit=limit, 
                        cursor=cursor
                    )
                )
                
                if not response.posts:
                    print(f"  -> no more posts found for query '{query}' on page {i+1}")
                    break

                for post_view in response.posts:
                    if posts_found_for_query >= 100:
                        break

                    post_text = post_view.record.text if hasattr(post_view.record, 'text') else 'n/a'
                    uri_parts = post_view.uri.split('/')
                    unique_id = uri_parts[-1] if len(uri_parts) > 0 else f"id_{random.randint(10000, 99999)}"

                    post_data = {
                        'id': unique_id,
                        'post_text': post_text.replace('\n', ' ').strip(),
                        'author_handle': post_view.author.handle,
                        'post_uri': post_view.uri,
                        'temporary_classification': category_label, 
                        'final_label': '', 
                        'is_synthetic': 'false' 
                    }
                    all_posts.append(post_data)
                    posts_found_for_query += 1

                print(f"  -> found {len(response.posts)} posts on page {i+1} total found so far {posts_found_for_query}")

                if response.cursor:
                    cursor = response.cursor
                    time.sleep(random.uniform(5, 10)) 
                else:
                    break 

            except Exception as e:
                print(f"  -> an error occurred while fetching posts for '{query}': {e}")
                time.sleep(15) 
                break
                
    return all_posts

# save collected data to csv file
def save_to_csv(data: list, filename: str):
    if not data:
        print("no data collected to save")
        return

    fieldnames = ['id', 'post_text', 'author_handle', 'post_uri', 'temporary_classification', 'final_label', 'is_synthetic']
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"\nsuccessfully saved {len(data)} posts to {filename}")
        print("note please manually review the 'temporary_classification' and fill in the 'final_label' column before sending the file back")
    except Exception as e:
        print(f"error saving file {e}")

if __name__ == '__main__':
    client = authenticate_client()
    
    if client:
        collected_data = collect_posts(client)
        save_to_csv(collected_data, output_filename)
