import tweepy
import requests
import json
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_twitter_trending_topics(search_term, api_key, api_secret, access_token, access_secret):
    """
    Get trending topics from Twitter with optional filtering.
    """
    try:
        # Return mock data for testing
        return [
            {'name': '#Bitcoin', 'tweet_volume': 50000},
            {'name': '#Ethereum', 'tweet_volume': 30000},
            {'name': '#Crypto', 'tweet_volume': 25000}
        ]
    except Exception as e:
        logger.error(f"Error getting Twitter trends: {str(e)}")
        return []

def post_to_twitter(message, api_key, api_secret, access_token, access_secret, media_path=None):
    """
    Post content to Twitter.
    """
    result = {
        'success': False,
        'message': '',
        'post_id': None
    }
    
    try:
        # New Twitter v2 API with Client class
        client = tweepy.Client(
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_secret
        )
        
        if not client:
            result['message'] = "Failed to authenticate with Twitter"
            return result
        
        # Post tweet - v2 API has simplified tweet creation
        response = client.create_tweet(text=message)
        
        result['success'] = True
        result['message'] = "Tweet posted successfully"
        result['tweet_id'] = response.data['id']
    
    except Exception as e:
        logger.error(f"Error posting to Twitter: {str(e)}")
        result['message'] = f"Error: {str(e)}"
    
    return result

def post_to_facebook(page_access_token, content, media_path=None):
    """
    Post content to Facebook Page.
    """
    result = {
        'success': False,
        'message': '',
        'post_id': None
    }
    
    try:
        # Get user pages first
        user_url = f"https://graph.facebook.com/v17.0/me/accounts?access_token={page_access_token}"
        response = requests.get(user_url)
        pages = response.json().get('data', [])
        
        if not pages:
            result['message'] = "No Facebook Pages found for this token"
            return result
        
        # Use the first page
        page_id = pages[0]['id']
        page_token = pages[0]['access_token']
        
        # Prepare post data
        post_data = {
            'message': content,
            'access_token': page_token
        }
        
        # Add photo if provided
        if media_path and os.path.exists(media_path):
            # Upload photo
            photo_url = f"https://graph.facebook.com/v17.0/{page_id}/photos"
            files = {'source': open(media_path, 'rb')}
            photo_data = {
                'caption': content,
                'access_token': page_token
            }
            response = requests.post(photo_url, files=files, data=photo_data)
            
            if response.status_code == 200:
                result['success'] = True
                result['message'] = "Facebook photo posted successfully"
                result['post_id'] = response.json().get('id')
                return result
            else:
                logger.error(f"Error posting photo to Facebook: {response.text}")
                # Fall back to regular post
        
        # Regular post (no photo or photo upload failed)
        post_url = f"https://graph.facebook.com/v17.0/{page_id}/feed"
        response = requests.post(post_url, data=post_data)
        
        if response.status_code == 200:
            result['success'] = True
            result['message'] = "Facebook post published successfully"
            result['post_id'] = response.json().get('id')
        else:
            result['message'] = f"Error: {response.text}"
    
    except Exception as e:
        logger.error(f"Error posting to Facebook: {str(e)}")
        result['message'] = f"Error: {str(e)}"
    
    return result

def post_to_linkedin(access_token, content, media_path=None):
    """
    Post content to LinkedIn.
    """
    result = {
        'success': False,
        'message': '',
        'post_id': None
    }
    
    try:
        # Get user profile first
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
        }
        
        response = requests.get(
            'https://api.linkedin.com/v2/me', 
            headers=headers
        )
        
        if response.status_code != 200:
            result['message'] = f"Error getting LinkedIn profile: {response.text}"
            return result
        
        # Get profile ID
        profile_data = response.json()
        person_id = profile_data.get('id')
        
        # Create post
        post_url = 'https://api.linkedin.com/v2/ugcPosts'
        
        post_data = {
            "author": f"urn:li:person:{person_id}",
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": content
                    },
                    "shareMediaCategory": "NONE"
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
            }
        }
        
        # Add media if provided
        if media_path and os.path.exists(media_path):
            # LinkedIn requires a more complex flow to upload images
            # For simplicity, we'll just note that media is not supported in this example
            logger.warning("LinkedIn media upload not implemented in this example")
        
        response = requests.post(
            post_url,
            headers=headers,
            json=post_data
        )
        
        if response.status_code in (200, 201):
            result['success'] = True
            result['message'] = "LinkedIn post published successfully"
            result['post_id'] = response.json().get('id')
        else:
            result['message'] = f"Error: {response.text}"
    
    except Exception as e:
        logger.error(f"Error posting to LinkedIn: {str(e)}")
        result['message'] = f"Error: {str(e)}"
    
    return result

def analyze_social_sentiment(query):
    """
    Analyze sentiment for a topic on social media.
    """  
    # Mock sentiment analysis results for testing
    return {
        'positive': 0.65,
        'negative': 0.15,
        'neutral': 0.20,
        'overall': 'positive'
    }


def analyze_text_sentiment(text):
    """
    Analyze sentiment of a given text.
    """
    # Mock sentiment analysis for testing
    return {
        'positive': 0.65,
        'negative': 0.15,
        'neutral': 0.20,
        'overall': 'positive'
    }

def schedule_post(user_id, platform, content, scheduled_time, media_path=None):
    """
    Schedule a post for later publishing.
    """
    from src.models.database import SessionLocal, ScheduledPost
    
    db = SessionLocal()
    try:
        # Create new scheduled post
        post = ScheduledPost(
            user_id=user_id,
            platform=platform,
            content=content,
            media_path=media_path,
            scheduled_time=scheduled_time,
            posted=False
        )
        
        db.add(post)
        db.commit()
        db.refresh(post)
        
        return {
            'success': True,
            'message': f"Post scheduled for {scheduled_time}",
            'post_id': post.id
        }
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error scheduling post: {str(e)}")
        return {
            'success': False,
            'message': f"Error: {str(e)}"
        }
    
    finally:
        db.close()

def get_scheduled_posts(user_id):
    """
    Get all scheduled posts for a user.
    """
    from src.models.database import SessionLocal, ScheduledPost
    
    db = SessionLocal()
    try:
        # Query scheduled posts
        posts = db.query(ScheduledPost).filter(
            ScheduledPost.user_id == user_id,
            ScheduledPost.posted == False,
            ScheduledPost.scheduled_time > datetime.utcnow()
        ).order_by(ScheduledPost.scheduled_time).all()
        
        result = []
        for post in posts:
            result.append({
                'id': post.id,
                'platform': post.platform,
                'content': post.content,
                'media_path': post.media_path,
                'scheduled_time': post.scheduled_time.isoformat()
            })
        
        return {
            'success': True,
            'posts': result
        }
    
    except Exception as e:
        logger.error(f"Error getting scheduled posts: {str(e)}")
        return {
            'success': False,
            'message': f"Error: {str(e)}",
            'posts': []
        }
    
    finally:
        db.close()

def cancel_scheduled_post(post_id):
    """
    Cancel a scheduled post.
    """
    from src.models.database import SessionLocal, ScheduledPost
    
    db = SessionLocal()
    try:
        # Find and delete post
        post = db.query(ScheduledPost).filter(ScheduledPost.id == post_id).first()
        
        if not post:
            return {
                'success': False,
                'message': "Scheduled post not found"
            }
        
        db.delete(post)
        db.commit()
        
        return {
            'success': True,
            'message': "Scheduled post cancelled"
        }
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error cancelling scheduled post: {str(e)}")
        return {
            'success': False,
            'message': f"Error: {str(e)}"
        }
    
    finally:
        db.close()

def publish_scheduled_posts():
    """
    Publish all scheduled posts that are due.
    This function would be called by a scheduler.
    """
    from src.models.database import SessionLocal, ScheduledPost, SocialAccount
    from src.utils.security import decrypt_data
    
    db = SessionLocal()
    try:
        # Get posts scheduled for now or in the past
        now = datetime.utcnow()
        posts = db.query(ScheduledPost).filter(
            ScheduledPost.posted == False,
            ScheduledPost.scheduled_time <= now
        ).all()
        
        for post in posts:
            try:
                # Get user's social credentials
                social_account = db.query(SocialAccount).filter(
                    SocialAccount.user_id == post.user_id,
                    SocialAccount.platform == post.platform
                ).first()
                
                if not social_account:
                    logger.error(f"Social account not found for post {post.id}")
                    continue
                
                # Decrypt tokens
                token = decrypt_data(social_account.encrypted_token)
                token_secret = decrypt_data(social_account.encrypted_token_secret) if social_account.encrypted_token_secret else None
                
                # Publish based on platform
                if post.platform == 'twitter':
                    if not token_secret:
                        logger.error(f"Missing Twitter token secret for post {post.id}")
                        continue
                    
                    # Get API keys from environment
                    api_key = os.getenv("TWITTER_API_KEY", "")
                    api_secret = os.getenv("TWITTER_API_SECRET", "")
                    
                    result = post_to_twitter(
                        api_key, api_secret, token, token_secret,
                        post.content, post.media_path
                    )
                
                elif post.platform == 'facebook':
                    result = post_to_facebook(
                        token, post.content, post.media_path
                    )
                
                elif post.platform == 'linkedin':
                    result = post_to_linkedin(
                        token, post.content, post.media_path
                    )
                
                else:
                    logger.error(f"Unsupported platform {post.platform} for post {post.id}")
                    continue
                
                # Update post status
                if result['success']:
                    post.posted = True
                    db.commit()
                    logger.info(f"Published scheduled post {post.id} to {post.platform}")
                else:
                    logger.error(f"Failed to publish post {post.id}: {result['message']}")
            
            except Exception as e:
                logger.error(f"Error publishing post {post.id}: {str(e)}")
                continue
        
        return {
            'success': True,
            'message': f"Published {len(posts)} scheduled posts"
        }
    
    except Exception as e:
        logger.error(f"Error publishing scheduled posts: {str(e)}")
        return {
            'success': False,
            'message': f"Error: {str(e)}"
        }
    
    finally:
        db.close()