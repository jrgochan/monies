import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Union

import tweepy

from src.api import API_KEYS, logger, make_api_request


class SocialMediaResponse(TypedDict, total=False):
    """Type definition for social media response"""

    success: bool
    message: str
    post_id: Optional[str]


class SocialMediaPublisher:
    """
    Handles social media interactions across multiple platforms
    """

    FACEBOOK_API_VERSION = "v17.0"
    LINKEDIN_API_VERSION = "v2"

    @staticmethod
    def get_twitter_client(
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_secret: Optional[str] = None,
    ) -> Optional[tweepy.Client]:
        """
        Create and return an authenticated Twitter client

        Args:
            api_key: Twitter API key (consumer key)
            api_secret: Twitter API secret (consumer secret)
            access_token: Twitter access token
            access_secret: Twitter access token secret

        Returns:
            Authenticated tweepy Client object or None if authentication fails
        """
        # Use provided credentials or fall back to environment variables
        api_key = api_key or API_KEYS["twitter"]
        api_secret = api_secret or API_KEYS["twitter_secret"]

        # If credentials are missing, return None
        if not api_key or not api_secret or not access_token or not access_secret:
            logger.error("Missing Twitter credentials")
            return None

        try:
            client = tweepy.Client(
                consumer_key=api_key,
                consumer_secret=api_secret,
                access_token=access_token,
                access_token_secret=access_secret,
            )

            # Verify the credentials by making a simple request
            # This will raise an exception if authentication fails
            client.get_me()

            return client
        except Exception as e:
            logger.error(f"Failed to authenticate with Twitter: {str(e)}")
            return None

    @classmethod
    def post_to_twitter(
        cls,
        message: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_secret: Optional[str] = None,
        media_path: Optional[str] = None,
    ) -> SocialMediaResponse:
        """
        Post content to Twitter

        Args:
            message: Text content to post
            api_key: Twitter API key
            api_secret: Twitter API secret
            access_token: User's access token
            access_secret: User's access token secret
            media_path: Optional path to media file to attach

        Returns:
            Dictionary with result information
        """
        result: SocialMediaResponse = {"success": False, "message": "", "post_id": None}

        # Get Twitter client
        client = cls.get_twitter_client(
            api_key, api_secret, access_token, access_secret
        )

        if not client:
            result["message"] = "Failed to authenticate with Twitter"
            return result

        try:
            # Post tweet
            response = client.create_tweet(text=message)

            result["success"] = True
            result["message"] = "Tweet posted successfully"
            result["post_id"] = response.data["id"]

        except Exception as e:
            logger.error(f"Error posting to Twitter: {str(e)}")
            result["message"] = f"Error: {str(e)}"

        return result

    @classmethod
    def post_to_facebook(
        cls, content: str, page_access_token: str, media_path: Optional[str] = None
    ) -> SocialMediaResponse:
        """
        Post content to Facebook Page

        Args:
            content: Text content to post
            page_access_token: Facebook page access token
            media_path: Optional path to media file to attach

        Returns:
            Dictionary with result information
        """
        result: SocialMediaResponse = {"success": False, "message": "", "post_id": None}

        if not page_access_token:
            result["message"] = "Missing Facebook access token"
            return result

        try:
            # Get user pages first
            success, pages_data = make_api_request(
                url=f"https://graph.facebook.com/{cls.FACEBOOK_API_VERSION}/me/accounts",
                params={"access_token": page_access_token},
                error_msg="Error retrieving Facebook pages",
            )

            if not success or not isinstance(pages_data, dict):
                result["message"] = "Failed to retrieve Facebook pages"
                return result

            pages = pages_data.get("data", [])

            if not pages:
                result["message"] = "No Facebook Pages found for this token"
                return result

            # Use the first page
            page_id = pages[0]["id"]
            page_token = pages[0]["access_token"]

            # Add photo if provided
            if media_path and os.path.exists(media_path):
                # Upload photo directly to page
                photo_url = f"https://graph.facebook.com/{cls.FACEBOOK_API_VERSION}/{page_id}/photos"

                # Using files parameter requires direct requests call
                import requests

                files = {"source": open(media_path, "rb")}
                photo_data = {"caption": content, "access_token": page_token}

                response = requests.post(photo_url, files=files, data=photo_data)

                if response.status_code == 200:
                    result["success"] = True
                    result["message"] = "Facebook photo posted successfully"
                    result["post_id"] = response.json().get("id")
                    return result
                else:
                    logger.error(f"Error posting photo to Facebook: {response.text}")
                    # Fall back to regular post

            # Regular post (no photo or photo upload failed)
            success, post_data = make_api_request(
                url=f"https://graph.facebook.com/{cls.FACEBOOK_API_VERSION}/{page_id}/feed",
                method="POST",
                data={"message": content, "access_token": page_token},
                error_msg="Error posting to Facebook",
            )

            if success and isinstance(post_data, dict) and "id" in post_data:
                result["success"] = True
                result["message"] = "Facebook post published successfully"
                result["post_id"] = post_data.get("id")
            else:
                result["message"] = f"Error publishing to Facebook"

        except Exception as e:
            logger.error(f"Error posting to Facebook: {str(e)}")
            result["message"] = f"Error: {str(e)}"

        return result

    @classmethod
    def post_to_linkedin(
        cls, content: str, access_token: str, media_path: Optional[str] = None
    ) -> SocialMediaResponse:
        """
        Post content to LinkedIn

        Args:
            content: Text content to post
            access_token: LinkedIn access token
            media_path: Optional path to media file to attach

        Returns:
            Dictionary with result information
        """
        result: SocialMediaResponse = {"success": False, "message": "", "post_id": None}

        if not access_token:
            result["message"] = "Missing LinkedIn access token"
            return result

        try:
            # Get user profile
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            success, profile_data = make_api_request(
                url=f"https://api.linkedin.com/{cls.LINKEDIN_API_VERSION}/me",
                headers=headers,
                error_msg="Error retrieving LinkedIn profile",
            )

            if not success or not isinstance(profile_data, dict):
                result["message"] = "Failed to retrieve LinkedIn profile"
                return result

            # Get profile ID
            person_id = profile_data.get("id")

            if not person_id:
                result["message"] = "LinkedIn profile ID not found"
                return result

            # Create post
            post_data = {
                "author": f"urn:li:person:{person_id}",
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {"text": content},
                        "shareMediaCategory": "NONE",
                    }
                },
                "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
            }

            # Add media if provided
            if media_path and os.path.exists(media_path):
                # LinkedIn requires a more complex flow to upload images
                # For simplicity, we'll just note that media is not supported in this example
                logger.warning("LinkedIn media upload not implemented in this example")

            success, response_data = make_api_request(
                url=f"https://api.linkedin.com/{cls.LINKEDIN_API_VERSION}/ugcPosts",
                method="POST",
                headers=headers,
                json_data=post_data,
                error_msg="Error posting to LinkedIn",
            )

            if success and isinstance(response_data, dict):
                result["success"] = True
                result["message"] = "LinkedIn post published successfully"
                result["post_id"] = response_data.get("id")
            else:
                result["message"] = "Error publishing to LinkedIn"

        except Exception as e:
            logger.error(f"Error posting to LinkedIn: {str(e)}")
            result["message"] = f"Error: {str(e)}"

        return result


# Legacy functions to maintain backward compatibility


def get_twitter_trending_topics(
    search_term: str,
    api_key: str,
    api_secret: str,
    access_token: str,
    access_secret: str,
) -> List[Dict[str, Any]]:
    """
    Get trending topics from Twitter with optional filtering.
    """
    try:
        # Return mock data for testing
        return [
            {"name": "#Bitcoin", "tweet_volume": 50000},
            {"name": "#Ethereum", "tweet_volume": 30000},
            {"name": "#Crypto", "tweet_volume": 25000},
        ]
    except Exception as e:
        logger.error(f"Error getting Twitter trends: {str(e)}")
        return []


def post_to_twitter(
    message: str,
    api_key: str,
    api_secret: str,
    access_token: str,
    access_secret: str,
    media_path: Optional[str] = None,
) -> SocialMediaResponse:
    """
    Legacy wrapper for the SocialMediaPublisher.post_to_twitter method
    """
    return SocialMediaPublisher.post_to_twitter(
        message, api_key, api_secret, access_token, access_secret, media_path
    )


def post_to_facebook(
    page_access_token: str, content: str, media_path: Optional[str] = None
) -> SocialMediaResponse:
    """
    Legacy wrapper for the SocialMediaPublisher.post_to_facebook method
    """
    return SocialMediaPublisher.post_to_facebook(content, page_access_token, media_path)


def post_to_linkedin(
    access_token: str, content: str, media_path: Optional[str] = None
) -> SocialMediaResponse:
    """
    Legacy wrapper for the SocialMediaPublisher.post_to_linkedin method
    """
    return SocialMediaPublisher.post_to_linkedin(content, access_token, media_path)


class SentimentAnalyzer:
    """
    Handles sentiment analysis for social media and text content
    """

    @staticmethod
    def analyze_social_sentiment(query: str) -> Dict[str, Union[float, str]]:
        """
        Analyze sentiment for a topic on social media

        Args:
            query: Topic or keyword to analyze

        Returns:
            Dictionary with sentiment analysis results
        """
        # Mock sentiment analysis results for testing
        # In a real implementation, this would call an external API
        return {
            "positive": 0.65,
            "negative": 0.15,
            "neutral": 0.20,
            "overall": "positive",
        }

    @staticmethod
    def get_twitter_sentiment(query: str, days: int = 7) -> Dict[str, Any]:
        """
        Get sentiment analysis for a Twitter search query over a specified number of days

        Args:
            query: Search term or hashtag to analyze
            days: Number of days to look back for tweets

        Returns:
            Dictionary with sentiment analysis results
        """
        # For demonstration purposes, return mock data
        # In a real implementation, this would use Twitter API
        sentiment_data = {
            "positive": 65,
            "neutral": 20,
            "negative": 15,
            "total_posts": 250,
            "trending_words": ["moon", "bullish", "invest", "future", "gains"],
            "notable_accounts": ["@elonmusk", "@CryptoAnalyst", "@TechInvestor"],
            "recent_posts": [
                {
                    "text": f"Really bullish on {query} right now! The fundamentals are strong and the technical analysis looks promising. #investing",
                    "user": "CryptoAnalyst",
                    "likes": 432,
                    "sentiment": "positive",
                },
                {
                    "text": f"Just increased my position in {query}. The recent dip was a great buying opportunity.",
                    "user": "InvestorDaily",
                    "likes": 256,
                    "sentiment": "positive",
                },
                {
                    "text": f"Not sure about {query} at current prices. Might be overvalued considering market conditions.",
                    "user": "TechInvestor",
                    "likes": 187,
                    "sentiment": "neutral",
                },
                {
                    "text": f"Disappointed with the performance of {query} lately. Was expecting more after the recent announcements.",
                    "user": "StockTrader",
                    "likes": 92,
                    "sentiment": "negative",
                },
            ],
        }

        return sentiment_data

    @staticmethod
    def analyze_text_sentiment(text: str) -> Dict[str, Union[float, str]]:
        """
        Analyze sentiment of a given text

        Args:
            text: Text content to analyze

        Returns:
            Dictionary with sentiment scores
        """
        # Mock sentiment analysis for testing
        # In a real implementation, this would use NLP or external API
        return {
            "positive": 0.65,
            "negative": 0.15,
            "neutral": 0.20,
            "overall": "positive",
        }


# Legacy functions to maintain backward compatibility


def analyze_social_sentiment(query: str) -> Dict[str, Union[float, str]]:
    """
    Legacy wrapper for SentimentAnalyzer.analyze_social_sentiment
    """
    return SentimentAnalyzer.analyze_social_sentiment(query)


def get_twitter_sentiment(query: str, days: int = 7) -> Dict[str, Any]:
    """
    Legacy wrapper for SentimentAnalyzer.get_twitter_sentiment
    """
    return SentimentAnalyzer.get_twitter_sentiment(query, days)


def analyze_text_sentiment(text: str) -> Dict[str, Union[float, str]]:
    """
    Legacy wrapper for SentimentAnalyzer.analyze_text_sentiment
    """
    return SentimentAnalyzer.analyze_text_sentiment(text)


class PostScheduler:
    """
    Manages social media post scheduling and publishing
    """

    @classmethod
    def schedule_post(
        cls,
        user_id: int,
        platform: str,
        content: str,
        scheduled_time: datetime,
        media_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Schedule a post for later publishing

        Args:
            user_id: ID of the user scheduling the post
            platform: Social media platform (twitter, facebook, linkedin)
            content: Text content of the post
            scheduled_time: When to publish the post
            media_path: Optional path to media file to attach

        Returns:
            Dictionary with result information
        """
        from src.models.database import ScheduledPost, SessionLocal

        db = SessionLocal()
        try:
            # Create new scheduled post
            post = ScheduledPost(
                user_id=user_id,
                platform=platform,
                content=content,
                media_path=media_path,
                scheduled_time=scheduled_time,
                posted=False,
            )

            db.add(post)
            db.commit()
            db.refresh(post)

            return {
                "success": True,
                "message": f"Post scheduled for {scheduled_time}",
                "post_id": post.id,
            }

        except Exception as e:
            db.rollback()
            logger.error(f"Error scheduling post: {str(e)}")
            return {"success": False, "message": f"Error: {str(e)}"}

        finally:
            db.close()

    @classmethod
    def get_scheduled_posts(cls, user_id: int) -> Dict[str, Any]:
        """
        Get all scheduled posts for a user

        Args:
            user_id: ID of the user

        Returns:
            Dictionary with list of scheduled posts
        """
        from src.models.database import ScheduledPost, SessionLocal

        db = SessionLocal()
        try:
            # Query scheduled posts
            posts = (
                db.query(ScheduledPost)
                .filter(
                    ScheduledPost.user_id == user_id,
                    ScheduledPost.posted.is_(False),
                    ScheduledPost.scheduled_time > datetime.utcnow(),
                )
                .order_by(ScheduledPost.scheduled_time)
                .all()
            )

            result = []
            for post in posts:
                result.append(
                    {
                        "id": post.id,
                        "platform": post.platform,
                        "content": post.content,
                        "media_path": post.media_path,
                        "scheduled_time": post.scheduled_time.isoformat(),
                    }
                )

            return {"success": True, "posts": result}

        except Exception as e:
            logger.error(f"Error getting scheduled posts: {str(e)}")
            return {"success": False, "message": f"Error: {str(e)}", "posts": []}

        finally:
            db.close()

    @classmethod
    def cancel_scheduled_post(cls, post_id: int) -> Dict[str, Any]:
        """
        Cancel a scheduled post

        Args:
            post_id: ID of the scheduled post to cancel

        Returns:
            Dictionary with result information
        """
        from src.models.database import ScheduledPost, SessionLocal

        db = SessionLocal()
        try:
            # Find and delete post
            post = db.query(ScheduledPost).filter(ScheduledPost.id == post_id).first()

            if not post:
                return {"success": False, "message": "Scheduled post not found"}

            db.delete(post)
            db.commit()

            return {"success": True, "message": "Scheduled post cancelled"}

        except Exception as e:
            db.rollback()
            logger.error(f"Error cancelling scheduled post: {str(e)}")
            return {"success": False, "message": f"Error: {str(e)}"}

        finally:
            db.close()

    @classmethod
    def publish_scheduled_posts(cls) -> Dict[str, Any]:
        """
        Publish all scheduled posts that are due
        This function would be called by a scheduler

        Returns:
            Dictionary with result information
        """
        from src.models.database import ScheduledPost, SessionLocal, SocialAccount
        from src.utils.security import decrypt_data

        db = SessionLocal()
        try:
            # Get posts scheduled for now or in the past
            now = datetime.utcnow()
            posts = (
                db.query(ScheduledPost)
                .filter(
                    ScheduledPost.posted.is_(False), ScheduledPost.scheduled_time <= now
                )
                .all()
            )

            published_count = 0
            for post in posts:
                try:
                    # Get user's social credentials
                    social_account = (
                        db.query(SocialAccount)
                        .filter(
                            SocialAccount.user_id == post.user_id,
                            SocialAccount.platform == post.platform,
                        )
                        .first()
                    )

                    if not social_account:
                        logger.error(f"Social account not found for post {post.id}")
                        continue

                    # Decrypt tokens
                    token = decrypt_data(social_account.encrypted_token)
                    token_secret = (
                        decrypt_data(social_account.encrypted_token_secret)
                        if social_account.encrypted_token_secret
                        else None
                    )

                    # Publish based on platform
                    if post.platform == "twitter":
                        if not token_secret:
                            logger.error(
                                f"Missing Twitter token secret for post {post.id}"
                            )
                            continue

                        # Use the SocialMediaPublisher
                        result = SocialMediaPublisher.post_to_twitter(
                            message=post.content,
                            api_key=API_KEYS["twitter"],
                            api_secret=API_KEYS["twitter_secret"],
                            access_token=token,
                            access_secret=token_secret,
                            media_path=post.media_path,
                        )

                    elif post.platform == "facebook":
                        result = SocialMediaPublisher.post_to_facebook(
                            content=post.content,
                            page_access_token=token,
                            media_path=post.media_path,
                        )

                    elif post.platform == "linkedin":
                        result = SocialMediaPublisher.post_to_linkedin(
                            content=post.content,
                            access_token=token,
                            media_path=post.media_path,
                        )

                    else:
                        logger.error(
                            f"Unsupported platform {post.platform} for post {post.id}"
                        )
                        continue

                    # Update post status
                    if result["success"]:
                        post.posted = True
                        db.commit()
                        published_count += 1
                        logger.info(
                            f"Published scheduled post {post.id} to {post.platform}"
                        )
                    else:
                        logger.error(
                            f"Failed to publish post {post.id}: {result['message']}"
                        )

                except Exception as e:
                    logger.error(f"Error publishing post {post.id}: {str(e)}")
                    continue

            return {
                "success": True,
                "message": f"Published {published_count} scheduled posts",
                "published_count": published_count,
                "total_processed": len(posts),
            }

        except Exception as e:
            logger.error(f"Error publishing scheduled posts: {str(e)}")
            return {"success": False, "message": f"Error: {str(e)}"}

        finally:
            db.close()


# Legacy functions to maintain backward compatibility


def schedule_post(
    user_id: int,
    platform: str,
    content: str,
    scheduled_time: datetime,
    media_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Legacy wrapper for PostScheduler.schedule_post
    """
    return PostScheduler.schedule_post(
        user_id, platform, content, scheduled_time, media_path
    )


def get_scheduled_posts(user_id: int) -> Dict[str, Any]:
    """
    Legacy wrapper for PostScheduler.get_scheduled_posts
    """
    return PostScheduler.get_scheduled_posts(user_id)


def cancel_scheduled_post(post_id: int) -> Dict[str, Any]:
    """
    Legacy wrapper for PostScheduler.cancel_scheduled_post
    """
    return PostScheduler.cancel_scheduled_post(post_id)


def publish_scheduled_posts() -> Dict[str, Any]:
    """
    Legacy wrapper for PostScheduler.publish_scheduled_posts
    """
    return PostScheduler.publish_scheduled_posts()
