#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia API Utility
"""

import requests
import time
import logging
from typing import Dict, Any, Optional

class WikipediaAPI:
    """Wikipedia API Client"""
    
    def __init__(self, timeout: int = 10, max_retries: int = 5, retry_delay: float = 1.0):
        self.base_url = "https://en.wikipedia.org/api/rest_v1"
        # Set User-Agent to comply with Wikipedia API requirements
        self.headers = {
            'User-Agent': 'LLM_KG_Pipeline/1.0 (https://example.com/contact) requests/2.28.0'
        }
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
    def get_page_summary(self, title: str) -> Optional[str]:
        """Get page summary with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}/page/summary/{title}"
                response = requests.get(url, headers=self.headers, timeout=self.timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('extract', '')
                elif response.status_code == 429:  # Rate limited
                    wait_time = self.retry_delay * (2 ** attempt) + 10  # Extra 10 seconds wait when rate limited
                    self.logger.warning(f"Wikipedia API rate limited, waiting {wait_time:.1f} seconds before retry (attempt {attempt + 1}/{self.max_retries})")
                    if attempt < self.max_retries - 1:
                        time.sleep(wait_time)
                        continue
                elif response.status_code == 404:
                    # Page doesn't exist, no need to retry
                    self.logger.debug(f"Wikipedia page doesn't exist: {title}")
                    return None
                else:
                    # Other HTTP errors
                    self.logger.warning(f"Wikipedia API HTTP error {response.status_code}: {title}")
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        self.logger.debug(f"Waiting {wait_time:.1f} seconds before retry (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(wait_time)
                        continue
                        
            except requests.exceptions.Timeout:
                self.logger.warning(f"Wikipedia API timeout: {title} (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    self.logger.debug(f"Waiting {wait_time:.1f} seconds before retry")
                    time.sleep(wait_time)
                    continue
            except requests.exceptions.ConnectionError:
                self.logger.warning(f"Wikipedia API connection error: {title} (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    self.logger.debug(f"Waiting {wait_time:.1f} seconds before retry")
                    time.sleep(wait_time)
                    continue
            except Exception as e:
                self.logger.error(f"Wikipedia API unknown error: {title}, error: {str(e)} (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                    
        # All retries failed
        self.logger.error(f"Wikipedia API failed to get summary after {self.max_retries} retries: {title}")
        return None
        
    def get_wikipedia_summary(self, title: str) -> Optional[str]:
        """Get Wikipedia page summary - same as get_page_summary, but method name matches WikidataRetriever's expectation"""
        return self.get_page_summary(title)
        
    def page_exists(self, title: str) -> bool:
        """Check if page exists with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}/page/summary/{title}"
                response = requests.get(url, headers=self.headers, timeout=self.timeout)
                return response.status_code == 200
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout checking page existence: {title} (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
            except Exception as e:
                self.logger.warning(f"Error checking page existence: {title}, error: {str(e)} (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
        
        # Retry failed, assume page doesn't exist
        return False
