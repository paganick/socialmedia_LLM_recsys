"""
Text feature extraction for social media posts.

Computes per-post features from raw text:
- Sentiment polarity and subjectivity  (VADER / TextBlob)
- Primary topic + polarization score   (Cardiff NLP RoBERTa —
    cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all)
- Style indicators                     (regex: has_emoji, has_hashtag, has_mention, has_url)
- Average word length                  (character-level)
- Toxicity                             (Detoxify)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import re
from collections import Counter


class SentimentAnalyzer:
    """Analyze sentiment of tweets using multiple methods."""

    def __init__(self, method: str = 'textblob', llm_client=None):
        """
        Initialize sentiment analyzer.

        Args:
            method: 'textblob', 'vader', or 'llm'
            llm_client: LLM client instance (required if method='llm')
        """
        self.method = method
        self.llm_client = llm_client

        if method == 'textblob':
            try:
                from textblob import TextBlob
                self.analyzer = TextBlob
            except ImportError:
                raise ImportError("textblob not installed. Run: pip install textblob")

        elif method == 'vader':
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.analyzer = SentimentIntensityAnalyzer()
            except ImportError:
                raise ImportError("vaderSentiment not installed. Run: pip install vaderSentiment")

        elif method == 'llm':
            if llm_client is None:
                raise ValueError("LLM client required for LLM-based sentiment analysis")
            self.llm_client = llm_client
        else:
            raise ValueError(f"Unknown method: {method}")

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.

        Returns:
            Dict with 'polarity' (-1 to 1), 'subjectivity' (0 to 1),
            'label' ('positive', 'negative', 'neutral')
        """
        # Handle missing/invalid text
        if not isinstance(text, str) or not text.strip():
            return {
                'polarity': 0.0,
                'subjectivity': 0.5,
                'label': 'neutral',
                'positive_score': 0.0,
                'negative_score': 0.0,
                'neutral_score': 1.0
            }

        if self.method == 'textblob':
            blob = self.analyzer(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Classify label
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'

            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'label': label,
                'positive_score': max(0, polarity),
                'negative_score': max(0, -polarity),
                'neutral_score': 1 - abs(polarity)
            }

        elif self.method == 'vader':
            scores = self.analyzer.polarity_scores(text)
            compound = scores['compound']

            # Classify label
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'

            return {
                'polarity': compound,
                'subjectivity': 1 - scores['neu'],  # Approximate
                'label': label,
                'positive_score': scores['pos'],
                'negative_score': scores['neg'],
                'neutral_score': scores['neu']
            }

        elif self.method == 'llm':
            prompt = f"""Analyze the sentiment of the following text. Provide:
1. Label: positive, negative, or neutral
2. Polarity score: -1 (very negative) to +1 (very positive)
3. Subjectivity score: 0 (objective) to 1 (subjective)

Text: "{text}"

Respond in this exact format:
Label: [positive/negative/neutral]
Polarity: [score from -1 to 1]
Subjectivity: [score from 0 to 1]"""

            try:
                response = self.llm_client.generate(prompt, temperature=0.0)

                # Parse response
                label = 'neutral'
                polarity = 0.0
                subjectivity = 0.5

                lines = response.strip().split('\n')
                for line in lines:
                    if line.startswith('Label:'):
                        label = line.split(':', 1)[1].strip().lower()
                    elif line.startswith('Polarity:'):
                        try:
                            polarity = float(line.split(':', 1)[1].strip())
                        except:
                            pass
                    elif line.startswith('Subjectivity:'):
                        try:
                            subjectivity = float(line.split(':', 1)[1].strip())
                        except:
                            pass

                return {
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'label': label,
                    'positive_score': max(0, polarity),
                    'negative_score': max(0, -polarity),
                    'neutral_score': 1 - abs(polarity)
                }

            except Exception as e:
                print(f"Error in LLM sentiment analysis: {e}")
                # Return neutral as fallback
                return {
                    'polarity': 0.0,
                    'subjectivity': 0.5,
                    'label': 'neutral',
                    'positive_score': 0.0,
                    'negative_score': 0.0,
                    'neutral_score': 1.0
                }

    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """Analyze sentiment for batch of texts."""
        results = [self.analyze(text) for text in texts]
        return pd.DataFrame(results)


class TopicClassifier:
    """Classify tweets into topics."""

    def __init__(self, method: str = 'keyword'):
        """
        Initialize topic classifier.

        Args:
            method: 'keyword', 'lda', or 'llm'
        """
        self.method = method

        # Define topic keywords
        self.topic_keywords = {
            'politics': ['trump', 'biden', 'election', 'vote', 'congress', 'senate',
                        'democrat', 'republican', 'liberal', 'conservative', 'political'],
            'sports': ['game', 'team', 'player', 'win', 'score', 'football', 'basketball',
                      'baseball', 'soccer', 'nfl', 'nba', 'sports'],
            'entertainment': ['movie', 'music', 'show', 'watch', 'netflix', 'song',
                            'concert', 'album', 'actor', 'celebrity'],
            'technology': ['tech', 'app', 'software', 'computer', 'phone', 'iphone',
                          'android', 'ai', 'code', 'google', 'apple'],
            'news': ['breaking', 'report', 'news', 'update', 'announced', 'happened'],
            'personal': ['i', 'me', 'my', 'just', 'feeling', 'today', 'life'],
            'social': ['friend', 'people', 'everyone', 'family', 'someone', 'anyone'],
            'health': ['health', 'covid', 'vaccine', 'sick', 'doctor', 'hospital', 'virus'],
            'business': ['company', 'market', 'stock', 'business', 'economy', 'ceo', 'jobs']
        }

    def classify(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Classify text into topics.

        Returns:
            List of dicts with 'topic' and 'score'
        """
        # Handle missing/invalid text
        if not isinstance(text, str) or not text.strip():
            return [{'topic': 'other', 'score': 0}]

        if self.method == 'keyword':
            text_lower = text.lower()

            # Count keyword matches for each topic
            topic_scores = {}
            for topic, keywords in self.topic_keywords.items():
                score = sum(1 for kw in keywords if kw in text_lower)
                if score > 0:
                    topic_scores[topic] = score

            # Sort by score
            sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)

            # Return top k
            results = [
                {'topic': topic, 'score': score}
                for topic, score in sorted_topics[:top_k]
            ]

            # If no topics matched, classify as 'other'
            if not results:
                results = [{'topic': 'other', 'score': 0}]

            return results

        else:
            raise NotImplementedError(f"Method {self.method} not yet implemented")

    def classify_batch(self, texts: List[str], top_k: int = 3) -> List[List[Dict[str, Any]]]:
        """Classify batch of texts."""
        return [self.classify(text, top_k=top_k) for text in texts]


class StyleAnalyzer:
    """Analyze writing style of tweets."""

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze writing style.

        Returns:
            Dict with style features:
            - has_emoji: bool
            - has_hashtag: bool
            - has_mention: bool
            - has_url: bool
            - has_capitalization: bool (excessive caps)
            - has_punctuation: bool (!!!, ???)
            - word_count: int
            - avg_word_length: float
            - formality_score: float (0-1, higher = more formal)
        """
        # Handle missing/invalid text
        if not isinstance(text, str) or not text.strip():
            return {
                'has_emoji': False,
                'has_hashtag': False,
                'has_mention': False,
                'has_url': False,
                'has_capitalization': False,
                'has_punctuation': False,
                'word_count': 0,
                'avg_word_length': 0.0,
                'formality_score': 0.5
            }

        # Emoji detection (basic)
        has_emoji = bool(re.search(r'[\U0001F300-\U0001F9FF]', text))

        # Hashtag, mention, URL
        has_hashtag = '#' in text
        has_mention = '@' in text
        has_url = bool(re.search(r'http[s]?://|www\.', text))

        # Excessive capitalization
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        has_capitalization = caps_ratio > 0.3

        # Excessive punctuation
        has_punctuation = bool(re.search(r'[!?]{2,}', text))

        # Word-level features
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0

        # Formality score (heuristic)
        # Higher formality = longer words, no emojis/caps/punctuation
        formality_score = 0.0
        if word_count > 0:
            formality_score = min(1.0, avg_word_length / 8.0)  # Normalize by 8 chars
            if has_emoji:
                formality_score *= 0.7
            if has_capitalization:
                formality_score *= 0.8
            if has_punctuation:
                formality_score *= 0.9

        return {
            'has_emoji': has_emoji,
            'has_hashtag': has_hashtag,
            'has_mention': has_mention,
            'has_url': has_url,
            'has_capitalization': has_capitalization,
            'has_punctuation': has_punctuation,
            'word_count': word_count,
            'avg_word_length': avg_word_length,
            'formality_score': formality_score
        }

    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """Analyze style for batch of texts."""
        results = [self.analyze(text) for text in texts]
        return pd.DataFrame(results)


class PolarizationAnalyzer:
    """Analyze polarization/controversy level of tweets."""

    def __init__(self):
        """Initialize analyzer with controversial keywords."""
        # Keywords associated with polarizing topics
        self.polarizing_keywords = [
            # Political
            'trump', 'biden', 'liberal', 'conservative', 'democrat', 'republican',
            'socialism', 'capitalism', 'leftist', 'right-wing',

            # Social issues
            'abortion', 'gun', 'immigration', 'blm', 'lgbtq', 'trans', 'gender',
            'racism', 'sexism', 'privilege', 'woke', 'cancel',

            # Religion
            'god', 'christian', 'muslim', 'atheist', 'religion',

            # Controversy markers
            'controversial', 'debate', 'argue', 'fight', 'disagree', 'outrage',
            'stupid', 'idiot', 'hate', 'disgusting'
        ]

        # Strong sentiment words
        self.strong_sentiment_words = [
            'amazing', 'terrible', 'awful', 'fantastic', 'horrible', 'disgusting',
            'love', 'hate', 'furious', 'ecstatic', 'devastating', 'outrageous'
        ]

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze polarization/controversy.

        Returns:
            Dict with:
            - polarization_score: 0-1 (higher = more polarizing)
            - has_polarizing_content: bool
            - polarizing_keywords_found: List[str]
            - strong_sentiment: bool
        """
        # Handle missing/invalid text
        if not isinstance(text, str) or not text.strip():
            return {
                'polarization_score': 0.0,
                'has_polarizing_content': False,
                'polarizing_keywords_count': 0,
                'strong_sentiment': False,
                'controversy_level': 'low'
            }

        text_lower = text.lower()

        # Find polarizing keywords
        found_keywords = [kw for kw in self.polarizing_keywords if kw in text_lower]

        # Find strong sentiment words
        found_strong_sentiment = [w for w in self.strong_sentiment_words if w in text_lower]

        # Calculate polarization score
        keyword_score = min(1.0, len(found_keywords) / 3)  # Normalize
        sentiment_score = min(1.0, len(found_strong_sentiment) / 2)

        polarization_score = (keyword_score * 0.7 + sentiment_score * 0.3)

        return {
            'polarization_score': polarization_score,
            'has_polarizing_content': len(found_keywords) > 0,
            'polarizing_keywords_count': len(found_keywords),
            'strong_sentiment': len(found_strong_sentiment) > 0,
            'controversy_level': 'high' if polarization_score > 0.6 else 'medium' if polarization_score > 0.3 else 'low'
        }

    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """Analyze polarization for batch of texts."""
        results = [self.analyze(text) for text in texts]
        return pd.DataFrame(results)


class CardiffNLPClassifier:
    """
    Topic classifier and polarization scorer using the Cardiff NLP RoBERTa model:
        cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all

    primary_topic      = argmax label across all 19 topics
    polarization_score = P(news_&_social_concern)  [0–1]
    """

    MODEL_ID          = "cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all"
    POLARIZATION_LABEL = "news_&_social_concern"

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self._pipeline = None   # lazy-loaded

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers torch"
            )
        print(f"  Loading {self.MODEL_ID} ...")
        self._pipeline = pipeline(
            "text-classification",
            model=self.MODEL_ID,
            top_k=None,
            truncation=True,
            max_length=128,
            device=-1,      # CPU; set to 0 for GPU
        )

    def score_batch(self, texts: List[str]) -> dict:
        """
        Run the model over unique texts and return a mapping
        {text: {label: probability, ...}}.
        """
        self._load()
        unique = list(dict.fromkeys(str(t) for t in texts if t))
        scores: dict = {}
        total = len(unique)
        print(f"  Scoring {total:,} unique texts with Cardiff NLP RoBERTa ...")
        for i in range(0, total, self.batch_size):
            batch = unique[i: i + self.batch_size]
            results = self._pipeline(batch)
            for text, preds in zip(batch, results):
                scores[text] = {p["label"]: p["score"] for p in preds}
            if (i // self.batch_size) % 10 == 0:
                pct = min(100, (i + len(batch)) / total * 100)
                print(f"    {i + len(batch):>6,} / {total:,}  ({pct:.0f}%)")
        return scores

    def classify_dataframe(self, texts: pd.Series) -> pd.DataFrame:
        """
        Classify a Series of texts. Returns DataFrame with
        primary_topic and polarization_score aligned to the input index.
        """
        scores = self.score_batch(texts.astype(str).tolist())
        primary_topics, polarization_scores = [], []
        for text in texts.astype(str):
            label_map = scores.get(text, {})
            if label_map:
                primary_topics.append(max(label_map, key=label_map.get))
                polarization_scores.append(label_map.get(self.POLARIZATION_LABEL, 0.0))
            else:
                primary_topics.append("unknown")
                polarization_scores.append(0.0)
        return pd.DataFrame(
            {"primary_topic": primary_topics, "polarization_score": polarization_scores},
            index=texts.index,
        )


class GenderAnalyzer:
    """Analyze/infer gender of tweet author from text."""

    def __init__(self, method: str = 'keyword', llm_client=None):
        """
        Initialize gender analyzer.

        Args:
            method: 'keyword' or 'llm'
            llm_client: LLM client instance (required if method='llm')
        """
        self.method = method
        self.llm_client = llm_client

        if method == 'llm' and llm_client is None:
            raise ValueError("LLM client required for LLM-based gender analysis")

        # Gender-associated patterns (heuristic-based)
        # Note: These are stereotypical patterns and should be used carefully
        self.gender_indicators = {
            'female': {
                'pronouns': ['she', 'her', 'hers', 'herself'],
                'self_refs': ['i\'m a woman', 'i\'m a girl', 'i\'m a mom', 'i\'m a mother',
                             'as a woman', 'as a girl', 'as a mom', 'as a mother',
                             'i\'m pregnant', 'my husband', 'my boyfriend'],
                'keywords': ['makeup', 'lipstick', 'dress', 'heels', 'bra', 'period',
                           'menstrual', 'pregnancy', 'pregnant', 'breastfeed']
            },
            'male': {
                'pronouns': ['he', 'him', 'his', 'himself'],
                'self_refs': ['i\'m a man', 'i\'m a guy', 'i\'m a dad', 'i\'m a father',
                             'as a man', 'as a guy', 'as a dad', 'as a father',
                             'my wife', 'my girlfriend'],
                'keywords': ['beard', 'prostate', 'testosterone']
            }
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze gender indicators in text.

        Returns:
            Dict with:
            - gender_prediction: 'male', 'female', 'unknown'
            - confidence: float (0-1)
            - male_score: float
            - female_score: float
            - has_gender_indicators: bool
        """
        # Handle missing/invalid text
        if not isinstance(text, str) or not text.strip():
            return {
                'gender_prediction': 'unknown',
                'confidence': 0.0,
                'male_score': 0.0,
                'female_score': 0.0,
                'has_gender_indicators': False
            }

        if self.method == 'keyword':
            text_lower = text.lower()

            male_score = 0
            female_score = 0

            # Check pronouns (when referring to self in third person or quoting)
            for pronoun in self.gender_indicators['female']['pronouns']:
                if pronoun in text_lower:
                    female_score += 0.5

            for pronoun in self.gender_indicators['male']['pronouns']:
                if pronoun in text_lower:
                    male_score += 0.5

            # Check self-references (stronger signal)
            for ref in self.gender_indicators['female']['self_refs']:
                if ref in text_lower:
                    female_score += 3

            for ref in self.gender_indicators['male']['self_refs']:
                if ref in text_lower:
                    male_score += 3

            # Check keywords
            for keyword in self.gender_indicators['female']['keywords']:
                if keyword in text_lower:
                    female_score += 1

            for keyword in self.gender_indicators['male']['keywords']:
                if keyword in text_lower:
                    male_score += 1

            # Determine prediction
            total_score = male_score + female_score
            has_indicators = total_score > 0

            if total_score == 0:
                prediction = 'unknown'
                confidence = 0.0
            elif male_score > female_score:
                prediction = 'male'
                confidence = min(1.0, male_score / max(total_score, 1))
            elif female_score > male_score:
                prediction = 'female'
                confidence = min(1.0, female_score / max(total_score, 1))
            else:
                prediction = 'unknown'
                confidence = 0.5

            return {
                'gender_prediction': prediction,
                'confidence': confidence,
                'male_score': male_score,
                'female_score': female_score,
                'has_gender_indicators': has_indicators
            }

        elif self.method == 'llm':
            prompt = f"""Based on the text below, infer the likely gender of the author. Consider self-references, pronouns, and context clues.

Text: "{text}"

Respond in this exact format:
Gender: [male/female/unknown]
Confidence: [score from 0 to 1]
Explanation: [brief explanation]"""

            try:
                response = self.llm_client.generate(prompt, temperature=0.0)

                # Parse response
                prediction = 'unknown'
                confidence = 0.0

                lines = response.strip().split('\n')
                for line in lines:
                    if line.startswith('Gender:'):
                        prediction = line.split(':', 1)[1].strip().lower()
                    elif line.startswith('Confidence:'):
                        try:
                            confidence = float(line.split(':', 1)[1].strip())
                        except:
                            pass

                # Calculate scores based on prediction
                male_score = confidence if prediction == 'male' else 0.0
                female_score = confidence if prediction == 'female' else 0.0

                return {
                    'gender_prediction': prediction,
                    'confidence': confidence,
                    'male_score': male_score,
                    'female_score': female_score,
                    'has_gender_indicators': confidence > 0.3
                }

            except Exception as e:
                print(f"Error in LLM gender analysis: {e}")
                return {
                    'gender_prediction': 'unknown',
                    'confidence': 0.0,
                    'male_score': 0.0,
                    'female_score': 0.0,
                    'has_gender_indicators': False
                }

        else:
            raise NotImplementedError(f"Method {self.method} not yet implemented")

    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """Analyze gender for batch of texts."""
        results = [self.analyze(text) for text in texts]
        return pd.DataFrame(results)


class PoliticalLeaningAnalyzer:
    """Analyze political leaning/ideology from tweet content."""

    def __init__(self, method: str = 'keyword', llm_client=None):
        """
        Initialize political leaning analyzer.

        Args:
            method: 'keyword' or 'llm'
            llm_client: LLM client instance (required if method='llm')
        """
        self.method = method
        self.llm_client = llm_client

        if method == 'llm' and llm_client is None:
            raise ValueError("LLM client required for LLM-based political analysis")

        # Political keywords and phrases
        self.political_indicators = {
            'left': {
                'values': ['progressive', 'liberal', 'equality', 'social justice',
                          'climate change', 'climate crisis', 'healthcare for all',
                          'universal healthcare', 'raise minimum wage', 'tax the rich',
                          'workers rights', 'union', 'blm', 'black lives matter',
                          'lgbtq', 'immigration reform', 'gun control'],
                'figures': ['bernie', 'sanders', 'aoc', 'warren', 'biden', 'obama',
                           'pelosi', 'democrat', 'democrats'],
                'media': ['cnn', 'msnbc', 'nytimes', 'wapo'],
                'negative_right': ['maga', 'trumpism', 'far-right', 'alt-right']
            },
            'right': {
                'values': ['conservative', 'traditional', 'freedom', 'liberty',
                          'small government', 'free market', 'second amendment',
                          '2a', 'pro-life', 'border security', 'law and order',
                          'back the blue', 'america first', 'patriot'],
                'figures': ['trump', 'desantis', 'cruz', 'mcconnell', 'republican',
                           'republicans', 'gop'],
                'media': ['fox', 'fox news', 'breitbart', 'daily wire'],
                'negative_left': ['socialism', 'communist', 'far-left', 'radical left',
                                 'woke', 'cancel culture']
            }
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze political leaning from text.

        Returns:
            Dict with:
            - political_leaning: 'left', 'right', 'center', 'unknown'
            - confidence: float (0-1)
            - left_score: float
            - right_score: float
            - is_political: bool
        """
        # Handle missing/invalid text
        if not isinstance(text, str) or not text.strip():
            return {
                'political_leaning': 'unknown',
                'confidence': 0.0,
                'left_score': 0.0,
                'right_score': 0.0,
                'is_political': False
            }

        if self.method == 'keyword':
            text_lower = text.lower()

            left_score = 0
            right_score = 0

            # Check left indicators
            for category in self.political_indicators['left'].values():
                for term in category:
                    if term in text_lower:
                        left_score += 1

            # Check right indicators
            for category in self.political_indicators['right'].values():
                for term in category:
                    if term in text_lower:
                        right_score += 1

            # Determine prediction
            total_score = left_score + right_score
            is_political = total_score > 0

            if total_score == 0:
                prediction = 'unknown'
                confidence = 0.0
            elif abs(left_score - right_score) < 2:
                # Too close to call or mentions both sides
                prediction = 'center'
                confidence = 0.5
            elif left_score > right_score:
                prediction = 'left'
                confidence = min(1.0, left_score / max(total_score, 1))
            else:
                prediction = 'right'
                confidence = min(1.0, right_score / max(total_score, 1))

            return {
                'political_leaning': prediction,
                'confidence': confidence,
                'left_score': left_score,
                'right_score': right_score,
                'is_political': is_political
            }

        elif self.method == 'llm':
            prompt = f"""Analyze the political leaning expressed in the following text. Consider the values, figures, and language used.

Text: "{text}"

Respond in this exact format:
Political_Leaning: [left/right/center/unknown]
Confidence: [score from 0 to 1]
Is_Political: [true/false]
Explanation: [brief explanation]"""

            try:
                response = self.llm_client.generate(prompt, temperature=0.0)

                # Parse response
                prediction = 'unknown'
                confidence = 0.0
                is_political = False

                lines = response.strip().split('\n')
                for line in lines:
                    if line.startswith('Political_Leaning:'):
                        prediction = line.split(':', 1)[1].strip().lower()
                    elif line.startswith('Confidence:'):
                        try:
                            confidence = float(line.split(':', 1)[1].strip())
                        except:
                            pass
                    elif line.startswith('Is_Political:'):
                        is_political = 'true' in line.lower()

                # Calculate scores based on prediction
                left_score = confidence if prediction == 'left' else 0.0
                right_score = confidence if prediction == 'right' else 0.0

                return {
                    'political_leaning': prediction,
                    'confidence': confidence,
                    'left_score': left_score,
                    'right_score': right_score,
                    'is_political': is_political
                }

            except Exception as e:
                print(f"Error in LLM political analysis: {e}")
                return {
                    'political_leaning': 'unknown',
                    'confidence': 0.0,
                    'left_score': 0.0,
                    'right_score': 0.0,
                    'is_political': False
                }

        else:
            raise NotImplementedError(f"Method {self.method} not yet implemented")

    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """Analyze political leaning for batch of texts."""
        results = [self.analyze(text) for text in texts]
        return pd.DataFrame(results)


class MetadataInferenceEngine:
    """
    Main interface for inferring all metadata from tweets.

    Combines sentiment, topic, style, polarization, gender, and political leaning analysis.
    """

    def __init__(self,
                 sentiment_method: str = 'vader',
                 topic_method: str = 'keyword',
                 gender_method: str = 'keyword',
                 political_method: str = 'keyword',
                 include_gender: bool = True,
                 include_political: bool = True,
                 llm_client=None):
        """
        Initialize inference engine.

        Args:
            sentiment_method: 'textblob', 'vader', or 'llm'
            topic_method: 'roberta' (Cardiff NLP), 'keyword', or 'llm'
            gender_method: 'keyword' or 'llm'
            political_method: 'keyword' or 'llm'
            include_gender: Whether to include gender inference
            include_political: Whether to include political leaning inference
            llm_client: LLM client (required if any method is 'llm')
        """
        # Check if LLM client is needed
        llm_methods = [sentiment_method, gender_method, political_method]
        if topic_method == 'llm':
            llm_methods.append('llm')
        if 'llm' in llm_methods and llm_client is None:
            raise ValueError("LLM client required when using 'llm' method for any analyzer")

        self.topic_method = topic_method
        self.sentiment_analyzer = SentimentAnalyzer(method=sentiment_method, llm_client=llm_client)
        # For 'roberta', topic/polarization are computed in batch by add_metadata_to_dataframe
        _kw_method = topic_method if topic_method != 'roberta' else 'keyword'
        self.topic_classifier = TopicClassifier(method=_kw_method)
        self.style_analyzer = StyleAnalyzer()
        self.polarization_analyzer = PolarizationAnalyzer()
        self.include_gender = include_gender
        self.include_political = include_political
        if topic_method == 'roberta':
            self.cardiffnlp = CardiffNLPClassifier()

        if include_gender:
            self.gender_analyzer = GenderAnalyzer(method=gender_method, llm_client=llm_client)
        if include_political:
            self.political_analyzer = PoliticalLeaningAnalyzer(method=political_method, llm_client=llm_client)

    def infer(self, text: str) -> Dict[str, Any]:
        """
        Infer all metadata for a single text.

        Returns:
            Dict with all inferred attributes
        """
        # Handle missing/invalid text - convert to string and check
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        text = text.strip()

        sentiment = self.sentiment_analyzer.analyze(text)
        topics = self.topic_classifier.classify(text, top_k=3)
        style = self.style_analyzer.analyze(text)
        polarization = self.polarization_analyzer.analyze(text)

        # Optional analyses
        if self.include_gender:
            gender = self.gender_analyzer.analyze(text)
        if self.include_political:
            political = self.political_analyzer.analyze(text)

        # Combine results
        result = {
            # Text
            'text': text,
            'text_length': len(text),

            # Sentiment
            'sentiment_polarity': sentiment['polarity'],
            'sentiment_subjectivity': sentiment['subjectivity'],
            'sentiment_label': sentiment['label'],
            'sentiment_positive': sentiment['positive_score'],
            'sentiment_negative': sentiment['negative_score'],
            'sentiment_neutral': sentiment['neutral_score'],

            # Topics
            'primary_topic': topics[0]['topic'] if topics else 'other',
            'primary_topic_score': topics[0]['score'] if topics else 0,
            'all_topics': [t['topic'] for t in topics],
            'all_topic_scores': [t['score'] for t in topics],

            # Style
            'has_emoji': style['has_emoji'],
            'has_hashtag': style['has_hashtag'],
            'has_mention': style['has_mention'],
            'has_url': style['has_url'],
            'word_count': style['word_count'],
            'avg_word_length': style['avg_word_length'],
            'formality_score': style['formality_score'],

            # Polarization
            'polarization_score': polarization['polarization_score'],
            'has_polarizing_content': polarization['has_polarizing_content'],
            'controversy_level': polarization['controversy_level']
        }

        # Add gender analysis if enabled
        if self.include_gender:
            result.update({
                'gender_prediction': gender['gender_prediction'],
                'gender_confidence': gender['confidence'],
                'gender_male_score': gender['male_score'],
                'gender_female_score': gender['female_score'],
                'has_gender_indicators': gender['has_gender_indicators']
            })

        # Add political analysis if enabled
        if self.include_political:
            result.update({
                'political_leaning': political['political_leaning'],
                'political_confidence': political['confidence'],
                'political_left_score': political['left_score'],
                'political_right_score': political['right_score'],
                'is_political': political['is_political']
            })

        return result

    def infer_batch(self, texts: List[str], verbose: bool = True) -> pd.DataFrame:
        """
        Infer metadata for batch of texts.

        Args:
            texts: List of text strings
            verbose: Print progress

        Returns:
            DataFrame with all inferred metadata
        """
        if verbose:
            print(f"Inferring metadata for {len(texts)} texts...")

        results = []
        for i, text in enumerate(texts):
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(texts)} texts")

            results.append(self.infer(text))

        df = pd.DataFrame(results)

        if verbose:
            print(f"Completed metadata inference for {len(df)} texts")

        return df

    def add_metadata_to_dataframe(self, df: pd.DataFrame,
                                   text_column: str = 'text',
                                   verbose: bool = True) -> pd.DataFrame:
        """
        Add inferred metadata to existing DataFrame.

        Args:
            df: DataFrame with tweets
            text_column: Name of column with text
            verbose: Print progress

        Returns:
            DataFrame with metadata columns added
        """
        # Infer metadata (sentiment, style, toxicity; topic/polarization are
        # keyword placeholders that will be overwritten if topic_method='roberta')
        metadata_df = self.infer_batch(df[text_column].tolist(), verbose=verbose)

        # Drop duplicate 'text' column from metadata
        metadata_df = metadata_df.drop('text', axis=1, errors='ignore')

        # Reset indices to ensure proper alignment
        df_reset = df.reset_index(drop=True)
        metadata_df_reset = metadata_df.reset_index(drop=True)

        # Concatenate with proper index alignment
        result_df = pd.concat([df_reset, metadata_df_reset], axis=1)

        # Overwrite topic and polarization with Cardiff NLP RoBERTa scores
        if self.topic_method == 'roberta':
            roberta_df = self.cardiffnlp.classify_dataframe(
                df_reset[text_column]
            )
            result_df["primary_topic"]      = roberta_df["primary_topic"].values
            result_df["polarization_score"] = roberta_df["polarization_score"].values

        # Compute toxicity with Detoxify
        try:
            from detoxify import Detoxify
            print("  Computing toxicity with Detoxify ...")
            scores = Detoxify('original').predict(df_reset[text_column].astype(str).tolist())
            result_df["toxicity"] = scores["toxicity"]
        except ImportError:
            print("  WARNING: detoxify not installed — toxicity set to NaN. "
                  "Run: pip install detoxify")
            result_df["toxicity"] = float("nan")

        return result_df


# Convenience function
def infer_tweet_metadata(tweets_df: pd.DataFrame,
                         text_column: str = 'text',
                         sentiment_method: str = 'vader',
                         topic_method: str = 'roberta',
                         gender_method: str = 'keyword',
                         political_method: str = 'keyword',
                         include_gender: bool = True,
                         include_political: bool = True,
                         llm_client=None) -> pd.DataFrame:
    """
    Convenience function to add metadata to tweet DataFrame.

    Args:
        tweets_df: DataFrame with tweets
        text_column: Name of column containing tweet text
        sentiment_method: 'textblob', 'vader', or 'llm'
        topic_method: 'keyword', 'lda', or 'llm'
        gender_method: 'keyword' or 'llm'
        political_method: 'keyword' or 'llm'
        include_gender: Whether to include gender inference
        include_political: Whether to include political leaning inference
        llm_client: LLM client (required if any method is 'llm')

    Returns:
        DataFrame with metadata columns added
    """
    engine = MetadataInferenceEngine(
        sentiment_method=sentiment_method,
        topic_method=topic_method,
        gender_method=gender_method,
        political_method=political_method,
        include_gender=include_gender,
        include_political=include_political,
        llm_client=llm_client
    )

    return engine.add_metadata_to_dataframe(tweets_df, text_column=text_column)
