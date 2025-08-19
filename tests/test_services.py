import pytest
import os
from unittest.mock import Mock, patch

# Prevent OpenAI client initialization during import
with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
  from obsidian_ai.services.openai_client import OpenAIClient, OpenAIError
  from obsidian_ai.services.embedding_service import EmbeddingService


class TestOpenAIClient:
  @patch("obsidian_ai.services.openai_client.OpenAI")
  def test_init_with_api_key(self, mock_openai):
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
      client = OpenAIClient()
      mock_openai.assert_called_once()

  @patch("obsidian_ai.services.openai_client.OpenAI")
  def test_chat_completion_success(self, mock_openai):
    mock_client_instance = Mock()
    mock_openai.return_value = mock_client_instance

    # Mock successful response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response"
    mock_client_instance.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
      client = OpenAIClient()
      result = client.chat_completion([{"role": "user", "content": "test"}])

    assert result == mock_response
    mock_client_instance.chat.completions.create.assert_called_once()

  @patch("obsidian_ai.services.openai_client.OpenAI")
  def test_chat_completion_error(self, mock_openai):
    mock_client_instance = Mock()
    mock_openai.return_value = mock_client_instance
    mock_client_instance.chat.completions.create.side_effect = Exception("API Error")

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
      client = OpenAIClient()

      with pytest.raises(OpenAIError, match="Failed to get OpenAI response"):
        client.chat_completion([{"role": "user", "content": "test"}])

  @patch("obsidian_ai.services.openai_client.OpenAI")
  def test_simple_completion(self, mock_openai):
    mock_client_instance = Mock()
    mock_openai.return_value = mock_client_instance

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Simple response"
    mock_client_instance.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
      client = OpenAIClient()
      result = client.simple_completion("test prompt")

    assert result == "Simple response"

  @patch("obsidian_ai.services.openai_client.OpenAI")
  def test_simple_completion_error_handling(self, mock_openai):
    mock_client_instance = Mock()
    mock_openai.return_value = mock_client_instance
    mock_client_instance.chat.completions.create.side_effect = Exception("API Error")

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
      client = OpenAIClient()
      result = client.simple_completion("test prompt")

    assert "Error:" in result
    assert "API Error" in result

  @patch("obsidian_ai.services.openai_client.OpenAI")
  def test_structured_completion(self, mock_openai):
    mock_client_instance = Mock()
    mock_openai.return_value = mock_client_instance

    # Mock tool calls
    mock_tool_call = Mock()
    mock_tool_call.model_dump.return_value = {"id": "call_123", "function": {"name": "test"}}

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Structured response"
    mock_response.choices[0].message.tool_calls = [mock_tool_call]
    mock_client_instance.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
      client = OpenAIClient()
      content, tool_calls = client.structured_completion("system", "user", [])

    assert content == "Structured response"
    assert tool_calls is not None
    assert len(tool_calls) == 1

  @patch("obsidian_ai.services.openai_client.OpenAI")
  def test_continue_conversation(self, mock_openai):
    mock_client_instance = Mock()
    mock_openai.return_value = mock_client_instance

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Continued response"
    mock_client_instance.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
      client = OpenAIClient()
      result = client.continue_conversation([{"role": "user", "content": "continue"}])

    assert result == "Continued response"


class TestEmbeddingService:
  def test_init_creates_cache_dir(self, temp_brain):
    cache_dir = temp_brain / ".cache"
    service = EmbeddingService(cache_dir)
    assert service.cache_dir == cache_dir
    assert cache_dir.exists()

  def test_create_embeddings_basic(self, temp_brain):
    cache_dir = temp_brain / ".cache"
    service = EmbeddingService(cache_dir)

    documents = [
      "Machine learning is a subset of artificial intelligence",
      "Deep learning uses neural networks with multiple layers",
      "Python is a popular programming language",
    ]

    embeddings = service.create_embeddings(documents)

    assert len(embeddings) == 3
    assert all(len(emb) > 0 for emb in embeddings)
    # Each embedding should be a list/array of numbers
    assert all(isinstance(emb, (list, tuple)) for emb in embeddings)

  def test_create_embeddings_empty(self, temp_brain):
    cache_dir = temp_brain / ".cache"
    service = EmbeddingService(cache_dir)

    embeddings = service.create_embeddings([])
    assert embeddings == []

  def test_search_similar_basic(self, temp_brain):
    cache_dir = temp_brain / ".cache"
    service = EmbeddingService(cache_dir)

    documents = [
      "Machine learning algorithms for data analysis",
      "Deep neural networks and artificial intelligence",
      "Python programming and software development",
      "Statistical analysis and data science",
      "Web development with JavaScript",
    ]

    # Create embeddings first
    service.create_embeddings(documents)

    # Search for similar documents
    results = service.search_similar("machine learning data", top_k=3)

    assert len(results) <= 3
    assert all(isinstance(result, tuple) and len(result) == 2 for result in results)
    assert all(isinstance(idx, int) and isinstance(score, float) for idx, score in results)
    assert all(0 <= score <= 1 for _, score in results)

  def test_search_similar_no_embeddings(self, temp_brain):
    cache_dir = temp_brain / ".cache"
    service = EmbeddingService(cache_dir)

    # Search without creating embeddings first
    results = service.search_similar("test query", top_k=5)
    assert results == []

  def test_search_similar_empty_query(self, temp_brain):
    cache_dir = temp_brain / ".cache"
    service = EmbeddingService(cache_dir)

    documents = ["Test document one", "Test document two"]
    service.create_embeddings(documents)

    results = service.search_similar("", top_k=2)
    # Should handle empty query gracefully
    assert isinstance(results, list)

  def test_caching_behavior(self, temp_brain):
    cache_dir = temp_brain / ".cache"

    documents = ["Cached document test"]

    # Create embeddings with first service instance
    service1 = EmbeddingService(cache_dir)
    embeddings1 = service1.create_embeddings(documents)

    # Create new service instance - should use cached embeddings
    service2 = EmbeddingService(cache_dir)
    embeddings2 = service2.create_embeddings(documents)

    # Should get same results (from cache)
    assert len(embeddings1) == len(embeddings2)
    assert len(embeddings1) == 1

  def test_large_document_set(self, temp_brain):
    cache_dir = temp_brain / ".cache"
    service = EmbeddingService(cache_dir)

    # Create 50 documents
    documents = [f"Document {i} about topic {i % 5}" for i in range(50)]

    embeddings = service.create_embeddings(documents)
    assert len(embeddings) == 50

    # Search in large set
    results = service.search_similar("topic 2", top_k=5)
    assert len(results) <= 5

  def test_special_characters_and_unicode(self, temp_brain):
    cache_dir = temp_brain / ".cache"
    service = EmbeddingService(cache_dir)

    documents = [
      "Document with special chars: @#$%^&*()",
      "Unicode text: café résumé naïve",
      "Mixed content: AI/ML & NLP research",
      "Normal text without special characters",
    ]

    embeddings = service.create_embeddings(documents)
    assert len(embeddings) == 4

    # Should handle special characters in search too
    results = service.search_similar("special chars", top_k=2)
    assert len(results) <= 2

  def test_duplicate_documents(self, temp_brain):
    cache_dir = temp_brain / ".cache"
    service = EmbeddingService(cache_dir)

    documents = ["Same document content", "Same document content", "Different document content", "Same document content"]

    embeddings = service.create_embeddings(documents)
    assert len(embeddings) == 4  # Should create embeddings for all, even duplicates

    results = service.search_similar("Same document", top_k=3)
    assert len(results) <= 3
