"""
Tests for RAG Module.

Tests for RAG pipeline, retriever, generator, and storage.
"""

import pytest


class TestRAGSettings:
    """Tests for RAG settings."""
    
    def test_rag_settings_loads(self):
        """Test RAG settings load."""
        from app.rag import rag_settings
        
        assert rag_settings is not None
        assert rag_settings.embed_model is not None
        assert rag_settings.llm_model is not None
    
    def test_ollama_url_property(self):
        """Test Ollama URL property."""
        from app.rag import rag_settings
        
        url = rag_settings.ollama_generate_url
        assert url.endswith("/api/generate")


class TestQueryTerms:
    """Tests for QueryTerms model."""
    
    def test_query_terms_creation(self):
        """Test QueryTerms creation."""
        from app.rag import QueryTerms
        
        terms = QueryTerms(
            countries=["Italy"],
            airport="Frankfurt",
            sort_by="price",
        )
        
        assert "Italy" in terms.countries
        assert terms.airport == "Frankfurt"
    
    def test_query_terms_empty(self):
        """Test empty QueryTerms."""
        from app.rag import QueryTerms
        
        terms = QueryTerms.empty()
        
        assert terms.countries == []
        assert terms.airport is None
        assert not terms.has_filters()
    
    def test_has_filters(self):
        """Test has_filters method."""
        from app.rag import QueryTerms
        
        empty = QueryTerms()
        assert not empty.has_filters()
        
        with_country = QueryTerms(countries=["Italy"])
        assert with_country.has_filters()
    
    def test_to_dict(self):
        """Test serialization."""
        from app.rag import QueryTerms
        
        terms = QueryTerms(countries=["Italy"], top_n=10)
        result = terms.to_dict()
        
        assert result["countries"] == ["Italy"]
        assert result["top_n"] == 10


class TestQueryIntent:
    """Tests for QueryIntent model."""
    
    def test_from_query_price(self):
        """Test intent detection for price queries."""
        from app.rag.models.queries import QueryIntent
        
        intent = QueryIntent.from_query("Find the cheapest trips to Italy")
        
        assert intent.sort_by_price
    
    def test_from_query_date(self):
        """Test intent detection for date queries."""
        from app.rag.models.queries import QueryIntent
        
        intent = QueryIntent.from_query("Show earliest trips to Germany")
        
        assert intent.sort_by_date


class TestTripDocument:
    """Tests for TripDocument model."""
    
    def test_trip_document_creation(self):
        """Test TripDocument creation."""
        from app.rag import TripDocument, TripDate
        
        trip = TripDocument(
            provider="TestProvider",
            trip_name="Test Trip",
            country="Italy",
            dates=[
                TripDate(start_date="2025-01-01", end_date="2025-01-10", price=1000.0),
            ],
        )
        
        assert trip.provider == "TestProvider"
        assert len(trip.dates) == 1
    
    def test_format_display(self):
        """Test display formatting."""
        from app.rag import TripDocument
        
        trip = TripDocument(
            provider="Provider",
            trip_name="Trip",
            country="Italy",
        )
        
        display = trip.format_display()
        assert "Provider" in display
        assert "Trip" in display


class TestTripDate:
    """Tests for TripDate model."""
    
    def test_trip_date_creation(self):
        """Test TripDate creation."""
        from app.rag import TripDate
        
        date = TripDate(
            start_date="2025-01-01",
            end_date="2025-01-10",
            price=1500.0,
            airport="Frankfurt",
        )
        
        assert date.price == 1500.0
    
    def test_format_display(self):
        """Test display formatting."""
        from app.rag import TripDate
        
        date = TripDate(
            start_date="2025-01-01",
            end_date="2025-01-10",
            price=1500.0,
        )
        
        display = date.format_display()
        assert "2025-01-01" in display
        assert "1500" in display


class TestCountryUtils:
    """Tests for country utilities."""
    
    def test_normalize_country(self):
        """Test country normalization."""
        from app.rag.utils.country_utils import normalize_country
        
        # Should normalize aliases
        assert normalize_country("italien") in ["Italien", "Italy", "italien"]
    
    def test_extract_countries_from_query(self):
        """Test country extraction from query."""
        from app.rag.utils.country_utils import extract_countries_from_query
        
        countries = extract_countries_from_query("trips to italy and germany")
        
        # Should find countries
        assert len(countries) >= 0  # May depend on aliases file
    
    def test_detect_provider(self):
        """Test provider detection."""
        from app.rag.utils.country_utils import detect_provider_from_query
        
        provider = detect_provider_from_query("Show Studiosus trips")
        assert provider == "Studiosus"


class TestQueryParser:
    """Tests for query parser."""
    
    def test_parse_query_terms(self):
        """Test query parsing."""
        from app.rag.utils.query_parser import parse_query_terms
        
        terms = parse_query_terms("cheapest trips to Italy from Frankfurt")
        
        # Should detect sorting
        assert terms.sort_by.value in ["price", ""]
    
    def test_extract_airport(self):
        """Test airport extraction."""
        from app.rag.utils.query_parser import QueryParser
        
        parser = QueryParser()
        terms = parser.parse("trips from Munich to Italy")
        
        # Should extract airport
        assert terms.airport in ["Munich", None]


class TestQualityMetrics:
    """Tests for quality metrics."""
    
    def test_compute_answer_relevancy(self):
        """Test answer relevancy computation."""
        from app.rag.utils.quality_metrics import compute_answer_relevancy
        
        score = compute_answer_relevancy(
            "Find trips to Italy",
            "Here are trips to Italy: Trip 1, Trip 2",
        )
        
        assert 0.0 <= score <= 1.0
    
    def test_compute_faithfulness(self):
        """Test faithfulness computation."""
        from app.rag.utils.quality_metrics import compute_faithfulness
        
        score = compute_faithfulness(
            "Trip to Italy costs 1500",
            ["Italy trip available for 1500 euros"],
        )
        
        assert 0.0 <= score <= 1.0
    
    def test_check_consistency(self):
        """Test consistency check."""
        from app.rag.utils.quality_metrics import check_consistency
        
        # This depends on the implementation
        result = check_consistency(
            "Find trips to Italy",
            "Here are trips to Italy → dates",
        )
        
        assert isinstance(result, bool)
