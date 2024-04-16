/*
 * Copyright 2023 - 2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.springframework.ai.vectorstore;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch._types.mapping.TypeMapping;
import co.elastic.clients.elasticsearch.core.BulkRequest;
import co.elastic.clients.elasticsearch.core.BulkResponse;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import co.elastic.clients.elasticsearch.core.search.Hit;
import co.elastic.clients.elasticsearch.indices.CreateIndexResponse;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.rest_client.RestClientTransport;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.elasticsearch.client.RestClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.filter.FilterExpressionConverter;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.util.Assert;

import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;

import static java.lang.Math.sqrt;

/**
 * @author Jemin Huh
 * @since 1.0.0
 */
public class ElasticsearchVectorStore implements VectorStore, InitializingBean {

	private static final Logger logger = LoggerFactory.getLogger(ElasticsearchVectorStore.class);

	private static final String INDEX_NAME = "spring-ai-document-index";

	private static final String EMBEDDING = "embedding";

	private static final String SIMILARITY_DEFAULT = "cosine";

	private final EmbeddingClient embeddingClient;

	private final ElasticsearchClient elasticsearchClient;

	private final String index;

	private String similarityFunction = SIMILARITY_DEFAULT;

	private final FilterExpressionConverter filterExpressionConverter;

	public ElasticsearchVectorStore(RestClient restClient, EmbeddingClient embeddingClient) {
		this(INDEX_NAME, restClient, embeddingClient);
	}

	public ElasticsearchVectorStore(String index, RestClient restClient, EmbeddingClient embeddingClient) {
		Objects.requireNonNull(embeddingClient, "RestClient must not be null");
		Objects.requireNonNull(embeddingClient, "EmbeddingClient must not be null");
		this.elasticsearchClient = new ElasticsearchClient(new RestClientTransport(restClient, new JacksonJsonpMapper(
				new ObjectMapper().configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false))));
		this.embeddingClient = embeddingClient;
		this.index = index;
		this.filterExpressionConverter = new ElasticsearchAiSearchFilterExpressionConverter();
	}

	@Override
	public void add(List<Document> documents) {
		BulkRequest.Builder builkRequestBuilder = new BulkRequest.Builder();
		for (Document document : documents) {
			if (Objects.isNull(document.getEmbedding()) || document.getEmbedding().isEmpty()) {
				logger.debug("Calling EmbeddingClient for document id = " + document.getId());
				document.setEmbedding(this.embeddingClient.embed(document));
			}
			builkRequestBuilder
				.operations(op -> op.index(idx -> idx.index(this.index).id(document.getId()).document(document)));
		}
		bulkRequest(builkRequestBuilder.build());
	}

	@Override
	public Optional<Boolean> delete(List<String> idList) {
		BulkRequest.Builder builkRequestBuilder = new BulkRequest.Builder();
		for (String id : idList)
			builkRequestBuilder.operations(op -> op.delete(idx -> idx.index(this.index).id(id)));
		return Optional.of(bulkRequest(builkRequestBuilder.build()).errors());
	}

	private BulkResponse bulkRequest(BulkRequest bulkRequest) {
		try {
			return this.elasticsearchClient.bulk(bulkRequest);
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public List<Document> similaritySearch(SearchRequest searchRequest) {
		Assert.notNull(searchRequest, "The search request must not be null.");
		try {
			float threshold = (float) searchRequest.getSimilarityThreshold();
			// reverting l2_norm distance to its original value
			if (similarityFunction.equals("l2_norm")) {
				threshold = 1 - threshold;
			}
			final float finalThreshold = threshold;
			List<Float> vectors = this.embeddingClient.embed(searchRequest.getQuery())
				.stream()
				.map(Double::floatValue)
				.toList();

			SearchResponse<Document> res = elasticsearchClient.search(
					sr -> sr.index(this.index)
						.knn(knn -> knn.queryVector(vectors)
							.similarity(finalThreshold)
							.k(searchRequest.getTopK())
							.field(EMBEDDING)
							.numCandidates((long) (1.5 * searchRequest.getTopK()))
							.filter(fl -> fl.queryString(
									qs -> qs.query(getElasticsearchQueryString(searchRequest.getFilterExpression()))))),
					Document.class);

			return res.hits().hits().stream().map(this::toDocument).collect(Collectors.toList());

		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	private String getElasticsearchQueryString(Filter.Expression filterExpression) {
		return Objects.isNull(filterExpression) ? "*"
				: this.filterExpressionConverter.convertExpression(filterExpression);

	}

	private Document toDocument(Hit<Document> hit) {
		Document document = hit.source();
		document.getMetadata().put("distance", calculateDistance(hit.score().floatValue()));
		return document;
	}

	// more info on score/distance calculation
	// https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#knn-similarity-search
	private float calculateDistance(Float score) {
		switch (similarityFunction) {
			case "l2_norm":
				// the returned value of l2_norm is the opposite of the other functions
				// (closest to zero means more accurate), so to make it consistent
				// with the other functions the reverse is returned applying a "1-"
				// to the standard transformation
				return (float) (1 - (sqrt((1 / score) - 1)));
			// cosine and dot_product
			default:
				return (2 * score) - 1;
		}
	}

	public boolean existsIndex() {
		try {
			return this.elasticsearchClient.indices().exists(ex -> ex.index(this.index)).value();
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	// possible similarity functions and mapping examples:
	// https://www.elastic.co/guide/en/elasticsearch/reference/master/dense-vector.html
	// max_inner_product is currently not supported because
	// the distance value is not normalized and would not
	// comply with the requirement of being between 0 and 1
	public CreateIndexResponse createIndexMapping(TypeMapping mapping) {
		try {
			this.similarityFunction = mapping.properties().get(EMBEDDING).denseVector().similarity();
			return this.elasticsearchClient.indices().create(cr -> cr.index(this.index).mappings(mapping));
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void afterPropertiesSet() {
	}

}
