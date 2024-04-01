/*
 * Copyright 2024 - 2024 the original author or authors.
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

import com.fasterxml.jackson.core.JsonProcessingException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingClient;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * The <b>SAP HANA Cloud vector engine</b> offers multiple use cases in AI scenarios.
 * <p/>
 * <p>
 * Recent advances in Generative AI (GenAI) and Large Language Models (LLM) have led to increased awareness of and
 * popularity for vector databases. Similarity search, a key functionality of vector databases, complements traditional
 * relational databases as well as full-text search systems. Using natural language text as an example, embedding
 * functions map data to high dimensional vectors to preserve their semantic similarity. Developers can then use
 * vector-based semantic search to find similarity between different passages of text. Because the data within an
 * LLM is current only up to a specific point in time, vector databases can offer additional relevant text to make
 * searches more accurate – known as <b>Retrieval Augmented Generation</b> (RAG). Therefore, the addition of RAG to an LLM using
 * a vector database like SAP HANA Cloud provides an effective approach to increase the quality of responses from an LLM.
 * <p/>
 * <p>
 * The SAP HANA Cloud vector engine supports the create, read, update, and delete (CRUD) operations involving vectors using SQL.
 * <p/>
 *
 * <code>HanaCloudVectorStore</code> is an implementation of <code>org.springframework.ai.vectorstore.VectorStore</code>
 * interface that provides implementation of <code>COSINE_SIMILARITY</code> function introduced in HanaDB in Mar, 2024
 * <p/>
 * <p>
 * Hana DB introduced a new datatype <code>REAL_VECTOR</code> that can store embeddings generated by
 * <code>org.springframework.ai.embedding.EmbeddingClient</code>
 * <p/>
 *
 * @see <a href="https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-vector-engine-guide/introduction">SAP HANA Database Vector Engine Guide</a>
 *
 * @author Rahul Mittal
 * @since 1.0.0
 */
@Slf4j
public class HanaCloudVectorStore implements VectorStore {

    private final HanaVectorRepository<? extends HanaVectorEntity> repository;

    private final EmbeddingClient embeddingClient;

    private final HanaCloudVectorStoreConfig config;

    public HanaCloudVectorStore(HanaVectorRepository<? extends HanaVectorEntity> repository,
                                EmbeddingClient embeddingClient, HanaCloudVectorStoreConfig config) {
        this.repository = repository;
        this.embeddingClient = embeddingClient;
        this.config = config;
    }

    @Override
    public void add(List<Document> documents) {
        int count = 1;
        for (Document document : documents) {
            log.info("[{}/{}] Calling EmbeddingClient for document id = {}", count++, documents.size(),
                    document.getId());
            String content = document.getContent().replaceAll("\\s+", " ");
            String embedding = getEmbedding(document);
            repository.save(config.getTableName(), document.getId(), embedding, content);
        }
        log.info("Embeddings saved in HanaCloudVectorStore for {} documents", count - 1);
    }

    @Override
    public Optional<Boolean> delete(List<String> idList) {
        int deleteCount = repository.deleteEmbeddingsById(config.getTableName(), idList);
        log.info("{} embeddings deleted", deleteCount);
        return Optional.of(deleteCount == idList.size());
    }

    public int purgeEmbeddings() {
        int deleteCount = repository.deleteAllEmbeddings(config.getTableName());
        log.info("{} embeddings deleted", deleteCount);
        return deleteCount;
    }

    @Override
    public List<Document> similaritySearch(String query) {
        return similaritySearch(SearchRequest.query(query).withTopK(config.getTopK()));
    }

    @Override
    public List<Document> similaritySearch(SearchRequest request) {
        String queryEmbedding = getEmbedding(request);
        List<? extends HanaVectorEntity> searchResult = repository.cosineSimilaritySearch(config.getTableName(),
                request.getTopK(), queryEmbedding);
        log.info("Hana cosine-similarity returned {} results for topK={}", searchResult.size(), request.getTopK());
        return searchResult.stream().map(c -> {
            try {
                return new Document(c.get_id(), c.toJson(), Collections.emptyMap());
            } catch (JsonProcessingException e) {
                throw new RuntimeException(e);
            }
        }).collect(Collectors.toList());
    }

    private String getEmbedding(SearchRequest searchRequest) {
        return "[" + this.embeddingClient.embed(searchRequest.getQuery())
                .stream()
                .map(String::valueOf)
                .collect(Collectors.joining(", ")) + "]";
    }

    private String getEmbedding(Document document) {
        return "["
                + this.embeddingClient.embed(document).stream().map(String::valueOf).collect(Collectors.joining(", "))
                + "]";
    }

}
