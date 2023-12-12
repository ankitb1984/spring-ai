/*
 * Copyright 2023-2023 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.vertex.embedding;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.vertex.api.VertexAiApi;

/**
 *
 * @author Christian Tzolov
 */
public class VertexAiEmbeddingClient implements EmbeddingClient {

	private final VertexAiApi vertexAiApi;

	public VertexAiEmbeddingClient(VertexAiApi vertexAiApi) {
		this.vertexAiApi = vertexAiApi;
	}

	@Override
	public List<Double> embed(String text) {
		return this.vertexAiApi.embedText(text).value();
	}

	@Override
	public List<Double> embed(Document document) {
		return embed(document.getContent());
	}

	@Override
	public List<List<Double>> embed(List<String> texts) {
		List<VertexAiApi.Embedding> vertexEmbeddings = this.vertexAiApi.batchEmbedText(texts);
		return vertexEmbeddings.stream().map(e -> e.value()).toList();
	}

	@Override
	public EmbeddingResponse embedForResponse(List<String> texts) {
		List<VertexAiApi.Embedding> vertexEmbeddings = this.vertexAiApi.batchEmbedText(texts);
		int index = 0;
		List<Embedding> embeddings = new ArrayList<>();
		for (VertexAiApi.Embedding vertexEmbedding : vertexEmbeddings) {
			embeddings.add(new Embedding(vertexEmbedding.value(), index++));
		}
		return new EmbeddingResponse(embeddings, Map.of());
	}

}
