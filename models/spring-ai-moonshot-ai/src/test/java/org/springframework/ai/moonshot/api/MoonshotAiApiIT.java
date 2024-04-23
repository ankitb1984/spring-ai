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
package org.springframework.ai.moonshot.api;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import reactor.core.publisher.Flux;

import org.springframework.ai.moonshot.api.MoonshotAiApi.ChatCompletion;
import org.springframework.ai.moonshot.api.MoonshotAiApi.ChatCompletionChunk;
import org.springframework.ai.moonshot.api.MoonshotAiApi.ChatCompletionMessage;
import org.springframework.ai.moonshot.api.MoonshotAiApi.ChatCompletionMessage.Role;
import org.springframework.ai.moonshot.api.MoonshotAiApi.ChatCompletionRequest;
import org.springframework.http.ResponseEntity;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * @author Geng Rong
 */
@EnabledIfEnvironmentVariable(named = "MOONSHOT_AI_API_KEY", matches = ".+")
public class MoonshotAiApiIT {

	MoonshotAiApi moonshotAiApi = new MoonshotAiApi(System.getenv("MOONSHOT_AI_API_KEY"));

	@Test
	void chatCompletionEntity() {
		ChatCompletionMessage chatCompletionMessage = new ChatCompletionMessage("Hello world", Role.USER);
		ResponseEntity<ChatCompletion> response = moonshotAiApi.chatCompletionEntity(new ChatCompletionRequest(
				List.of(chatCompletionMessage), MoonshotAiApi.ChatModel.MOONSHOT_V1_8K.getValue(), 0.8f, false));

		assertThat(response).isNotNull();
		assertThat(response.getBody()).isNotNull();
	}

	@Test
	void chatCompletionEntityWithSystemMessage() {
		ChatCompletionMessage userMessage = new ChatCompletionMessage(
				"Tell me about 3 famous pirates from the Golden Age of Piracy and why they did?", Role.USER);
		ChatCompletionMessage systemMessage = new ChatCompletionMessage("""
				You are an AI assistant that helps people find information.
				Your name is Bob.
				You should reply to the user's request with your name and also in the style of a pirate.
					""", Role.SYSTEM);

		ResponseEntity<ChatCompletion> response = moonshotAiApi.chatCompletionEntity(new ChatCompletionRequest(
				List.of(systemMessage, userMessage), MoonshotAiApi.ChatModel.MOONSHOT_V1_8K.getValue(), 0.8f, false));

		assertThat(response).isNotNull();
		assertThat(response.getBody()).isNotNull();
	}

	@Test
	void chatCompletionStream() {
		ChatCompletionMessage chatCompletionMessage = new ChatCompletionMessage("Hello world", Role.USER);
		Flux<ChatCompletionChunk> response = moonshotAiApi.chatCompletionStream(new ChatCompletionRequest(
				List.of(chatCompletionMessage), MoonshotAiApi.ChatModel.MOONSHOT_V1_8K.getValue(), 0.8f, true));

		assertThat(response).isNotNull();
		assertThat(response.collectList().block()).isNotNull();
	}

}
