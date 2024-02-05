/*
 * Copyright 2024-2024 the original author or authors.
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

package org.springframework.ai.openai.chat.api.tool;

import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.Role;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.ToolCall;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionRequest;
import org.springframework.ai.openai.api.OpenAiApi.FunctionTool.Type;
import org.springframework.http.ResponseEntity;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Based on the OpenAI Function Calling tutorial:
 * https://platform.openai.com/docs/guides/function-calling/parallel-function-calling
 *
 * @author Christian Tzolov
 */
public class OpenAiApiToolFunctionCallTests {

	private final Logger logger = LoggerFactory.getLogger(OpenAiApiToolFunctionCallTests.class);

	FakeWeatherService weatherService = new FakeWeatherService();

	OpenAiApi completionApi = new OpenAiApi(System.getenv("OPENAI_API_KEY"));

	@Test
	public void toolFunctionCall() {

		// Step 1: send the conversation and available functions to the model
		var message = new ChatCompletionMessage("What's the weather like in San Francisco, Tokyo, and Paris?",
				Role.USER);

		var functionTool = new OpenAiApi.FunctionTool(Type.FUNCTION,
				new OpenAiApi.FunctionTool.Function(
						"Get the weather in location. Return temperature in 30°F or 30°C format.", "getCurrentWeather",
						OpenAiApi.parseJson("""
								{
									"type": "object",
									"properties": {
										"location": {
											"type": "string",
											"description": "The city and state e.g. San Francisco, CA"
										},
										"lat": {
											"type": "number",
											"description": "The city latitude"
										},
										"lon": {
											"type": "number",
											"description": "The city longitude"
										},
										"unit": {
											"type": "string",
											"enum": ["c", "f"]
										}
									},
									"required": ["location", "lat", "lon", "unit"]
								}
								""")));

		// Or you can use the
		// ModelOptionsUtils.getJsonSchema(FakeWeatherService.Request.class))) to
		// auto-generate the JSON schema like:
		// var functionTool = new OpenAiApi.FunctionTool(Type.FUNCTION, new
		// OpenAiApi.FunctionTool.Function(
		// "Get the weather in location. Return temperature in 30°F or 30°C format.",
		// "getCurrentWeather",
		// ModelOptionsUtils.getJsonSchema(FakeWeatherService.Request.class)));

		List<ChatCompletionMessage> messages = new ArrayList<>(List.of(message));

		ChatCompletionRequest chatCompletionRequest = new ChatCompletionRequest(messages, "gpt-4-1106-preview",
				List.of(functionTool), null);

		ResponseEntity<ChatCompletion> chatCompletion = completionApi.chatCompletionEntity(chatCompletionRequest);

		assertThat(chatCompletion.getBody()).isNotNull();
		assertThat(chatCompletion.getBody().choices()).isNotEmpty();

		ChatCompletionMessage responseMessage = chatCompletion.getBody().choices().get(0).message();

		assertThat(responseMessage.role()).isEqualTo(Role.ASSISTANT);
		assertThat(responseMessage.toolCalls()).isNotNull();

		// Check if the model wanted to call a function
		if (responseMessage.toolCalls() != null) {

			// extend conversation with assistant's reply.
			messages.add(responseMessage);

			// Send the info for each function call and function response to the model.
			for (ToolCall toolCall : responseMessage.toolCalls()) {
				var functionName = toolCall.function().name();
				if ("getCurrentWeather".equals(functionName)) {
					FakeWeatherService.Request weatherRequest = fromJson(toolCall.function().arguments(),
							FakeWeatherService.Request.class);

					FakeWeatherService.Response weatherResponse = weatherService.apply(weatherRequest);

					// extend conversation with function response.
					messages.add(new ChatCompletionMessage("" + weatherResponse.temp() + weatherRequest.unit(),
							Role.TOOL, null, toolCall.id(), null));
				}
			}

			var functionResponseRequest = new ChatCompletionRequest(messages, "gpt-4-1106-preview", 0.8f);

			ResponseEntity<ChatCompletion> chatCompletion2 = completionApi
				.chatCompletionEntity(functionResponseRequest);

			logger.info("Final response: " + chatCompletion2.getBody());

			assertThat(chatCompletion2.getBody().choices()).isNotEmpty();

			assertThat(chatCompletion2.getBody().choices().get(0).message().role()).isEqualTo(Role.ASSISTANT);
			assertThat(chatCompletion2.getBody().choices().get(0).message().content()).contains("San Francisco")
				.containsAnyOf("30.0°F", "30°F");
			assertThat(chatCompletion2.getBody().choices().get(0).message().content()).contains("Tokyo")
				.containsAnyOf("10.0°C", "10°C");
			;
			assertThat(chatCompletion2.getBody().choices().get(0).message().content()).contains("Paris")
				.containsAnyOf("15.0°C", "15°C");
			;
		}

	}

	private static <T> T fromJson(String json, Class<T> targetClass) {
		try {
			return new ObjectMapper().readValue(json, targetClass);
		}
		catch (JsonProcessingException e) {
			throw new RuntimeException(e);
		}
	}

}