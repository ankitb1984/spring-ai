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

package org.springframework.ai.openai;

import org.junit.jupiter.api.Test;

import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.openai.api.OpenAiChatOptions;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * @author Christian Tzolov
 */
public class ChatCompletionRequestIT {

	@Test
	public void chatOptions() {

		var client = new OpenAiChatClient(new OpenAiApi("TEST"))
			.withDefaultOptions(OpenAiChatOptions.builder().withModel("DEFAULT_MODEL").withTemperature(66.6f).build());

		var request = client.createRequest(new Prompt("Test message content"), false);

		assertThat(request.getMessages()).hasSize(1);
		assertThat(request.getStream()).isFalse();

		assertThat(request.getModel()).isEqualTo("DEFAULT_MODEL");
		assertThat(request.getTemperature()).isEqualTo(66.6f);

		request = client.createRequest(new Prompt("Test message content",
				OpenAiChatOptions.builder().withModel("PROMPT_MODEL").withTemperature(99.9f).build()), true);

		assertThat(request.getMessages()).hasSize(1);
		assertThat(request.getStream()).isTrue();

		assertThat(request.getModel()).isEqualTo("PROMPT_MODEL");
		assertThat(request.getTemperature()).isEqualTo(99.9f);
	}

}
