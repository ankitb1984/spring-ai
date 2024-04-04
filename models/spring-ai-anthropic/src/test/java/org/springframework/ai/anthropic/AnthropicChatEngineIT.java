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
package org.springframework.ai.anthropic;

import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import org.springframework.ai.chat.BaseChatEngineTests;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

/**
 * @author Christian Tzolov
 */
@SpringBootTest(classes = AnthropicTestConfiguration.class)
@EnabledIfEnvironmentVariable(named = "ANTHROPIC_API_KEY", matches = ".+")
public class AnthropicChatEngineIT extends BaseChatEngineTests {

	@Autowired
	AnthropicChatEngineIT(AnthropicChatClient chatClient) {
		super(chatClient, chatClient);
	}

}
