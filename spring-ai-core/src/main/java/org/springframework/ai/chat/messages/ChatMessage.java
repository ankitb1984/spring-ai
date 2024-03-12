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
package org.springframework.ai.chat.messages;

import java.util.Map;

/**
 * Represents a chat message in a chat application.
 *
 */
public class ChatMessage extends AbstractMessage {

	public ChatMessage(String role, String content) {
		super(MessageType.valueOf(role), content);
	}

	public ChatMessage(String role, String content, Map<String, Object> properties) {
		super(MessageType.valueOf(role), content, properties);
	}

	public ChatMessage(MessageType messageType, String content) {
		super(messageType, content);
	}

	public ChatMessage(MessageType messageType, String content, Map<String, Object> properties) {
		super(messageType, content, properties);
	}

}
