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
package org.springframework.ai.testcontainers.service.connection.weaviate;

import org.springframework.ai.autoconfigure.vectorstore.weaviate.WeaviateConnectionDetails;
import org.springframework.boot.testcontainers.service.connection.ContainerConnectionDetailsFactory;
import org.springframework.boot.testcontainers.service.connection.ContainerConnectionSource;
import org.testcontainers.weaviate.WeaviateContainer;

/**
 * @author Eddú Meléndez
 */
class WeaviateContainerConnectionDetailsFactory
		extends ContainerConnectionDetailsFactory<WeaviateContainer, WeaviateConnectionDetails> {

	@Override
	public WeaviateConnectionDetails getContainerConnectionDetails(
			ContainerConnectionSource<WeaviateContainer> source) {
		return new WeaviateContainerConnectionDetails(source);
	}

	/**
	 * {@link WeaviateConnectionDetails} backed by a {@link ContainerConnectionSource}.
	 */
	private static final class WeaviateContainerConnectionDetails extends ContainerConnectionDetails<WeaviateContainer>
			implements WeaviateConnectionDetails {

		private WeaviateContainerConnectionDetails(ContainerConnectionSource<WeaviateContainer> source) {
			super(source);
		}

		@Override
		public String getHost() {
			return getContainer().getHttpHostAddress();
		}

	}

}
