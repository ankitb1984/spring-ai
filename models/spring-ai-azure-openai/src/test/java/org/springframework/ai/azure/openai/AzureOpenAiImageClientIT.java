package org.springframework.ai.azure.openai;

import com.azure.ai.openai.OpenAIClient;
import com.azure.ai.openai.OpenAIClientBuilder;
import com.azure.core.credential.AzureKeyCredential;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.azure.openai.AzureOpenAiChatClient;
import org.springframework.ai.azure.openai.AzureOpenAiChatOptions;
import org.springframework.ai.image.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Bean;
import org.springframework.util.StringUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(classes = AzureOpenAiImageClientIT.TestConfiguration.class)
@EnabledIfEnvironmentVariable(named = "AZURE_OPENAI_API_KEY", matches = ".+")
@EnabledIfEnvironmentVariable(named = "AZURE_OPENAI_ENDPOINT", matches = ".+")
public class AzureOpenAiImageClientIT {

	private static final Logger logger = LoggerFactory.getLogger(AzureOpenAiImageClientIT.class);

	@Autowired
	protected ImageClient imageClient;

	@Test
	void imageAsBase64Test() throws IOException {

		AzureOpenAiImageOptions imageOptions = AzureOpenAiImageOptions.builder()
			.withN(1)
			.withModel(AzureOpenAiImageOptions.ImageModel.DALL_E_3.getValue())
			.withResponseFormat("b64_json")
			.build();

		var instructions = """
				A light cream colored mini golden doodle.
				""";

		ImagePrompt imagePrompt = new ImagePrompt(instructions, imageOptions);

		ImageResponse imageResponse = this.imageClient.call(imagePrompt);

		ImageGeneration imageGeneration = imageResponse.getResult();
		Image image = imageGeneration.getOutput();

		assertThat(image.getB64Json()).isNotEmpty();

		writeFile(image);
	}

	@Test
	void imageAsUrlTest() {
		var options = ImageOptionsBuilder.builder().withHeight(1024).withWidth(1024).build();

		var instructions = """
				A light cream colored mini golden doodle with a sign that contains the message "I'm on my way to BARCADE!".""";

		ImagePrompt imagePrompt = new ImagePrompt(instructions, options);

		ImageResponse imageResponse = imageClient.call(imagePrompt);

		assertThat(imageResponse.getResults()).hasSize(1);

		ImageResponseMetadata imageResponseMetadata = imageResponse.getMetadata();
		assertThat(imageResponseMetadata.created()).isPositive();

		var generation = imageResponse.getResult();
		Image image = generation.getOutput();
		assertThat(image.getUrl()).isNotEmpty();
		logger.info(image.getUrl());
		assertThat(image.getB64Json()).isNull();
	}

	private static void writeFile(Image image) throws IOException {
		byte[] imageBytes = Base64.getDecoder().decode(image.getB64Json());
		String systemTempDir = System.getProperty("java.io.tmpdir");
		String filePath = systemTempDir + File.separator + "dog.png";
		File file = new File(filePath);
		logger.info("generated filed is {}", file);
		try (FileOutputStream fos = new FileOutputStream(file)) {
			fos.write(imageBytes);
		}
	}

	@SpringBootConfiguration
	public static class TestConfiguration {

		@Bean
		public OpenAIClient openAIClient() {
			String azureOpenaiApiKey = System.getenv("AZURE_OPENAI_API_KEY");
			return new OpenAIClientBuilder().credential(new AzureKeyCredential(azureOpenaiApiKey))
				.endpoint(System.getenv("AZURE_OPENAI_ENDPOINT"))
				.buildClient();
		}

	}

}
