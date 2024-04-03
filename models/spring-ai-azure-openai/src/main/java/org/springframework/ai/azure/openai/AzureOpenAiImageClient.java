package org.springframework.ai.azure.openai;

import static java.lang.String.format;

import java.util.List;

import com.azure.ai.openai.models.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.image.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.azure.ai.openai.OpenAIClient;

@Service
public class AzureOpenAiImageClient implements ImageClient {

	private static final Logger logger = LoggerFactory.getLogger(AzureOpenAiImageClient.class);

	@Autowired
	private final OpenAIClient openAIClient;

	public AzureOpenAiImageClient(OpenAIClient openAIClient) {
		this.openAIClient = openAIClient;
	}

	@Override
	public ImageResponse call(ImagePrompt imagePrompt) {
		if (imagePrompt.getInstructions().size() > 1) {
			throw new RuntimeException(format("implementation support 1 image instruction only, found %s",
					imagePrompt.getInstructions().size()));
		}
		if (imagePrompt.getInstructions().isEmpty()) {
			throw new RuntimeException("please provide image instruction, current is empty");
		}

		String instructions = imagePrompt.getInstructions().get(0).getText();

		var imageOptions = imagePrompt.getOptions();
		ImageGenerationOptions imageGenerationOptions = toOpenAiImageOptions(instructions, imageOptions);
		var model = imageOptions.getModel();
		if (model == null || model.isEmpty()) {
			model = AzureOpenAiImageOptions.DEFAULT_IMAGE_MODEL;
		}
		logger.info("image generation with model {} and options : {} ", model, imageOptions);
		var images = openAIClient.getImageGenerations(model, imageGenerationOptions);

		List<ImageGeneration> imageGenerations = images.getData().stream().map(entry -> {
			return new ImageGeneration(new Image(entry.getUrl(), entry.getBase64Data()));
		}).toList();

		return new ImageResponse(imageGenerations);
	}

	private ImageGenerationOptions toOpenAiImageOptions(String prompt, ImageOptions runtimeImageOptions) {
		ImageGenerationOptions imageGenerationOptions = new ImageGenerationOptions(prompt);
		if (runtimeImageOptions != null) {
			// Handle portable image options
			if (runtimeImageOptions.getN() != null) {
				imageGenerationOptions.setN(runtimeImageOptions.getN());
			}
			if (runtimeImageOptions.getModel() != null) {
				imageGenerationOptions.setModel(runtimeImageOptions.getModel());
			}
			if (runtimeImageOptions.getResponseFormat() != null) {
				// b64_json or url
				imageGenerationOptions.setResponseFormat(
						ImageGenerationResponseFormat.fromString(runtimeImageOptions.getResponseFormat()));
			}
			if (runtimeImageOptions.getWidth() != null && runtimeImageOptions.getHeight() != null) {
				imageGenerationOptions.setSize(
						ImageSize.fromString(runtimeImageOptions.getWidth() + "x" + runtimeImageOptions.getHeight()));
			}

			// Handle OpenAI specific image options
			if (runtimeImageOptions instanceof AzureOpenAiImageOptions runtimeAzureOpenAiImageOptions) {
				if (runtimeAzureOpenAiImageOptions.getQuality() != null) {
					imageGenerationOptions
						.setQuality(ImageGenerationQuality.fromString(runtimeAzureOpenAiImageOptions.getQuality()));
				}
				if (runtimeAzureOpenAiImageOptions.getStyle() != null) {
					imageGenerationOptions
						.setStyle(ImageGenerationStyle.fromString(runtimeAzureOpenAiImageOptions.getStyle()));
				}
				if (runtimeAzureOpenAiImageOptions.getUser() != null) {
					imageGenerationOptions.setUser(runtimeAzureOpenAiImageOptions.getUser());
				}
			}
		}
		return imageGenerationOptions;
	}

}
