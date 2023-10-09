import fs from "fs";
import path from "path";
import { HfInference } from "@huggingface/inference";

const localImages = ["./c1.png", "./c2.png", "./c3.png", "./c4.png"];
const hf = new HfInference("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");

async function processImage(imagePath) {
  // Read the image from the local file system
  const imageBuffer = fs.readFileSync(path.resolve(imagePath));
  // Use the blob for inference
  const response = await hf.imageToText({
    data: imageBuffer,
    model: "Salesforce/blip-image-captioning-base",
  });

  return response;
}

// async function main() {
//   let scenes = [];

//   for (let img of localImages) {
//     const result = await processImage(img);
//     scenes.push(result.generated_text);
//     console.log(result.generated_text);

//     let last = scenes.slice(-1)[0];

//     const similar = await hf.sentenceSimilarity({
//       model: "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
//       inputs: {
//         source_sentence: last,
//         sentences: scenes,
//       },
//     });
//     // console.log(img, similar);
//   }
// }

async function main() {
  let scenes = [];
  let processedImages = [];

  for (let img of localImages) {
    const result = await processImage(img);
    const currentCaption = result.generated_text;
    console.log(currentCaption);

    if (scenes.length > 0) {
      const similar = await hf.sentenceSimilarity({
        model: "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        inputs: {
          source_sentence: currentCaption,
          sentences: scenes,
        },
      });

      // Check similarity scores
      let isDuplicate = false;
      for (let score of similar) {
        if (score > 0.85) {
          // Adjust this threshold as needed
          isDuplicate = true;
          break;
        }
      }

      if (isDuplicate) {
        continue; // Skip this image as it's similar to a previous one
      }
    }

    scenes.push(currentCaption);
    processedImages.push(img);
  }

  console.log("Unique images based on captions:", processedImages);
}

main().catch((error) => {
  console.error("Error processing images:", error);
});

// Docs: https://huggingface.co/tasks/image-to-text
// https://huggingface.co/docs/huggingface.js/index
// https://huggingface.co/docs/huggingface.js/inference/README
// https://observablehq.com/@huggingface/hello-huggingface-js-inference#ImageSegmentation
