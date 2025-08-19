import { bootstrap } from "global-agent";
import { promises as fs } from "node:fs";
import { listFiles, downloadFile } from "@huggingface/hub";

const BASE_DIR = "./tests/models/";

bootstrap();

export async function downloadModel(repo) {
  console.log(`Downloading model '${repo}'`);

  const fetch = await import("node-fetch");
  const modelName = repo.split("/")[1];
  const destDir = `${BASE_DIR}${modelName}`;

  await fs.mkdir(destDir, { recursive: true });

  const fileList = await listFiles({
    repo,
    fetch: fetch.default,
  });
  const fileNames = [];
  for await (const file of fileList) {
    fileNames.push(file.path);
  }

  for (const path of fileNames) {
    console.log(`Downloading file '${path}'`);
    const response = await downloadFile({
      repo,
      path,
      fetch: fetch.default,
    });
    const filename = `${destDir}/${path}`;

    await saveFile(filename, response);
    console.log(`File '${path}' downloaded`);
  }

  console.log(`Model '${repo}' downloaded`);
}

async function saveFile(file, response) {
  const arrayBuffer = await response.arrayBuffer();

  await fs.writeFile(file, Buffer.from(arrayBuffer));
}
