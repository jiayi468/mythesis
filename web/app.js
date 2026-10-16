const MODEL_PATH = "/midterm/deployment/clip_b16_patch_head_b1.onnx";
const MEAN = [0.48145466, 0.4578275, 0.40821073];
const STD = [0.26862954, 0.26130258, 0.27577711];
const statusNode = document.querySelector("#status");
const imageInput = document.querySelector("#image");
const runButton = document.querySelector("#run");
const inputCanvas = document.querySelector("#input");
const heatmapCanvas = document.querySelector("#heatmap");
const overlayCanvas = document.querySelector("#overlay");
let session;
let loadedImage;
let referenceTensor;

function log(message) {
  statusNode.textContent += `\n${message}`;
}

document.querySelector("#load").addEventListener("click", async () => {
  statusNode.textContent = "检查 WebGPU...";
  if (!navigator.gpu) throw new Error("navigator.gpu 不可用；未创建 WASM 回退会话。 ");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("浏览器未返回 WebGPU adapter。");
  const started = performance.now();
  try {
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
    });
  } catch (error) {
    log(`WebGPU Session 创建失败：${error.stack || error}`);
    throw error;
  }
  log(`WebGPU Session 创建成功；模型加载 ${(performance.now() - started).toFixed(1)} ms`);
  log(`输入：${session.inputNames.join(", ")}；输出：${session.outputNames.join(", ")}`);
  imageInput.disabled = false;
  const samplePath = new URLSearchParams(location.search).get("sample");
  const tensorPath = new URLSearchParams(location.search).get("tensor");
  if (tensorPath) {
    const tensorResponse = await fetch(tensorPath);
    if (!tensorResponse.ok) throw new Error(`参考张量加载失败：HTTP ${tensorResponse.status}`);
    const values = new Float32Array(await tensorResponse.arrayBuffer());
    if (values.length !== 3 * 224 * 224) throw new Error(`参考张量长度错误：${values.length}`);
    referenceTensor = new ort.Tensor("float32", values, [1, 3, 224, 224]);
    log("使用 Python 预处理的固定 float32 参考张量。");
  }
  if (samplePath) {
    const response = await fetch(samplePath);
    if (!response.ok) throw new Error(`自动样本加载失败：HTTP ${response.status}`);
    loadedImage = await createImageBitmap(await response.blob());
    inputCanvas.getContext("2d").drawImage(loadedImage, 0, 0, 224, 224);
    runButton.disabled = false;
    runButton.click();
  }
});

imageInput.addEventListener("change", async () => {
  const file = imageInput.files[0];
  if (!file) return;
  loadedImage = await createImageBitmap(file);
  inputCanvas.getContext("2d").drawImage(loadedImage, 0, 0, 224, 224);
  runButton.disabled = false;
});

function inputTensor() {
  if (referenceTensor) return referenceTensor;
  const pixels = inputCanvas.getContext("2d").getImageData(0, 0, 224, 224).data;
  const chw = new Float32Array(3 * 224 * 224);
  for (let i = 0; i < 224 * 224; i += 1) {
    chw[i] = (pixels[4 * i] / 255 - MEAN[0]) / STD[0];
    chw[224 * 224 + i] = (pixels[4 * i + 1] / 255 - MEAN[1]) / STD[1];
    chw[2 * 224 * 224 + i] = (pixels[4 * i + 2] / 255 - MEAN[2]) / STD[2];
  }
  return new ort.Tensor("float32", chw, [1, 3, 224, 224]);
}

function colorize(values) {
  const tiny = document.createElement("canvas");
  tiny.width = 14; tiny.height = 14;
  const context = tiny.getContext("2d");
  const image = context.createImageData(14, 14);
  for (let i = 0; i < values.length; i += 1) {
    const value = Math.max(0, Math.min(1, values[i]));
    image.data[4 * i] = Math.round(255 * value);
    image.data[4 * i + 1] = Math.round(255 * (1 - Math.abs(2 * value - 1)));
    image.data[4 * i + 2] = Math.round(255 * (1 - value));
    image.data[4 * i + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  heatmapCanvas.getContext("2d").drawImage(tiny, 0, 0, 224, 224);
  const overlay = overlayCanvas.getContext("2d");
  overlay.globalAlpha = 1; overlay.drawImage(inputCanvas, 0, 0);
  overlay.globalAlpha = .45; overlay.drawImage(heatmapCanvas, 0, 0);
  overlay.globalAlpha = 1;
}

runButton.addEventListener("click", async () => {
  const tensor = inputTensor();
  const timings = [];
  let outputs;
  for (let i = 0; i < 11; i += 1) {
    const started = performance.now();
    outputs = await session.run({ image: tensor });
    timings.push(performance.now() - started);
  }
  colorize(outputs.anomaly_map.data);
  const warm = timings.slice(1).sort((a, b) => a - b);
  const percentile = p => warm[Math.ceil(p * warm.length) - 1];
  log(`异常分数：${outputs.image_score.data[0].toFixed(9)}`);
  log(`首次推理：${timings[0].toFixed(1)} ms；预热后 P50：${percentile(.5).toFixed(1)} ms；P95：${percentile(.95).toFixed(1)} ms；轮数：10`);
});

window.addEventListener("error", event => log(`错误：${event.error?.stack || event.message}`));

if (new URLSearchParams(location.search).has("autorun")) {
  document.querySelector("#load").click();
}
