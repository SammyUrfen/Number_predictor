const grid  = document.querySelector('.grid');
const maxIntensity = 1;
const gridSize = 28;
const pixels = [];
const pixelValues = new Array(gridSize*gridSize).fill(-1);

let isDrawing = false;

for (let i = 0; i < gridSize * gridSize; i++) {
    const pixel = document.createElement('div');
    pixel.className = 'pixel';
    pixel.dataset.index = i;
    grid.appendChild(pixel);
    pixels.push(pixel);
}

grid.addEventListener('mousedown', startDrawing);
grid.addEventListener('mouseover', draw);

document.addEventListener('mouseup', () => isDrawing = false);

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function draw(e) {
    if (!isDrawing) return;
    const pixel = e.target;
    updatePixel(pixel);  
}

function updatePixel(pixel) {
    const index = parseInt(pixel.dataset.index);
    const row = Math.floor(index/gridSize);
    const col = index%gridSize;
    for (let r = -1; r <= 1; r++) {
        for (let c = -1; c <= 1; c++) {
            const sideRow = row + r;
            const sideCol = col + c;
            if (sideRow >= 0 && sideRow < gridSize && sideCol >= 0 && sideCol < gridSize) {
                const sideIndex = sideRow * gridSize + sideCol;
                const distance = Math.sqrt(r*r + c*c);
                const intensity = maxIntensity - distance * 0.5;
                const normalizedIntensity = (intensity * 2) - 1; // Normalize between -1 and 1
                pixelValues[sideIndex] = Math.max(pixelValues[sideIndex], normalizedIntensity);
                const finalIntensity = Math.floor((pixelValues[sideIndex] + 1) * 127.5); // Convert to 0-255 range for display
                pixels[sideIndex].style.backgroundColor = `rgb(${finalIntensity}, ${finalIntensity}, ${finalIntensity})`;
            }
        }
    }
}


async function clearGrid() {
    for (let i = 0; i < pixelValues.length; i++) {
        pixelValues[i] = -1;
        pixels[i].style.backgroundColor = 'black';
        if (i % 28 === 0) {
            await sleep(10);
        }
    }
}

function sleep(time) {
    return new Promise((resolve) => setTimeout(resolve, time));
}

async function scanNumber() {
    const h = document.querySelector('.scanLine-h');
    const v = document.querySelector('.scanLine-v')
    h.style.display = 'block';
    v.style.display = 'block';
    await sleep(100);
    h.style.animation = 'scanDown 1s linear forwards';
    v.style.animation = 'scanRight 1s linear forwards';
    await sleep(1300);
    h.style.animation = 'scanUp 1s linear forwards';
    v.style.animation = 'scanLeft 1s linear forwards';
    await sleep(1300);
    h.style.display = 'none';
    v.style.display = 'none';
    h.style.animation = '';
    v.style.animation = '';
}


async function loadModel() {
    const session = await ort.InferenceSession.create('mnist_model.onnx');
    console.log('Model loaded:', session);
    return session;
}

let session;

async function initialize() {
    session = await loadModel();
}

initialize();
async function predict(session, pixelValues) {
    const input = new ort.Tensor('float32', pixelValues, [1, 784]);
    const feeds = { input: input };
    const results = await session.run(feeds);
    const output = results.output.data;
    const predictedNumber = output.indexOf(Math.max(...output));
    return predictedNumber;
}


async function getPixelValues() {
    scanNumber();
    await sleep(2600);
    let predictedNumber = await predict(session, pixelValues);
    if (predictedNumber === 10) {
        predictedNumber = " ?";
    }
    // console.log('Pixel values:', pixelValues);
    // Here you would add the actual prediction logic
    document.getElementById('overlay').style.display = 'block';
    document.getElementById('resultModal').style.display = 'block';
    document.getElementById('predictedNumber').textContent = ` ${predictedNumber}`; // Replace with actual prediction
}

function tryAgain() {
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('resultModal').style.display = 'none';
    clearGrid();
}

