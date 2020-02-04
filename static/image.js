'use strict'

function drawImage(image, canvas, alpha) {
    const ctx = canvas.getContext('2d')
    ctx.globalAlpha = alpha
    ctx.drawImage(image,
        0, 0, canvasWidth, canvasHeight,
        0, 0, canvasWidth, canvasHeight)
    return ctx
}

function drawKeypoints(canvas) {
    const ctx = canvas.getContext('2d')
    ctx.lineWidth = 2
    ctx.globalAlpha = 0.7
    for (let i = 0; i < keypoints.length; i++) {
        const [x, y, size] = keypoints[i]
        ctx.beginPath()
        ctx.strokeStyle = '#0f0'
        ctx.arc(x, y, size, 0, 2 * Math.PI)
        ctx.stroke()
    }
}

function drawCanvas(canvas, alphaValue, doDrawKeypoints) {
    const srcImage = document.getElementById('sourceimage')
    const dstImage = document.getElementById('targetimage')
    drawImage(srcImage, canvas, 1.0)
    drawImage(dstImage, canvas, alphaValue)
    if (doDrawKeypoints) {
        drawKeypoints(canvas)
    }
}

function channelOverlay(canvas, alphaValue, doDrawKeypoints) {
    const srcImage = document.getElementById('targetimage')
    let ctx = drawImage(srcImage, canvas, 1.0)
    const img1 = ctx.getImageData(0, 0, canvasWidth, canvasHeight)
    const dstImage = document.getElementById('sourceimage')
    ctx = drawImage(dstImage, canvas, 1.0)
    const img2 = ctx.getImageData(0, 0, canvasWidth, canvasHeight)

    for (let i = 0; i < img1.data.length; i+=4) {
        img1.data[i] = img2.data[i]
    }
    for (let i = 1; i < img1.data.length; i+=4) {
        img1.data[i] = (img1.data[i] + img2.data[i]) / 2
    }
    ctx.putImageData(img1, 0, 0)
    if (doDrawKeypoints) {
        drawKeypoints(canvas)
    }
}

function updateCanvas() {
    const canvas = document.getElementById('overlay')
    const vizStyleButtons = document.getElementsByName('vizStyle')
    const slider = document.getElementById('alphaSlider')
    const drawFeaturesCheckbox = document.getElementById('featuresCheckbox')
    let drawFunction = undefined
    if (vizStyleButtons[0].checked) {
        drawFunction = () => {
            drawCanvas(canvas, parseInt(slider.value) / 4, drawFeaturesCheckbox.checked)
        }
    } else if (vizStyleButtons[1].checked) {
        drawFunction = () => {
            channelOverlay(canvas, parseInt(slider.value) / 4, drawFeaturesCheckbox.checked)
        }
    }
    drawFunction()
    return drawFunction
}

function overlayMain() {
    const drawFunction = updateCanvas()
    const vizStyleButtons = document.getElementsByName('vizStyle')
    vizStyleButtons[0].onchange = updateCanvas
    vizStyleButtons[1].onchange = updateCanvas
    const slider = document.getElementById('alphaSlider')
    slider.onchange = updateCanvas
    const drawFeaturesCheckbox = document.getElementById('featuresCheckbox')
    drawFeaturesCheckbox.onchange = updateCanvas
}