/**
 * I would like to use WebGL here for comparing the two images since
 * the GPU can do this much faster, enabling nicer UI elements and
 * it should also be safer since I don't need to read the image buffer
 * directly in JavaScript.
 */

'use strict'

const vertexShaderSource = `
    attribute vec2 aVertex;
    attribute vec2 aUV;
    varying vec2 vTex;
    void main(void) {
        gl_Position = vec4(aVertex, 0.0, 1.0);
        vTex = aUV;
    }
`
const fragmentShaderSource = `
    precision lowp float;
    varying vec2 vTex;
    uniform sampler2D dstSampler;
    uniform sampler2D srcSampler;
    uniform float uRatio;
    float logistic_half(float x) {
        return 1.0 / (1.0+exp(-6.0 * (x-0.5)));
    }
    float logistic_quarter(float x) {
        return 1.0 / (1.0+exp(-12.0 * (x-0.25)));
    }
    void main(void) {
        vec2 texPos = vec2(vTex.x, vTex.y * -1. + 1.);
        vec4 dst = texture2D(dstSampler, texPos);
        vec4 src = texture2D(srcSampler, texPos);
        float ratio_h =  logistic_half(uRatio);
        float ratio_q =  logistic_quarter(uRatio);
        float nratio_q = logistic_quarter(1.0 - uRatio);
        gl_FragColor = vec4(
            ((1.0 - nratio_q) * src.z) + (nratio_q * dst.z),
            (ratio_h * src.y) + ((1.0 - ratio_h) * dst.y),
            (ratio_q * src.x) + ((1.0 - ratio_q) * dst.x),
            1.0);
    }
`

function overlayMain() {
    const canvas = document.getElementById('overlay')
    const gl = canvas.getContext('webgl')
    const shaderProgram = initShaderProgram(gl, vertexShaderSource, fragmentShaderSource)
    gl.useProgram(shaderProgram)
    gl.viewport(0, 0, canvasWidth, canvasHeight)
    const vertexBuffer = createStaticBuffer(gl, [-1, 1, -1, -1, 1, -1, 1, 1])
    const uvBuffer = createStaticBuffer(gl, [0, 1, 0, 0, 1, 0, 1, 1])
    // Bind first image to a sampler
    const dstSampler = gl.getUniformLocation(shaderProgram, 'dstSampler')
    gl.uniform1i(dstSampler, 0)
    gl.activeTexture(gl.TEXTURE0)
    bindTexture(gl, 'targetimage')
    // bind second image to another sampler
    const srcSampler = gl.getUniformLocation(shaderProgram, 'srcSampler')
    gl.uniform1i(srcSampler, 1)
    gl.activeTexture(gl.TEXTURE1)
    bindTexture(gl, 'sourceimage')
    // bind all the vertex buffers to their respective attributes
    bindBufferToVertexAttrib(gl, shaderProgram, vertexBuffer, 'aVertex')
    bindBufferToVertexAttrib(gl, shaderProgram, uvBuffer, 'aUV')
    // set color ratio
    const uRatio = gl.getUniformLocation(shaderProgram, 'uRatio')
    gl.uniform1f(uRatio, 1.0)
    // draw everything
    gl.drawArrays(gl.TRIANGLE_FAN, 0, 4)
    canvas.onmousemove = (mouseEvent) => {
        const mousePosition = getMousePosition(canvas, mouseEvent)
        gl.uniform1f(uRatio, mousePosition.x)
        gl.drawArrays(gl.TRIANGLE_FAN, 0, 4)
    }
}

/**
 * Returns the position of the mouse on the canvas.
 * The position is normalized, its always in range [0.0, 1.0].
 */
function getMousePosition(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (evt.clientX - rect.left) / canvasWidth,
        y: (evt.clientY - rect.top) / canvasHeight
    }
}

function bindBufferToVertexAttrib(gl, shaderProgram, buffer, attribName) {
    const attribute = gl.getAttribLocation(shaderProgram, attribName)
    gl.enableVertexAttribArray(attribute)
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
    gl.vertexAttribPointer(attribute, 2, gl.FLOAT, false, 0, 0)
}

function createStaticBuffer(gl, bufferDataList) {
    const buffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(bufferDataList), gl.STATIC_DRAW)
    return buffer
}

function bindTexture(gl, imageElementId) {
    const image = document.getElementById(imageElementId)
    const texture = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, image)
}

function initShaderProgram(gl, vsSource, fsSource) {
    const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vsSource)
    const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource)
    const shaderProgram = gl.createProgram()
    gl.attachShader(shaderProgram, vertexShader)
    gl.attachShader(shaderProgram, fragmentShader)
    gl.linkProgram(shaderProgram)
    return shaderProgram
}

function loadShader(gl, type, source) {
    const shader = gl.createShader(type)
    gl.shaderSource(shader, source)
    gl.compileShader(shader)
    return shader
}